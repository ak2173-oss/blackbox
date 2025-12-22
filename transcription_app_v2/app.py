# app.py - Transcription App v2
# Improved Flask app with better architecture, security, and performance

from flask import Flask, render_template, request, jsonify, send_file, Response
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False
import os
import json
import threading
import time
import re
from pathlib import Path
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
from queue import Queue
import hashlib

# Import config
from config import Config, validate_config, get_device_config

# Import pipeline
from pipeline import AudioPipeline

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS if configured and available
if HAS_CORS and Config.CORS_ENABLED:
    CORS(app, origins=Config.CORS_ORIGINS)

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Validate configuration on startup
validate_config()
device, compute_type = get_device_config()

# Global state management (TODO: move to Redis/DB for production)
processing_jobs = {}
processing_lock = threading.Lock()

# Cached Whisper model (loaded once, reused)
_whisper_model = None
_model_lock = threading.Lock()

# Ollama preload status
_ollama_preloaded = False
_ollama_lock = threading.Lock()


def get_whisper_model():
    """Get cached Whisper model, loading if necessary"""
    global _whisper_model

    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:  # Double-check locking
                logger.info(f"Loading Whisper model: {Config.WHISPER_MODEL}")
                from faster_whisper import WhisperModel

                device, compute = get_device_config()
                _whisper_model = WhisperModel(
                    Config.WHISPER_MODEL,
                    device=device,
                    compute_type=compute,
                    cpu_threads=Config.WHISPER_CPU_THREADS,
                    num_workers=Config.WHISPER_NUM_WORKERS
                )
                logger.info("✓ Whisper model loaded and cached")

    return _whisper_model


def preload_ollama_model():
    """Preload Qwen2.5 model into GPU memory on startup"""
    global _ollama_preloaded

    if _ollama_preloaded:
        return

    with _ollama_lock:
        if _ollama_preloaded:  # Double-check locking
            return

        logger.info(f"Preloading Ollama model: {Config.OLLAMA_MODEL} into GPU...")
        try:
            import requests

            # Send a simple warm-up prompt to load model into GPU
            response = requests.post(
                f"{Config.OLLAMA_URL}/api/generate",
                json={
                    "model": Config.OLLAMA_MODEL,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": Config.OLLAMA_CONTEXT_LENGTH
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                _ollama_preloaded = True
                logger.info(f"✓ Ollama model {Config.OLLAMA_MODEL} preloaded and ready")
            else:
                logger.warning("⚠ Ollama model preload failed - will load on first use")

        except Exception as e:
            logger.warning(f"⚠ Could not preload Ollama model: {e}")
            logger.info("  Model will load on first transcription request")


def sanitize_filename(filename):
    """Sanitize uploaded filename to prevent path traversal attacks"""
    # Remove path components
    filename = os.path.basename(filename)
    # Secure the filename
    filename = secure_filename(filename)
    # Remove any remaining dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    return filename


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def generate_job_id(filename):
    """Generate unique job ID"""
    timestamp = str(time.time())
    return hashlib.md5(f"{filename}{timestamp}".encode()).hexdigest()[:12]


def update_job_status(job_id, status='processing', step='', progress=0, message='', detail='', project_name=None):
    """Update job status in thread-safe manner"""
    with processing_lock:
        if job_id not in processing_jobs:
            processing_jobs[job_id] = {
                'status': 'idle',
                'step': '',
                'progress': 0,
                'message': '',
                'details': [],
                'start_time': time.time(),
                'project_name': None
            }

        job = processing_jobs[job_id]
        job['status'] = status
        job['step'] = step
        job['progress'] = progress
        job['message'] = message
        job['elapsed_time'] = int(time.time() - job['start_time'])

        if project_name:
            job['project_name'] = project_name

        if detail:
            timestamp = datetime.now().strftime('%H:%M:%S')
            job['details'].append(f"[{timestamp}] {detail}")
            # Keep only last 50 details
            job['details'] = job['details'][-50:]
            logger.info(f"[{job_id}] {step}: {detail}")


def load_projects():
    """Scan projects folder and load project metadata"""
    projects = []
    projects_dir = Path(Config.PROJECTS_FOLDER)

    if not projects_dir.exists():
        return projects

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name

        # Skip invalid project names
        if project_name.startswith('.'):
            continue

        try:
            project_info = {
                'name': project_name,
                'created': project_dir.stat().st_mtime,
                'has_transcript': False,
                'has_summary': False,
                'processing_time': None
            }

            # Check for transcript
            transcript_file = project_dir / 'transcripts' / 'transcript_with_speakers.txt'
            if transcript_file.exists():
                project_info['has_transcript'] = True

            # Check for summary
            summary_file = project_dir / 'summary' / 'summary.txt'
            if summary_file.exists():
                project_info['has_summary'] = True

            # Load processing time
            timing_file = project_dir / 'processing_time.json'
            if timing_file.exists():
                try:
                    with open(timing_file, 'r') as f:
                        timing_data = json.load(f)
                        total_seconds = timing_data.get('total_seconds', 0)
                        project_info['processing_time'] = f"{total_seconds / 60:.1f} min"
                except Exception as e:
                    logger.error(f"Error loading timing for {project_name}: {e}")

            # Format creation date
            created_date = datetime.fromtimestamp(project_info['created'])
            project_info['created'] = created_date.strftime('%b %d, %Y at %I:%M %p')

            projects.append(project_info)
        except Exception as e:
            logger.error(f"Error loading project {project_name}: {e}")

    # Sort by creation date (newest first)
    projects.sort(key=lambda x: x['created'], reverse=True)
    return projects


@app.route('/')
def index():
    """Main page - show all projects"""
    projects = load_projects()
    return render_template('index.html', projects=projects)


@app.route('/project/<project_name>')
def view_project(project_name):
    """View a specific project"""
    # Sanitize project name to prevent path traversal
    project_name = secure_filename(project_name)
    project_path = Path(Config.PROJECTS_FOLDER) / project_name

    if not project_path.exists() or not project_path.is_dir():
        return "Project not found", 404

    # Initialize empty data
    transcript = []
    summary = ""
    timing_data = None

    # Load transcript with speakers
    transcript_file = project_path / 'transcripts' / 'transcript_with_speakers.json'
    if transcript_file.exists():
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
        except Exception as e:
            logger.error(f"Error loading transcript: {e}")

    # Load summary
    summary_file = project_path / 'summary' / 'summary.txt'
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
        except Exception as e:
            logger.error(f"Error loading summary: {e}")

    # Load timing data
    timing_file = project_path / 'processing_time.json'
    if timing_file.exists():
        try:
            with open(timing_file, 'r', encoding='utf-8') as f:
                timing_raw = json.load(f)
                timing_data = {
                    'total_minutes': f"{timing_raw.get('total_seconds', 0) / 60:.1f}",
                    'audio_file': timing_raw.get('audio_file', 'Unknown'),
                    'processed_at': timing_raw.get('processed_at', datetime.now().isoformat()),
                    'stages': timing_raw.get('stages', {})
                }
        except Exception as e:
            logger.error(f"Error loading timing data: {e}")

    return render_template('project.html',
                           project_name=project_name,
                           transcript=transcript,
                           summary=summary,
                           timing_data=timing_data)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload and process new audio file"""
    if request.method == 'POST':
        if 'audio' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(Config.ALLOWED_EXTENSIONS)}'}), 400

        # Sanitize filename
        filename = sanitize_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400

        # Generate job ID
        job_id = generate_job_id(filename)

        # Save uploaded file
        filepath = Path(Config.UPLOAD_FOLDER) / f"{job_id}_{filename}"

        try:
            file.save(str(filepath))
            logger.info(f"File uploaded: {filename} (job: {job_id})")

            # Initialize job status
            update_job_status(job_id, 'processing', 'Upload', 10, 'File uploaded, starting processing...', f"File: {filename}")

            # Process in background thread
            thread = threading.Thread(
                target=process_audio_background,
                args=(str(filepath), job_id),
                daemon=True
            )
            thread.start()

            return jsonify({'success': True, 'job_id': job_id, 'message': 'Processing started'})

        except Exception as e:
            logger.error(f"Upload error: {e}")
            update_job_status(job_id, 'error', 'Upload failed', 0, str(e))
            return jsonify({'error': str(e)}), 500

    return render_template('upload.html')


def process_audio_background(filepath, job_id):
    """Process audio using pipeline (runs in background thread)"""
    try:
        update_job_status(job_id, 'processing', 'Initializing', 15, 'Starting pipeline...')

        # Create pipeline with cached model
        pipeline = AudioPipeline(
            audio_file=filepath,
            job_id=job_id,
            status_callback=update_job_status,
            whisper_model=get_whisper_model()
        )

        # Run pipeline
        project_dir = pipeline.run()

        # Extract project name from path
        project_name = Path(project_dir).name

        # Update final status
        update_job_status(
            job_id,
            'complete',
            'Complete',
            100,
            'Processing complete!',
            f"Project saved: {project_name}",
            project_name=project_name
        )

        # Clean up upload file
        try:
            Path(filepath).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not delete upload file: {e}")

    except Exception as e:
        logger.error(f"Processing error for job {job_id}: {e}", exc_info=True)
        update_job_status(job_id, 'error', 'Failed', 0, f"Processing failed: {str(e)}")


@app.route('/api/status/<job_id>')
def get_job_status(job_id):
    """Get status of a specific job"""
    with processing_lock:
        if job_id in processing_jobs:
            return jsonify(processing_jobs[job_id])
        else:
            return jsonify({'error': 'Job not found'}), 404


@app.route('/api/status/stream/<job_id>')
def stream_job_status(job_id):
    """Server-Sent Events stream for job status updates"""
    def generate():
        last_update = None
        timeout = 300  # 5 minutes timeout
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                yield f"data: {json.dumps({'error': 'Timeout'})}\n\n"
                break

            with processing_lock:
                if job_id in processing_jobs:
                    current_status = processing_jobs[job_id].copy()

                    # Only send if status changed
                    if current_status != last_update:
                        yield f"data: {json.dumps(current_status)}\n\n"
                        last_update = current_status

                    # Stop streaming if job is complete or errored
                    if current_status['status'] in ['complete', 'error']:
                        break
                else:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break

            time.sleep(0.5)  # Check every 500ms

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/ask_question', methods=['POST'])
def ask_question():
    """Answer questions about a specific transcript using Qwen2.5"""
    data = request.get_json()
    project_name = data.get('project')
    question = data.get('question')

    if not project_name or not question:
        return jsonify({'error': 'Missing project or question'}), 400

    # Sanitize inputs
    project_name = secure_filename(project_name)
    question = question.strip()[:500]  # Limit question length

    # Load the transcript
    transcript_file = Path(Config.PROJECTS_FOLDER) / project_name / 'transcripts' / 'transcript_with_speakers.txt'

    if not transcript_file.exists():
        return jsonify({'error': 'Transcript not found'}), 404

    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            full_transcript = f.read()

        # Use retrieval for Q&A based on context length
        # Qwen2.5:7b-instruct supports 8K context (configurable)
        # Estimate: 1 token ~= 4 characters, so 8K tokens ~= 32K characters
        max_context = min(Config.OLLAMA_CONTEXT_LENGTH * 4, 100000)
        transcript = full_transcript[:max_context]

        # Log if transcript was truncated
        if len(full_transcript) > max_context:
            logger.warning(f"Transcript truncated: {len(full_transcript)} -> {max_context} chars")
            context_note = f"[Using first {max_context:,} characters of {len(full_transcript):,} total]"
        else:
            logger.info(f"Using full transcript: {len(transcript):,} characters")
            context_note = f"[Using complete transcript: {len(transcript):,} characters]"

        # Check if it's a greeting or casual message first
        casual_greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy', 'sup', 'yo', 'thanks', 'thank you', 'bye', 'goodbye']
        question_lower = question.lower().strip()

        if question_lower in casual_greetings or (len(question.split()) <= 3 and any(greeting in question_lower for greeting in casual_greetings)):
            # Respond to casual greetings
            return jsonify({
                'answer': f"Hello! 👋 I'm your transcript analysis assistant. I can help you find information from this conversation.\n\nTry asking questions like:\n• What is this conversation about?\n• Who are the speakers?\n• What decisions were made?\n• What action items were mentioned?\n\nWhat would you like to know?"
            })

        # Create prompt for Qwen2.5 with chain-of-thought reasoning
        prompt = f"""You are a highly accurate AI assistant analyzing a meeting transcript. Answer questions using ONLY information from the transcript below.

{context_note}

TRANSCRIPT:
{transcript}

===

USER QUESTION: {question}

===

INSTRUCTIONS:

1. Search the transcript for information relevant to the question
2. If found, extract exact quotes with speaker names
3. Provide a clear answer based on the quotes
4. If NOT found, respond: "This information is not discussed in the transcript."

OUTPUT FORMAT:

📎 EVIDENCE FROM TRANSCRIPT:
• "exact quote..." — Speaker Name
[Only include if relevant quotes exist]

💬 ANSWER:
[Clear answer based on evidence, or state it's not in the transcript]

===

CRITICAL RULES:
- NEVER make up or infer information not in the transcript
- ALWAYS use exact quotes (not paraphrases)
- If you're unsure, say so
- Quality over quantity - be thorough in your search"""

        # Send to Ollama
        import requests
        response = requests.post(
            f"{Config.OLLAMA_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused answers
                    "top_p": 0.9,
                    "top_k": 40
                }
            },
            timeout=Config.OLLAMA_TIMEOUT
        )

        if response.status_code == 200:
            answer = response.json().get('response', 'No response generated')
            return jsonify({'answer': answer})
        else:
            logger.error(f"Ollama error: {response.status_code}")
            return jsonify({'error': 'Could not generate response'}), 500

    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timed out'}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Could not connect to Ollama. Make sure it is running.'}), 503
    except Exception as e:
        logger.error(f"Question answering error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/download/<project_name>/<file_type>')
def download(project_name, file_type):
    """Download project files"""
    # Sanitize inputs
    project_name = secure_filename(project_name)
    project_path = Path(Config.PROJECTS_FOLDER) / project_name

    if not project_path.exists() or not project_path.is_dir():
        return "Project not found", 404

    # Determine file path based on type
    if file_type == 'transcript':
        file_path = project_path / 'transcripts' / 'transcript_with_speakers.txt'
    elif file_type == 'summary':
        file_path = project_path / 'summary' / 'summary.txt'
    elif file_type == 'audio':
        audio_files = list((project_path / 'audio').glob('*'))
        file_path = audio_files[0] if audio_files else None
    else:
        return "Invalid file type", 400

    if file_path and file_path.exists():
        return send_file(str(file_path), as_attachment=True)
    else:
        return "File not found", 404


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'whisper_model': Config.WHISPER_MODEL,
        'ollama_model': Config.OLLAMA_MODEL,
        'device': device,
        'compute_type': compute_type
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" TRANSCRIPTION HUB V2")
    print("=" * 70)
    print(f"\n Server: http://localhost:5000")
    print(f" Whisper Model: {Config.WHISPER_MODEL} ({device}/{compute_type})")
    print(f" LLM Model: {Config.OLLAMA_MODEL}")
    print(f" Logs: {Config.LOG_FILE}")
    print("\n Tip: The Whisper model is cached in memory for faster processing")
    print(" Tip: Use Server-Sent Events for real-time progress updates")

    # Preload Qwen2.5 model into GPU for instant AI summaries
    print(f"\n⏳ Preloading {Config.OLLAMA_MODEL} into GPU...")
    preload_ollama_model()

    print("\nPress Ctrl+C to stop\n")

    app.run(debug=False, port=5000, threaded=True)