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

# USB Device monitoring
RECORDER_DRIVE = "/mnt/d"  # D: drive in WSL (your recorder)
RECORDER_FOLDER = "record"  # Folder name on recorder
SUPPORTED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.wma', '.aac'}

device_status = {
    'connected': False,
    'files': [],
    'last_check': None
}
device_lock = threading.Lock()

# Batch import state
batch_import_status = {
    'active': False,
    'total_files': 0,
    'completed_files': 0,
    'current_file': None,
    'files_status': [],  # List of {name, status, project_name}
    'estimated_remaining': None,
    'started_at': None
}
batch_lock = threading.Lock()

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


def unload_ollama_from_gpu():
    """Unload Ollama model from GPU to free VRAM for larger Whisper models"""
    global _ollama_preloaded

    logger.info(f"Unloading {Config.OLLAMA_MODEL} from GPU to free memory...")
    try:
        import requests
        import subprocess

        # Get GPU memory before unload
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            mem_before = result.stdout.strip() if result.returncode == 0 else "N/A"
        except:
            mem_before = "N/A"

        # Unload model by setting keep_alive to 0
        response = requests.post(
            f"{Config.OLLAMA_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": "",
                "keep_alive": 0  # Unload immediately
            },
            timeout=10
        )

        # Small delay for GPU to release memory
        import time
        time.sleep(0.5)

        # Get GPU memory after unload
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            mem_after = result.stdout.strip() if result.returncode == 0 else "N/A"
        except:
            mem_after = "N/A"

        if response.status_code == 200:
            _ollama_preloaded = False
            logger.info(f"✓ Ollama unloaded - GPU memory: {mem_before}MiB → {mem_after}MiB")
            return True, mem_before, mem_after
        else:
            logger.warning(f"⚠ Ollama unload returned status {response.status_code}")
            return False, mem_before, mem_after

    except Exception as e:
        logger.warning(f"⚠ Could not unload Ollama model: {e}")
        return False, "N/A", "N/A"


def reload_ollama_to_gpu():
    """Reload Ollama model to GPU after transcription"""
    logger.info(f"Reloading {Config.OLLAMA_MODEL} to GPU...")
    preload_ollama_model()


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


# ============================================================================
# USB Device Monitor (uses PowerShell for Windows USB drive access)
# ============================================================================

def check_recorder_device_powershell():
    """Check if recorder is connected using PowerShell (for USB drives in WSL2)"""
    global device_status
    import subprocess

    windows_path = f"{RECORDER_DRIVE[-1].upper()}:\\{RECORDER_FOLDER}"  # e.g., "D:\record"

    with device_lock:
        was_connected = device_status['connected']

        try:
            # Check if folder exists using PowerShell
            check_cmd = f'powershell.exe -c "Test-Path \'{windows_path}\'"'
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=5)

            if 'True' in result.stdout:
                # Folder exists - get file list (case-insensitive extension match)
                ps_script = f"Get-ChildItem '{windows_path}' -File | Where-Object {{ $_.Extension -imatch '\\.(wav|mp3|m4a|flac|ogg|wma|aac)$' }} | Select-Object Name, Length, LastWriteTime | ConvertTo-Json"
                result = subprocess.run(
                    ['powershell.exe', '-Command', ps_script],
                    capture_output=True, text=True, timeout=10
                )

                files = []
                if result.stdout.strip():
                    import json as json_module
                    try:
                        file_data = json_module.loads(result.stdout)
                        # Handle single file (dict) vs multiple files (list)
                        if isinstance(file_data, dict):
                            file_data = [file_data]

                        for f in file_data:
                            name = f.get('Name', '')
                            size = f.get('Length', 0)
                            # Parse PowerShell date format /Date(timestamp)/
                            date_str = f.get('LastWriteTime', '')
                            try:
                                if '/Date(' in str(date_str):
                                    ts = int(str(date_str).split('(')[1].split(')')[0]) / 1000
                                    modified = datetime.fromtimestamp(ts).strftime('%d-%b-%Y %H:%M')
                                else:
                                    modified = str(date_str)[:16]
                            except:
                                modified = 'Unknown'

                            files.append({
                                'name': name,
                                'path': f"{windows_path}\\{name}",
                                'size': size,
                                'size_mb': round(size / (1024 * 1024), 1),
                                'modified': modified
                            })
                    except json_module.JSONDecodeError:
                        logger.error(f"Failed to parse file list: {result.stdout[:200]}")

                # Sort by modified date (newest first)
                files.sort(key=lambda x: x['modified'], reverse=True)

                device_status['connected'] = True
                device_status['files'] = files
                device_status['last_check'] = datetime.now().isoformat()

                if not was_connected and files:
                    logger.info(f"Recorder connected! Found {len(files)} audio files")
            else:
                device_status['connected'] = False
                device_status['files'] = []
                device_status['last_check'] = datetime.now().isoformat()

                if was_connected:
                    logger.info("Recorder disconnected")

        except subprocess.TimeoutExpired:
            logger.warning("Device check timed out")
        except Exception as e:
            logger.error(f"Device check error: {e}")


def check_recorder_device():
    """Check if the audio recorder is connected (tries WSL path first, then PowerShell)"""
    global device_status

    recorder_path = Path(RECORDER_DRIVE) / RECORDER_FOLDER

    # First try direct WSL path (faster if mounted)
    if recorder_path.exists() and recorder_path.is_dir():
        with device_lock:
            was_connected = device_status['connected']
            files = []
            for f in recorder_path.iterdir():
                if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                    try:
                        stat = f.stat()
                        files.append({
                            'name': f.name,
                            'path': str(f),
                            'size': stat.st_size,
                            'size_mb': round(stat.st_size / (1024 * 1024), 1),
                            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%d-%b-%Y %H:%M')
                        })
                    except Exception as e:
                        logger.error(f"Error reading file {f}: {e}")

            files.sort(key=lambda x: x['modified'], reverse=True)
            device_status['connected'] = True
            device_status['files'] = files
            device_status['last_check'] = datetime.now().isoformat()

            if not was_connected and files:
                logger.info(f"Recorder connected (WSL path)! Found {len(files)} audio files")
    else:
        # Fall back to PowerShell method for USB drives
        check_recorder_device_powershell()


def device_monitor_thread():
    """Background thread to monitor for USB recorder"""
    logger.info("Starting USB device monitor (checking D:/record)")
    while True:
        try:
            check_recorder_device()
        except Exception as e:
            logger.error(f"Device monitor error: {e}")
        time.sleep(3)  # Check every 3 seconds


def start_device_monitor():
    """Start the device monitor background thread"""
    monitor_thread = threading.Thread(target=device_monitor_thread, daemon=True)
    monitor_thread.start()
    logger.info("USB device monitor started")


def process_batch_import(files_to_import, whisper_model=None):
    """Process batch import of files from recorder"""
    global batch_import_status

    with batch_lock:
        batch_import_status = {
            'active': True,
            'total_files': len(files_to_import),
            'completed_files': 0,
            'current_file': None,
            'current_step': 'Starting...',
            'current_progress': 0,
            'current_detail': '',
            'files_status': [{
                'name': f['name'],
                'status': 'pending',
                'project_name': None
            } for f in files_to_import],
            'estimated_remaining': None,
            'started_at': time.time(),
            'whisper_model': whisper_model or Config.WHISPER_MODEL
        }

    # Estimate ~5 seconds per MB of audio (more realistic for GPU transcription)
    total_size_mb = sum(f['size_mb'] for f in files_to_import)

    for i, file_info in enumerate(files_to_import):
        with batch_lock:
            batch_import_status['current_file'] = file_info['name']
            batch_import_status['current_step'] = 'Copying from device...'
            batch_import_status['current_progress'] = 0
            batch_import_status['current_detail'] = f"Transferring {file_info['size_mb']} MB"
            batch_import_status['files_status'][i]['status'] = 'copying'

            # Update time estimate based on actual performance
            elapsed = time.time() - batch_import_status['started_at']
            if i > 0:
                avg_time_per_file = elapsed / i
                remaining_files = len(files_to_import) - i
                batch_import_status['estimated_remaining'] = int(avg_time_per_file * remaining_files)
            else:
                # Initial estimate: ~5 seconds per MB
                remaining_mb = sum(f['size_mb'] for f in files_to_import[i:])
                batch_import_status['estimated_remaining'] = int(remaining_mb * 5)

        try:
            # Copy file to uploads folder
            source_path = file_info['path']
            dest_path = Path(Config.UPLOAD_FOLDER) / file_info['name']

            # Use PowerShell for Windows paths (USB drives), shutil for WSL paths
            if '\\' in source_path:
                # Windows path - use PowerShell to copy
                import subprocess
                # Convert WSL dest path to Windows path
                wsl_to_win = subprocess.run(
                    ['wslpath', '-w', str(dest_path)],
                    capture_output=True, text=True
                )
                win_dest = wsl_to_win.stdout.strip()
                copy_cmd = f'''powershell.exe -c "Copy-Item -Path '{source_path}' -Destination '{win_dest}'"'''
                result = subprocess.run(copy_cmd, shell=True, capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    raise Exception(f"PowerShell copy failed: {result.stderr}")
                logger.info(f"Copied {file_info['name']} to uploads (via PowerShell)")
            else:
                # WSL path - use shutil
                import shutil
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied {file_info['name']} to uploads")

            with batch_lock:
                batch_import_status['current_step'] = 'Processing audio...'
                batch_import_status['files_status'][i]['status'] = 'processing'

            # Process with pipeline
            job_id = generate_job_id(file_info['name'])

            def status_callback(job_id, status='processing', step='', progress=0, message='', detail=''):
                update_job_status(job_id, status, step, progress, message, detail)
                # Also update batch status with current step details
                with batch_lock:
                    batch_import_status['current_step'] = step
                    batch_import_status['current_progress'] = progress
                    batch_import_status['current_detail'] = message

            # Unload Ollama before transcription
            unload_ollama_from_gpu()

            # Use selected whisper model - temporarily set config
            selected_model = whisper_model or Config.WHISPER_MODEL
            original_model = Config.WHISPER_MODEL
            Config.WHISPER_MODEL = selected_model

            pipeline = AudioPipeline(
                dest_path,
                job_id=job_id,
                status_callback=status_callback,
                whisper_model=None,  # Let pipeline load from Config
                reload_ollama_callback=reload_ollama_to_gpu
            )

            project_dir = pipeline.run()
            project_name = Path(project_dir).name

            # Restore original config
            Config.WHISPER_MODEL = original_model

            # Clean up upload file
            dest_path.unlink(missing_ok=True)

            with batch_lock:
                batch_import_status['files_status'][i]['status'] = 'complete'
                batch_import_status['files_status'][i]['project_name'] = project_name
                batch_import_status['completed_files'] = i + 1

            logger.info(f"Batch import: Completed {file_info['name']} -> {project_name}")

        except Exception as e:
            logger.error(f"Batch import error for {file_info['name']}: {e}")
            # Restore config on error too
            try:
                Config.WHISPER_MODEL = original_model
            except:
                pass
            with batch_lock:
                batch_import_status['files_status'][i]['status'] = 'error'
                batch_import_status['files_status'][i]['error'] = str(e)
                batch_import_status['completed_files'] = i + 1

    with batch_lock:
        batch_import_status['active'] = False
        batch_import_status['current_file'] = None
        batch_import_status['estimated_remaining'] = 0

    logger.info(f"Batch import complete: {batch_import_status['completed_files']}/{batch_import_status['total_files']} files")


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
                'duration': None
            }

            # Check for transcript
            transcript_file = project_dir / 'transcripts' / 'transcript_with_speakers.txt'
            if transcript_file.exists():
                project_info['has_transcript'] = True

            # Check for summary
            summary_file = project_dir / 'summary' / 'summary.txt'
            if summary_file.exists():
                project_info['has_summary'] = True

            # Get audio duration from transcript (last segment's end time)
            transcript_json = project_dir / 'transcripts' / 'transcript.json'
            if transcript_json.exists():
                try:
                    with open(transcript_json, 'r') as f:
                        transcript_data = json.load(f)
                        if transcript_data:
                            last_segment = transcript_data[-1]
                            total_seconds = last_segment.get('end', 0)
                            hours = int(total_seconds // 3600)
                            minutes = int((total_seconds % 3600) // 60)
                            seconds = int(total_seconds % 60)
                            project_info['duration'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                except Exception as e:
                    logger.error(f"Error loading duration for {project_name}: {e}")

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

    # Check for audio file
    audio_path = project_path / 'audio'
    has_audio = False
    audio_filename = None
    if audio_path.exists():
        audio_files = list(audio_path.glob('*'))
        if audio_files:
            has_audio = True
            audio_filename = audio_files[0].name

    # Load all projects for sidebar
    all_projects = load_projects()

    return render_template('project.html',
                           project_name=project_name,
                           transcript=transcript,
                           summary=summary,
                           timing_data=timing_data,
                           all_projects=all_projects,
                           has_audio=has_audio,
                           audio_filename=audio_filename)


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

        # Get selected transcription engine (default to whisper)
        engine = request.form.get('engine', 'whisper').lower()
        if engine not in ['whisper', 'parakeet']:
            engine = 'whisper'

        # Get selected Whisper model size (if using Whisper)
        whisper_model = None
        if engine == 'whisper':
            whisper_model = request.form.get('whisper_model', Config.WHISPER_MODEL)
            # Validate whisper model
            valid_models = ['tiny.en', 'base.en', 'small.en', 'medium.en', 'large-v2']
            if whisper_model not in valid_models:
                whisper_model = Config.WHISPER_MODEL

        # Generate job ID
        job_id = generate_job_id(filename)

        # Save uploaded file
        filepath = Path(Config.UPLOAD_FOLDER) / f"{job_id}_{filename}"

        try:
            file.save(str(filepath))
            logger.info(f"File uploaded: {filename} (job: {job_id}, engine: {engine}, whisper_model: {whisper_model})")

            # Initialize job status
            if engine == 'whisper':
                engine_name = f"Whisper ({whisper_model})"
            else:
                engine_name = "Wav2Vec2"
            update_job_status(job_id, 'processing', 'Upload', 10, 'File uploaded, starting processing...', f"File: {filename}, Engine: {engine_name}")

            # Process in background thread with selected engine and model
            thread = threading.Thread(
                target=process_audio_background,
                args=(str(filepath), job_id, engine, whisper_model),
                daemon=True
            )
            thread.start()

            return jsonify({'success': True, 'job_id': job_id, 'message': 'Processing started'})

        except Exception as e:
            logger.error(f"Upload error: {e}")
            update_job_status(job_id, 'error', 'Upload failed', 0, str(e))
            return jsonify({'error': str(e)}), 500

    # GET method - render upload page with LLM info
    return render_template('upload.html', llm_model=Config.OLLAMA_MODEL)


def process_audio_background(filepath, job_id, engine='whisper', whisper_model=None):
    """Process audio using pipeline (runs in background thread)"""
    try:
        # Temporarily override engine for this job
        original_engine = Config.TRANSCRIPTION_ENGINE
        original_whisper_model = Config.WHISPER_MODEL

        Config.TRANSCRIPTION_ENGINE = engine
        if whisper_model:
            Config.WHISPER_MODEL = whisper_model

        engine_name = "Wav2Vec2" if engine == 'parakeet' else f"Whisper ({whisper_model or Config.WHISPER_MODEL})"
        update_job_status(job_id, 'processing', 'Initializing', 15, f'Starting pipeline with {engine_name}...')

        # Check if we need to unload Ollama for larger Whisper models
        MODELS_REQUIRING_UNLOAD = ['small.en', 'medium.en', 'large-v2']
        needs_unload = engine == 'whisper' and whisper_model in MODELS_REQUIRING_UNLOAD

        # Create reload callback for pipeline to use before summary
        def reload_callback():
            if needs_unload:
                update_job_status(
                    job_id, 'processing', 'Memory Management', 85,
                    '⏳ Reloading LLM for summary generation...',
                    f'Loading {Config.OLLAMA_MODEL} back into GPU'
                )
                reload_ollama_to_gpu()
                logger.info(f"Ollama model {Config.OLLAMA_MODEL} reloaded for summary")

        # Create pipeline - use cached model if available (loads on first use, not at startup)
        pipeline = AudioPipeline(
            audio_file=filepath,
            job_id=job_id,
            status_callback=update_job_status,
            whisper_model=get_whisper_model(),  # Cached after first load
            reload_ollama_callback=reload_callback if needs_unload else None
        )

        # Log memory management steps
        if needs_unload:
            update_job_status(
                job_id, 'processing', 'Memory Management', 18,
                '⏳ Unloading LLM to free GPU memory for larger Whisper model...',
                f'Temporarily unloading {Config.OLLAMA_MODEL} to free ~5GB VRAM'
            )
            pipeline.add_external_debug('Memory Management', 'Unloading Ollama LLM', f'Model: {Config.OLLAMA_MODEL}')
            success, mem_before, mem_after = unload_ollama_from_gpu()
            if success:
                pipeline.add_external_debug('Memory Management', 'Ollama unloaded successfully', f'GPU: {mem_before}MiB → {mem_after}MiB')
            else:
                pipeline.add_external_debug('Memory Management', 'Ollama unload skipped', 'Ollama may not be running')

        # Run pipeline (Ollama reloads inside pipeline before summary if needed)
        project_dir = pipeline.run()

        # Ollama stays loaded after summary generation - no unloading

        # Restore original settings
        Config.TRANSCRIPTION_ENGINE = original_engine
        Config.WHISPER_MODEL = original_whisper_model

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


def save_chat_message(project_name, role, content, quotes=None):
    """Save a chat message to the project's chat history"""
    try:
        chat_file = Path(Config.PROJECTS_FOLDER) / project_name / 'chat_history.json'

        # Load existing history
        history = []
        if chat_file.exists():
            with open(chat_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

        # Add new message
        message = {
            'timestamp': datetime.now().isoformat(),
            'role': role,  # 'user' or 'assistant'
            'content': content
        }
        if quotes:
            message['quotes'] = quotes

        history.append(message)

        # Save back
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"Chat message saved to {project_name}")
    except Exception as e:
        logger.error(f"Failed to save chat message: {e}")


@app.route('/api/chat_history/<project_name>')
def get_chat_history(project_name):
    """Get chat history for a project"""
    project_name = secure_filename(project_name)
    chat_file = Path(Config.PROJECTS_FOLDER) / project_name / 'chat_history.json'

    if chat_file.exists():
        with open(chat_file, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    return jsonify([])


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

        # Use FULL transcript for Q&A - qwen2.5 supports up to 128K context
        # Estimate: 1 token ~= 4 characters
        max_context = Config.OLLAMA_CONTEXT_LENGTH * 4  # 32K tokens = ~128K chars
        transcript = full_transcript[:max_context]

        # Log transcript size
        if len(full_transcript) > max_context:
            logger.warning(f"Transcript truncated: {len(full_transcript)} -> {max_context} chars")
            context_note = f"[Note: Using first {max_context:,} of {len(full_transcript):,} characters due to context limit]"
        else:
            logger.info(f"Using FULL transcript: {len(transcript):,} characters")
            context_note = f"[Using COMPLETE transcript: {len(transcript):,} characters]"

        # Check if it's a greeting or casual message first
        casual_greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy', 'sup', 'yo', 'thanks', 'thank you', 'bye', 'goodbye']
        question_lower = question.lower().strip()

        if question_lower in casual_greetings or (len(question.split()) <= 3 and any(greeting in question_lower for greeting in casual_greetings)):
            # Respond to casual greetings
            greeting_response = "Hello! 👋 I'm your transcript analysis assistant. I can help you find information from this conversation.\n\nTry asking questions like:\n• What is this conversation about?\n• Who are the speakers?\n• What decisions were made?\n• What action items were mentioned?\n\nWhat would you like to know?"
            # Save greeting exchange
            save_chat_message(project_name, 'user', question)
            save_chat_message(project_name, 'assistant', greeting_response)
            return jsonify({'answer': greeting_response})

        # Create prompt for Qwen2.5
        prompt = f"""Answer the question based ONLY on this transcript. Be direct and concise.

TRANSCRIPT:
{transcript}

QUESTION: {question}

INSTRUCTIONS:
- Search the ENTIRE transcript carefully before answering
- If names, numbers, or specific details are mentioned, include them
- Quote relevant parts when helpful
- If something is NOT in the transcript, say "Not mentioned in the transcript"
- Be direct - no unnecessary suggestions or analysis tips

ANSWER:"""

        # Send to Ollama
        import requests
        response = requests.post(
            f"{Config.OLLAMA_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_ctx": Config.OLLAMA_CONTEXT_LENGTH
                }
            },
            timeout=Config.OLLAMA_TIMEOUT
        )

        if response.status_code == 200:
            answer = response.json().get('response', 'No response generated')

            # Always extract relevant quotes from the transcript that were used to generate the answer
            # Extract keywords from the question (ignore common words)
            common_words = {'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were',
                          'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
                          'with', 'from', 'about', 'like', 'do', 'does', 'did', 'can', 'could', 'would',
                          'should', 'me', 'you', 'some', 'give', 'show', 'tell'}

            keywords = [word for word in question_lower.split()
                       if len(word) > 3 and word not in common_words][:8]

            # Find relevant quotes from transcript
            quotes_used = []
            transcript_lines = transcript.split('\n')

            # Score each line by keyword relevance
            scored_lines = []
            for line in transcript_lines:
                if len(line.strip()) < 50:  # Skip very short lines
                    continue

                line_lower = line.lower()
                score = sum(1 for kw in keywords if kw in line_lower)

                if score > 0:
                    scored_lines.append((score, line.strip()))

            # Sort by score and take top 3-5 quotes
            scored_lines.sort(reverse=True, key=lambda x: x[0])
            quotes_used = [line for score, line in scored_lines[:5] if score >= 2]

            # If no high-scoring quotes found, take top 3 with any keyword match
            if not quotes_used and scored_lines:
                quotes_used = [line for score, line in scored_lines[:3]]

            # Save conversation to chat history
            save_chat_message(project_name, 'user', question)
            save_chat_message(project_name, 'assistant', answer.strip(), quotes_used if quotes_used else None)

            return jsonify({
                'answer': answer.strip(),
                'quotes_used': quotes_used if quotes_used else []
            })
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


@app.route('/audio/<project_name>')
def stream_audio(project_name):
    """Stream audio file for playback"""
    project_name = secure_filename(project_name)
    project_path = Path(Config.PROJECTS_FOLDER) / project_name / 'audio'

    if not project_path.exists():
        return "Audio not found", 404

    # Find audio file
    audio_files = list(project_path.glob('*'))
    if not audio_files:
        return "No audio file found", 404

    audio_file = audio_files[0]
    return send_file(str(audio_file), mimetype='audio/mpeg')


# ============================================================================
# USB Device / Batch Import API
# ============================================================================

@app.route('/api/device/status')
def get_device_status():
    """Check if recorder device is connected"""
    with device_lock:
        return jsonify({
            'connected': device_status['connected'],
            'file_count': len(device_status['files']),
            'last_check': device_status['last_check']
        })


@app.route('/api/device/scan', methods=['POST'])
def scan_for_device():
    """Manually trigger device scan"""
    check_recorder_device()
    with device_lock:
        return jsonify({
            'connected': device_status['connected'],
            'file_count': len(device_status['files']),
            'files': device_status['files']
        })


@app.route('/api/device/files')
def get_device_files():
    """Get list of audio files on the recorder"""
    with device_lock:
        return jsonify({
            'connected': device_status['connected'],
            'files': device_status['files']
        })


@app.route('/api/device/import', methods=['POST'])
def start_batch_import():
    """Start importing selected files from recorder"""
    data = request.get_json()
    selected_files = data.get('files', [])
    whisper_model = data.get('whisper_model', Config.WHISPER_MODEL)

    # Validate whisper model
    valid_models = ['tiny.en', 'base.en', 'small.en', 'medium.en', 'large-v2']
    if whisper_model not in valid_models:
        whisper_model = Config.WHISPER_MODEL

    if not selected_files:
        return jsonify({'error': 'No files selected'}), 400

    # Check if already importing
    with batch_lock:
        if batch_import_status['active']:
            return jsonify({'error': 'Import already in progress'}), 409

    # Get full file info for selected files
    with device_lock:
        files_to_import = [f for f in device_status['files'] if f['name'] in selected_files]

    if not files_to_import:
        return jsonify({'error': 'Selected files not found on device'}), 404

    # Start import in background thread
    import_thread = threading.Thread(
        target=process_batch_import,
        args=(files_to_import, whisper_model),
        daemon=True
    )
    import_thread.start()

    return jsonify({
        'status': 'started',
        'total_files': len(files_to_import),
        'whisper_model': whisper_model
    })


@app.route('/api/device/import/status')
def get_import_status():
    """Get current batch import status"""
    with batch_lock:
        return jsonify(batch_import_status.copy())


@app.route('/api/device/import/cancel', methods=['POST'])
def cancel_import():
    """Cancel the current batch import (will finish current file)"""
    # Note: This is a soft cancel - current file will finish
    with batch_lock:
        if batch_import_status['active']:
            # Mark remaining pending files as cancelled
            for f in batch_import_status['files_status']:
                if f['status'] == 'pending':
                    f['status'] = 'cancelled'
            return jsonify({'status': 'cancelling'})
        return jsonify({'status': 'no_import_active'})


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

    # Display transcription engine info
    if Config.TRANSCRIPTION_ENGINE.lower() == 'parakeet':
        print(f" Transcription: Wav2Vec2 ({Config.PARAKEET_MODEL})")
        print(f" Device: {'cuda' if device == 'cuda' else 'cpu'}")
    else:
        print(f" Transcription: Whisper {Config.WHISPER_MODEL} ({device}/{compute_type})")

    print(f" LLM Model: {Config.OLLAMA_MODEL}")
    print(f" Logs: {Config.LOG_FILE}")

    # Display tips based on engine
    if Config.TRANSCRIPTION_ENGINE.lower() == 'parakeet':
        print("\n Tip: Wav2Vec2 uses Facebook's model for GPU-accelerated transcription")
    else:
        print("\n Tip: Whisper model loads on-demand for each transcription")
    print(" Tip: Use Server-Sent Events for real-time progress updates")

    # No preloading - models load on demand
    print("\n Models will load on-demand (no preloading)")

    # Start USB device monitor
    print(f" USB Monitor: Watching {RECORDER_DRIVE}/{RECORDER_FOLDER}")
    start_device_monitor()

    print("\nPress Ctrl+C to stop\n")

    app.run(debug=False, port=5000, threaded=True)