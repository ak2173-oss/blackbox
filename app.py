# app.py
# Complete Flask app for transcription hub

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import json
import threading
import time
from pathlib import Path
from datetime import datetime
import subprocess
import logging
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('projects', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global status tracking
processing_status = {
    'status': 'idle',
    'current_step': '',
    'progress': 0,
    'message': 'Ready',
    'details': []
}


def update_status(status, step='', progress=0, message='', detail=''):
    """Update processing status"""
    global processing_status
    processing_status = {
        'status': status,
        'current_step': step,
        'progress': progress,
        'message': message,
        'details': processing_status.get('details', [])
    }
    if detail:
        processing_status['details'].append(f"{datetime.now().strftime('%H:%M:%S')} - {detail}")
        logging.info(f"Status: {step} - {detail}")


def load_projects():
    """Scan projects folder and load project metadata"""
    projects = []
    projects_dir = Path('projects')

    if not projects_dir.exists():
        return projects

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue

        project_name = project_dir.name

        # Skip invalid project names
        if project_name.startswith('.') or project_name.startswith('*'):
            continue

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
                    total_minutes = timing_data.get('total_time_seconds', 0) / 60
                    project_info['processing_time'] = f"{total_minutes:.1f} min"
            except:
                pass

        # Format creation date
        created_date = datetime.fromtimestamp(project_info['created'])
        project_info['created'] = created_date.strftime('%b %d, %Y at %I:%M %p')

        projects.append(project_info)

    # Sort by creation date (newest first)
    projects.sort(key=lambda x: x['created'], reverse=True)
    return projects


@app.route('/')
def index():
    """Main page - show all projects"""
    projects = load_projects()
    return render_template('index.html', projects=projects)

@app.route('/project/<project_name>')
@app.route('/project/<project_name>')
def view_project(project_name):
    """View a specific project"""
    project_path = Path('projects') / project_name

    if not project_path.exists():
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
            logging.error(f"Error loading transcript: {e}")

    # Load summary
    summary_file = project_path / 'summary' / 'summary.txt'
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = f.read()
        except Exception as e:
            logging.error(f"Error loading summary: {e}")

    # Load timing data
    timing_file = project_path / 'processing_time.json'
    if timing_file.exists():
        try:
            with open(timing_file, 'r', encoding='utf-8') as f:
                timing_raw = json.load(f)

                # Process timing data for template
                timing_data = {
                    'total_minutes': f"{timing_raw.get('total_time_seconds', 0) / 60:.1f}",
                    'audio_file': timing_raw.get('audio_file', 'Unknown'),
                    'processed_at': timing_raw.get('timestamp', datetime.now().isoformat()),
                    'stages': timing_raw.get('stage_timings', {})
                }
        except Exception as e:
            logging.error(f"Error loading timing data: {e}")

    return render_template('project.html',
                           project_name=project_name,
                           transcript=transcript,
                           summary=summary,
                           timing_data=timing_data)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload and process new audio file"""
    if request.method == 'POST':
        update_status('processing', 'Uploading', 10, 'Receiving file...')

        if 'audio' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file
        filename = file.filename
        filepath = Path('uploads') / filename

        try:
            file.save(str(filepath))
            update_status('processing', 'File saved', 15, 'Starting processing...', f"File: {filename}")

            # Process in background thread
            thread = threading.Thread(target=process_audio_background, args=(str(filepath),))
            thread.daemon = True
            thread.start()

            return jsonify({'success': True, 'message': 'Processing started'})

        except Exception as e:
            logging.error(f"Upload error: {e}")
            update_status('error', 'Upload failed', 0, str(e))
            return jsonify({'error': str(e)}), 500

    return render_template('upload.html')


def process_audio_background(filepath):
    """Process audio using optimized pipeline"""
    try:
        update_status('processing', 'Pipeline', 20, 'Starting optimized pipeline...')

        result = subprocess.run(
            ['python', 'optimized_pipeline.py', filepath],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            update_status('complete', 'Done', 100, 'Processing complete!')
        else:
            update_status('error', 'Failed', 0, 'Processing failed')

    except Exception as e:
        update_status('error', 'Failed', 0, str(e))

@app.route('/api/status')
def get_status():
    """Get current processing status"""
    return jsonify(processing_status)


@app.route('/api/ask_question', methods=['POST'])
def ask_question():
    """Answer questions about a specific transcript"""
    data = request.get_json()
    project_name = data.get('project')
    question = data.get('question')

    # Load the transcript
    transcript_file = Path('projects') / project_name / 'transcripts' / 'transcript_with_speakers.txt'

    if not transcript_file.exists():
        return jsonify({'error': 'Transcript not found'}), 404

    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = f.read()

    # More natural, conversational prompt
    prompt = f"""You are a helpful AI assistant. A user is asking you about a transcript from an audio recording. You can reference the transcript to answer their questions, but feel free to have a natural conversation.

Here's the transcript for reference:
---
{transcript[:8000]}
---

User question: {question}

Feel free to answer naturally. You can quote specific parts of the transcript, summarize sections, make observations, or just have a regular conversation about the content. Be helpful and conversational."""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b-instruct-q4_K_M",
                "prompt": prompt,
                "stream": False
            },
            timeout=30  # Reduced timeout to prevent hanging
        )

        if response.status_code == 200:
            answer = response.json()['response']
            return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        logging.error(f"Ollama error: {e}")

    return jsonify({'answer': 'Could not generate response'}), 500

@app.route('/download/<project_name>/<file_type>')
def download(project_name, file_type):
    """Download project files"""
    project_path = Path('projects') / project_name

    if file_type == 'transcript':
        file_path = project_path / 'transcripts' / 'transcript_with_speakers.txt'
    elif file_type == 'summary':
        file_path = project_path / 'summary' / 'summary.txt'
    elif file_type == 'audio':
        audio_files = list((project_path / 'audio').glob('*'))
        if audio_files:
            file_path = audio_files[0]
        else:
            return "Audio file not found", 404
    else:
        return "Invalid file type", 400

    if file_path.exists():
        return send_file(str(file_path), as_attachment=True)
    else:
        return "File not found", 404


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 TRANSCRIPTION HUB")
    print("=" * 60)
    print("\n📌 Open your browser to: http://localhost:5000")
    print("📝 Logs saved to: debug.log")
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=False, port=5000, threaded=True)