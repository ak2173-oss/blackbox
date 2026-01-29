# Transcription Hub v2 - Architecture Reference

**Version:** 2.0
**Last Updated:** 2025-09-30
**Purpose:** Complete code reference showing file relationships, data flow, and system architecture

---

## Table of Contents
1. [File Structure](#file-structure)
2. [File Dependencies](#file-dependencies)
3. [Data Flow](#data-flow)
4. [Key Components](#key-components)
5. [API Endpoints](#api-endpoints)
6. [Database Schema](#database-schema)
7. [Configuration](#configuration)
8. [External Dependencies](#external-dependencies)

---

## File Structure

```
transcription_app_v2/
├── app.py                           # Flask web application (515 lines)
├── pipeline.py                      # Audio processing pipeline (450 lines)
├── config.py                        # Configuration management (120 lines)
├── setup.py                         # Dependency validator (215 lines)
├── requirements.txt                 # Python dependencies
├── start.bat                        # Windows startup script
├── stop.bat                         # Windows shutdown script
├── .env.example                     # Environment template
├── .env                             # Environment config (created from .env.example)
│
├── templates/                       # HTML templates
│   ├── index.html                  # Project listing page (237 lines)
│   ├── upload.html                 # Upload interface (365 lines)
│   └── project.html                # Project viewer + chat (412 lines)
│
├── static/                          # Static assets (CSS, JS)
├── uploads/                         # Temporary upload storage
│
├── projects/                        # Processed projects
│   └── [project_name]_[timestamp]/
│       ├── audio/                  # Original audio file
│       ├── transcripts/            # JSON and text transcripts
│       │   ├── transcript.json
│       │   ├── transcript.txt
│       │   ├── transcript_with_speakers.json
│       │   └── transcript_with_speakers.txt
│       ├── summary/                # AI-generated summaries
│       │   ├── summary.json
│       │   └── summary.txt
│       ├── processing_time.json    # Timing report
│       └── processing_time.txt     # Human-readable timing
│
└── Documentation/
    ├── README.md                    # User documentation
    ├── QUICKSTART.md               # Quick start guide
    ├── IMPROVEMENTS.md             # Version history
    └── ARCHITECTURE.md             # This file
```

---

## File Dependencies

### Import Graph

```
app.py
├── imports: flask, flask_cors, threading, pathlib, logging, werkzeug
├── imports: config.py
│   └── Config class
│   └── validate_config()
│   └── get_device_config()
├── imports: pipeline.py
│   └── AudioPipeline class
└── imports: faster_whisper
    └── WhisperModel (cached globally)

pipeline.py
├── imports: subprocess, requests, pathlib, json, shutil
├── imports: config.py
│   └── Config class
│   └── get_device_config()
└── uses: FFmpeg (external binary)
    uses: Ollama API (external service)

config.py
├── imports: os, pathlib
├── imports: dotenv (optional)
└── reads: .env file

setup.py
├── imports: sys, subprocess, shutil, pathlib, requests
└── validates: all dependencies

templates/*.html
├── references: app.py routes
├── uses: Jinja2 templating
└── fetches: API endpoints via JavaScript
```

### Call Graph

```
User Browser
    ↓
app.py Routes
    ↓
┌────────────────────────────────────┐
│  /upload (POST)                    │
│    ↓                               │
│  save file to uploads/             │
│    ↓                               │
│  threading.Thread()                │
│    ↓                               │
│  process_audio_background()        │
│    ↓                               │
│  pipeline.AudioPipeline()          │
│    ↓                               │
│  pipeline.run()                    │
│    ├─ create_project_folder()     │
│    ├─ transcribe()                │
│    │   └─ FFmpeg → WhisperModel   │
│    ├─ add_speakers()              │
│    ├─ generate_summary()          │
│    │   └─ Ollama API (Phi-3)     │
│    └─ save_timing_report()        │
│                                    │
│  update_job_status() [callbacks]  │
│    ↓                               │
│  SSE stream to browser             │
└────────────────────────────────────┘
```

---

## Data Flow

### Upload → Processing Pipeline

```
1. User uploads file via templates/upload.html
   ↓
2. POST /upload (app.py:261-309)
   - Validates file type
   - Sanitizes filename
   - Generates job_id
   - Saves to uploads/
   ↓
3. Background thread starts: process_audio_background() (app.py:312-351)
   - Creates AudioPipeline instance
   - Passes cached WhisperModel
   - Sets status_callback
   ↓
4. AudioPipeline.run() (pipeline.py:421-439)
   ├─ create_project_folder() (pipeline.py:55-78)
   │  - Creates projects/[name]_[timestamp]/
   │  - Creates subdirs: audio/, transcripts/, summary/
   │  - Copies audio file
   │
   ├─ transcribe() (pipeline.py:79-192)
   │  - Converts to WAV with FFmpeg
   │  - Runs faster-whisper transcription
   │  - Saves transcript.json and transcript.txt
   │
   ├─ add_speakers() (pipeline.py:193-242)
   │  - Analyzes gaps between segments
   │  - Assigns speakers (Speaker 0, Speaker 1)
   │  - Merges consecutive segments
   │  - Saves transcript_with_speakers.json/txt
   │
   ├─ generate_summary() (pipeline.py:243-378)
   │  - Sends transcript to Ollama
   │  - Gets title from Phi-3
   │  - Gets summary from Phi-3
   │  - Saves summary.json and summary.txt
   │  - Renames project folder with title
   │
   └─ save_timing_report() (pipeline.py:379-420)
      - Saves processing_time.json
      - Saves processing_time.txt
   ↓
5. Status updates sent via callback: update_job_status() (app.py:106-136)
   - Updates processing_jobs dict
   - Adds timestamped details
   ↓
6. Browser receives updates via SSE: /api/status/stream/<job_id> (app.py:363-394)
   - Streams JSON status updates
   - Frontend updates progress bar
   ↓
7. Completion: Redirects to /project/<name>
```

### Project Viewing

```
1. User clicks "View" on templates/index.html
   ↓
2. GET /project/<project_name> (app.py:206-259)
   - Loads transcript_with_speakers.json
   - Loads summary.txt
   - Loads processing_time.json
   - Renders templates/project.html
   ↓
3. User asks question in chat interface
   ↓
4. POST /api/ask_question (app.py:397-462)
   - Loads transcript_with_speakers.txt
   - Creates prompt with context
   - Sends to Ollama Phi-3
   - Returns answer to frontend
```

---

## Key Components

### app.py

**Purpose:** Flask web server, route handlers, job management

**Key Functions:**

| Function | Line | Purpose |
|----------|------|---------|
| `get_whisper_model()` | 60-81 | Lazy-loads and caches WhisperModel |
| `sanitize_filename()` | 83-92 | Prevents path traversal attacks |
| `allowed_file()` | 94-98 | Validates file extensions |
| `generate_job_id()` | 100-104 | Creates unique 12-char job ID |
| `update_job_status()` | 106-136 | Thread-safe status updates |
| `load_projects()` | 138-197 | Scans projects/ and returns metadata |
| `process_audio_background()` | 312-351 | Background processing thread |

**Routes:**

| Route | Method | Purpose | Template |
|-------|--------|---------|----------|
| `/` | GET | List all projects | index.html |
| `/project/<name>` | GET | View project details | project.html |
| `/upload` | GET, POST | Upload interface & handler | upload.html |
| `/api/status/<job_id>` | GET | Get job status (JSON) | - |
| `/api/status/stream/<job_id>` | GET | SSE stream for status | - |
| `/api/ask_question` | POST | Q&A with Phi-3 | - |
| `/download/<project>/<type>` | GET | Download files | - |
| `/health` | GET | System health check | - |

**Global State:**

```python
processing_jobs = {}           # job_id → status dict
processing_lock = threading.Lock()  # Thread safety
_whisper_model = None         # Cached Whisper model
_model_lock = threading.Lock() # Model loading lock
```

**Job Status Structure:**

```python
{
    'status': 'idle' | 'processing' | 'complete' | 'error',
    'step': str,              # Current step name
    'progress': int,          # 0-100
    'message': str,           # Status message
    'details': [str],         # Timestamped log entries
    'start_time': float,      # Unix timestamp
    'elapsed_time': int,      # Seconds
    'project_name': str       # Final project folder name
}
```

---

### pipeline.py

**Purpose:** Audio processing pipeline with transcription, speaker detection, summarization

**Class: AudioPipeline**

**Constructor:**
```python
def __init__(self, audio_file, job_id=None, status_callback=None, whisper_model=None)
```
- `audio_file`: Path to input audio
- `job_id`: Unique identifier for status tracking
- `status_callback`: Function to call with status updates
- `whisper_model`: Pre-loaded WhisperModel (for caching)

**Key Methods:**

| Method | Line | Purpose |
|--------|------|---------|
| `update_status()` | 36-47 | Calls status_callback with progress |
| `log_time()` | 49-54 | Records stage completion time |
| `create_project_folder()` | 55-78 | Creates project directory structure |
| `transcribe()` | 79-192 | FFmpeg + Whisper transcription |
| `add_speakers()` | 193-242 | Speaker diarization |
| `generate_summary()` | 243-378 | Ollama Phi-3 summarization |
| `save_timing_report()` | 379-420 | Saves performance metrics |
| `run()` | 421-439 | Main pipeline orchestrator |

**Pipeline Stages:**

```python
run()
├─ [20-25%]  create_project_folder()
├─ [30-70%]  transcribe()
│   ├─ [30-35%]  FFmpeg conversion
│   ├─ [40-50%]  Whisper model loading
│   └─ [50-70%]  Segment processing
├─ [72-75%]  add_speakers()
├─ [80-95%]  generate_summary()
│   ├─ [82-85%]  Title generation
│   └─ [87-95%]  Summary generation
└─ [98-100%] save_timing_report()
```

**External Calls:**

```python
# FFmpeg for audio conversion
subprocess.run(["ffmpeg", "-y", "-i", input, "-ar", "16000", "-ac", "1", output])

# Whisper for transcription
segments, info = whisper_model.transcribe(
    audio_file,
    beam_size=1,
    vad_filter=True,
    vad_parameters={...}
)

# Ollama for AI summary
requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "phi3:latest", "prompt": prompt}
)
```

---

### config.py

**Purpose:** Centralized configuration with environment variable support

**Class: Config**

Configuration loaded from environment variables or defaults:

| Variable | Default | Source | Usage |
|----------|---------|--------|-------|
| `WHISPER_MODEL` | `base.en` | ENV | app.py:67, pipeline.py:128 |
| `WHISPER_DEVICE` | `auto` | ENV | config.py:65, pipeline.py:114 |
| `WHISPER_COMPUTE_TYPE` | `auto` | ENV | config.py:68, pipeline.py:114 |
| `WHISPER_CPU_THREADS` | CPU count | ENV | pipeline.py:92, pipeline.py:132 |
| `OLLAMA_URL` | `http://localhost:11434` | ENV | app.py:434, pipeline.py:273 |
| `OLLAMA_MODEL` | `phi3:latest` | ENV | app.py:436, pipeline.py:276 |
| `OLLAMA_TIMEOUT` | 180 | ENV | app.py:445, pipeline.py:326 |
| `SPEAKER_GAP_THRESHOLD` | 2.0 | ENV | pipeline.py:203 |
| `SPEAKER_MERGE_THRESHOLD` | 1.0 | ENV | pipeline.py:212 |
| `MAX_CONTENT_LENGTH` | 500MB | ENV | app.py:26 |
| `LOG_LEVEL` | `INFO` | ENV | app.py:38 |

**Functions:**

```python
get_device_config() → (device, compute_type)
    # Returns: ('cuda', 'float16') or ('cpu', 'int8')
    # Auto-detects GPU via torch.cuda.is_available()
    # Called by: app.py:49, pipeline.py:114

validate_config() → None
    # Creates directories, validates Ollama connection
    # Called by: app.py:48, setup.py
```

---

### templates/

**Template Inheritance & Data Flow:**

#### index.html (Line 1-237)

**Receives from app.py:**
```python
render_template('index.html', projects=projects)
```

**Data Structure:**
```python
projects = [
    {
        'name': str,              # Folder name
        'created': str,           # Formatted date
        'has_transcript': bool,
        'has_summary': bool,
        'processing_time': str    # "5.2 min"
    }
]
```

**JavaScript:**
- `filterProjects()`: Search functionality

**Links to:**
- `/upload` - Upload button
- `/project/<name>` - View button
- `/download/<name>/transcript` - Download transcript
- `/download/<name>/summary` - Download summary

---

#### upload.html (Line 1-365)

**Features:**
- Drag & drop file upload
- Real-time progress via Server-Sent Events
- Fallback to polling if SSE fails

**Data Flow:**
```javascript
// 1. File upload
POST /upload
    FormData { audio: File }
    ↓
    Response { success: true, job_id: "abc123..." }

// 2. Status streaming
EventSource → GET /api/status/stream/abc123
    ↓
    data: { status, step, progress, message, details, elapsed_time }
    ↓
    Update UI

// 3. Completion
On status === 'complete':
    Show buttons → /project/<name>
```

**JavaScript Functions:**
- `handleFile(file)`: Validates and uploads
- `uploadFile(file)`: POST to /upload
- `startStatusStreaming(jobId)`: Opens SSE connection
- `startStatusPolling(jobId)`: Fallback polling

---

#### project.html (Line 1-412)

**Receives from app.py:**
```python
render_template('project.html',
    project_name=str,
    transcript=[{speaker, start, end, text}, ...],
    summary=str,
    timing_data={total_minutes, audio_file, processed_at, stages}
)
```

**Sections:**
1. **Header** - Project name, download buttons
2. **Main Content** - AI summary, transcript segments
3. **Sidebar:**
   - Chat interface (Q&A with Phi-3)
   - Processing time breakdown
   - Project files status

**Chat Interface:**
```javascript
// User asks question
POST /api/ask_question
    { project: "project_name", question: "..." }
    ↓
    Response { answer: "..." }
    ↓
    Display in chat
```

**JavaScript Functions:**
- `sendMessage()`: Sends question to API
- `addMessage(content, type)`: Adds message to chat
- `addThinkingMessage()`: Shows loading indicator

---

## API Endpoints

### GET /

**Handler:** `index()` (app.py:199-204)
**Returns:** HTML page with project list
**Data:** Calls `load_projects()` to scan projects/

---

### GET /project/<project_name>

**Handler:** `view_project()` (app.py:206-259)
**Parameters:**
- `project_name` (path) - Sanitized folder name

**Process:**
1. Sanitize project_name with `secure_filename()`
2. Load `transcript_with_speakers.json`
3. Load `summary/summary.txt`
4. Load `processing_time.json`
5. Render template

**Returns:** HTML page with transcript, summary, timing

---

### GET /upload

**Handler:** `upload()` (app.py:261-309)
**Returns:** Upload interface HTML

---

### POST /upload

**Handler:** `upload()` (app.py:261-309)
**Content-Type:** multipart/form-data
**Body:** `audio` file

**Process:**
1. Validate file exists and has allowed extension
2. Sanitize filename
3. Generate job_id with `generate_job_id()`
4. Save to uploads/
5. Start background thread `process_audio_background()`
6. Return job_id

**Returns:**
```json
{
    "success": true,
    "job_id": "abc123456789",
    "message": "Processing started"
}
```

**Error Response:**
```json
{
    "error": "Error message"
}
```

---

### GET /api/status/<job_id>

**Handler:** `get_job_status()` (app.py:353-361)
**Returns:** Current job status (JSON)

```json
{
    "status": "processing",
    "step": "Transcription",
    "progress": 45,
    "message": "Processing segments...",
    "details": ["[12:34:56] Starting...", "..."],
    "elapsed_time": 23,
    "project_name": null
}
```

---

### GET /api/status/stream/<job_id>

**Handler:** `stream_job_status()` (app.py:363-394)
**Returns:** Server-Sent Events stream
**Content-Type:** text/event-stream

**Event Format:**
```
data: {"status":"processing","progress":45,...}

data: {"status":"complete","progress":100,...}
```

**Lifecycle:**
- Checks status every 500ms
- Sends updates only when status changes
- Closes when status is 'complete' or 'error'
- Timeout after 5 minutes

---

### POST /api/ask_question

**Handler:** `ask_question()` (app.py:397-462)
**Content-Type:** application/json

**Request:**
```json
{
    "project": "project_name",
    "question": "What was discussed about AI?"
}
```

**Process:**
1. Sanitize project name
2. Load transcript (max 10,000 chars)
3. Create prompt for Phi-3
4. Send to Ollama API
5. Return answer

**Response:**
```json
{
    "answer": "The discussion covered..."
}
```

**Error Responses:**
```json
{"error": "Missing project or question"}  // 400
{"error": "Transcript not found"}          // 404
{"error": "Could not generate response"}   // 500
{"error": "Request timed out"}            // 504
{"error": "Could not connect to Ollama"}  // 503
```

---

### GET /download/<project_name>/<file_type>

**Handler:** `download()` (app.py:464-489)
**Parameters:**
- `project_name` (path)
- `file_type` (path) - `transcript`, `summary`, or `audio`

**Returns:** File download

**File Mapping:**
- `transcript` → `transcripts/transcript_with_speakers.txt`
- `summary` → `summary/summary.txt`
- `audio` → `audio/*` (first file)

---

### GET /health

**Handler:** `health()` (app.py:491-501)
**Returns:** System status (JSON)

```json
{
    "status": "healthy",
    "whisper_model": "base.en",
    "ollama_model": "phi3:latest",
    "device": "cuda",
    "compute_type": "float16"
}
```

---

## Database Schema

**Storage:** File-based (no database)

### Project Structure

```
projects/<project_name>_<timestamp>/
├── audio/
│   └── <original_filename>              # Original audio file
│
├── transcripts/
│   ├── transcript.json                  # Raw segments
│   │   Format: [{"start": 0.0, "end": 5.2, "text": "..."}]
│   │
│   ├── transcript.txt                   # Plain text (no speakers)
│   │   Format: "text text text..."
│   │
│   ├── transcript_with_speakers.json    # With speaker labels
│   │   Format: [{"start": 0.0, "end": 5.2, "text": "...", "speaker": "Speaker 0"}]
│   │
│   └── transcript_with_speakers.txt     # Human-readable
│       Format:
│       [Speaker 0]
│       Text here...
│
│       [Speaker 1]
│       More text...
│
├── summary/
│   ├── summary.json                     # Structured summary
│   │   {
│   │       "title": "Meeting Title",
│   │       "summary": "Full summary text...",
│   │       "model": "phi3:latest",
│   │       "generated_at": "2025-09-30T12:00:00"
│   │   }
│   │
│   └── summary.txt                      # Plain text summary
│
├── processing_time.json
│   {
│       "total_seconds": 123.4,
│       "total_minutes": 2.1,
│       "stages": {
│           "folder_creation": 0.5,
│           "audio_conversion": 2.3,
│           "transcription": 45.2,
│           "speaker_detection": 1.2,
│           "title_generation": 3.4,
│           "summary_generation": 12.8
│       },
│       "audio_file": "meeting.mp3",
│       "processed_at": "2025-09-30T12:00:00",
│       "config": {
│           "whisper_model": "base.en",
│           "ollama_model": "phi3:latest",
│           "device": "cuda"
│       }
│   }
│
└── processing_time.txt                  # Human-readable timing
```

---

## Configuration

### Environment Variables (.env)

**Loaded by:** config.py:9-13
**Used by:** config.py:23-63

```bash
# Flask
SECRET_KEY=dev-secret-key-change-in-production
MAX_UPLOAD_SIZE=524288000                # 500MB in bytes

# Whisper
WHISPER_MODEL=base.en                    # tiny.en, base.en, small.en, medium.en, large-v2
WHISPER_DEVICE=auto                      # auto, cuda, cpu
WHISPER_COMPUTE_TYPE=auto                # auto, float16, int8
WHISPER_CPU_THREADS=8                    # Number of CPU threads
WHISPER_NUM_WORKERS=4                    # Parallel workers

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3:latest
OLLAMA_TIMEOUT=180                       # Seconds
OLLAMA_TEMPERATURE=0.7                   # 0.0-1.0

# Speaker Detection
SPEAKER_GAP_THRESHOLD=2.0                # Seconds of silence → new speaker
SPEAKER_MERGE_THRESHOLD=1.0              # Merge segments < 1s apart

# Logging
LOG_LEVEL=INFO                           # DEBUG, INFO, WARNING, ERROR
LOG_FILE=app.log

# Security
CORS_ENABLED=False
CORS_ORIGINS=*
```

### Startup Scripts

#### start.bat (Windows)

**Location:** transcription_app_v2/start.bat
**Purpose:** Start application on Windows

**Process:**
1. Check for venv/ (virtual environment)
2. Activate if exists
3. Check for .env file
4. Copy from .env.example if missing
5. Run `python app.py`

---

#### stop.bat (Windows)

**Location:** transcription_app_v2/stop.bat
**Purpose:** Stop all related processes

**Process:**
1. Kill all python.exe processes
2. Kill ollama.exe processes
3. Check GPU memory status with nvidia-smi

---

## External Dependencies

### Required Services

#### 1. FFmpeg

**Used by:** pipeline.py:86-100
**Purpose:** Audio format conversion
**Command:**
```bash
ffmpeg -y -i input.mp3 -ar 16000 -ac 1 output.wav
```

**Installation:**
- Windows: `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `apt install ffmpeg`

---

#### 2. Ollama + Phi-3

**Used by:** pipeline.py:272-327, app.py:432-446
**Purpose:** AI summarization and Q&A
**API Endpoint:** `http://localhost:11434/api/generate`

**Request Format:**
```json
{
    "model": "phi3:latest",
    "prompt": "...",
    "stream": false,
    "options": {
        "temperature": 0.7
    }
}
```

**Response Format:**
```json
{
    "response": "Generated text..."
}
```

**Installation:**
```bash
# Install Ollama
# Visit: https://ollama.com/download

# Pull Phi-3 model
ollama pull phi3:latest

# Start Ollama
ollama serve
```

---

### Python Packages (requirements.txt)

| Package | Version | Used By | Purpose |
|---------|---------|---------|---------|
| `faster-whisper` | >=0.10.0 | pipeline.py:68, app.py:68 | GPU-accelerated Whisper |
| `ffmpeg-python` | >=0.2.0 | pipeline.py | FFmpeg Python wrapper |
| `Flask` | >=3.0.0 | app.py | Web framework |
| `Flask-CORS` | >=4.0.0 | app.py:6,34 | Cross-origin requests |
| `torch` | >=2.0.0 | config.py:72 | GPU detection |
| `torchaudio` | >=2.0.0 | - | Audio processing |
| `requests` | >=2.31.0 | pipeline.py:10, app.py:432 | HTTP client for Ollama |
| `python-dotenv` | >=1.0.0 | config.py:9 | Load .env files |
| `numpy` | >=1.24.0 | - | Numerical operations |

---

## Performance Optimization

### Model Caching

**Implementation:** app.py:56-81

```python
_whisper_model = None  # Global cache
_model_lock = threading.Lock()

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:  # Double-check locking
                _whisper_model = WhisperModel(...)
    return _whisper_model
```

**Benefits:**
- Model loaded once on first request
- Subsequent transcriptions reuse same model
- ~5-10 second speedup per request
- Reduces GPU memory thrashing

**Called by:**
- app.py:322 - Passed to AudioPipeline
- pipeline.py:125 - Used if not provided

---

### GPU Acceleration

**Auto-detection:** config.py:65-87

```python
if torch.cuda.is_available():
    device = 'cuda'
    compute_type = 'float16'
else:
    device = 'cpu'
    compute_type = 'int8'
```

**Speed Improvements:**
- GPU: ~5-10x faster than CPU
- base.en model on RTX 3060: ~30s for 5-min audio
- base.en model on CPU (8 cores): ~2min for 5-min audio

---

### Server-Sent Events (SSE)

**Implementation:** app.py:363-394

**Instead of:**
```javascript
// Polling (old method)
setInterval(() => fetch('/api/status'), 1000)  // Every 1 second
```

**Now:**
```javascript
// SSE (new method)
EventSource('/api/status/stream')  // Push updates
```

**Benefits:**
- Reduces server load (no constant polling)
- Real-time updates (500ms check interval)
- Automatic reconnection
- Lower bandwidth usage

---

### Maximum Accuracy Q&A with Full Transcript Analysis

**Implementation:** app.py:417-491

**Design Philosophy:**
- **Accuracy over speed** - Send full transcript to maximize accuracy
- **Rigorous citation** - Every answer must have exact quotes
- **Chain-of-thought reasoning** - Force systematic searching and verification
- **Clean UI** - Collapsible evidence sections to avoid overwhelming users

**Solution:**
```python
# 1. Read FULL transcript (no truncation for accuracy)
full_transcript = f.read()

# 2. Use maximum context possible (100k chars = ~25k tokens)
# phi3-mini-128k supports 131k tokens total
max_context = 100000
transcript = full_transcript[:max_context]

# 3. Chain-of-thought prompt forces:
#    - Step 1: Search ENTIRE transcript
#    - Step 2: Extract EXACT quotes
#    - Step 3: Verify accuracy
#    - Step 4: Formulate answer from quotes only

# 4. Structured output separates evidence from answer
OUTPUT FORMAT:
📎 EVIDENCE FROM TRANSCRIPT:
• "exact quote 1..." — Speaker Name
• "exact quote 2..." — Speaker Name

💬 ANSWER:
[Answer synthesizing the quotes]
```

**UI Enhancement:**
- Evidence section is **collapsible** (click to expand/collapse)
- Quotes displayed in scrollable section (max 200px height)
- Clean separation between evidence and answer
- Prevents quote overload while maintaining full accuracy

**Benefits:**
- ✅ Uses **full transcript** (up to 100k chars)
- ✅ Model searches entire context systematically
- ✅ Forces **exact quote extraction** with speaker attribution
- ✅ **Chain-of-thought** prevents hallucination
- ✅ Structured output enables clean UI presentation
- ✅ Collapsible quotes keep interface uncluttered

**Trade-offs:**
- ⏱️ **Slower responses** (5-15 seconds vs 2-5 seconds)
  - Reason: 10x more content to process (100k vs 10k chars)
  - Acceptable for accuracy-critical applications
- 🖥️ **Higher GPU/CPU usage** during inference
  - RTX 3050 (4GB): May be slower but functional
  - Better GPU: Faster responses while maintaining accuracy

**Example Flow:**
1. User asks: "What was said about Instagram?"
2. System sends **entire transcript** (e.g., 45,000 characters)
3. AI performs systematic search through ALL content
4. AI extracts: `"We should focus on Instagram marketing in Q2" — Speaker 1`
5. UI displays collapsible evidence + synthesized answer

**Context Limits Evolution:**
- v1.0: 10,000 characters (~2,500 tokens) - Fast but incomplete
- v2.0 (keyword approach): 50,000 chars (~12,500 tokens) - Fast but unreliable keyword matching
- v2.1 (current): 100,000 chars (~25,000 tokens) - Maximum accuracy, slower
- Model capacity: 131,072 tokens (~400,000+ characters available)

---

## Security Features

### Input Sanitization

**File uploads:** app.py:83-98

```python
def sanitize_filename(filename):
    filename = os.path.basename(filename)  # Remove path
    filename = secure_filename(filename)   # Werkzeug sanitization
    filename = re.sub(r'[^\w\s\-\.]', '', filename)  # Remove special chars
    return filename
```

**Project names:** app.py:210, 408, 468

```python
project_name = secure_filename(project_name)  # All project access sanitized
```

**Questions:** app.py:409

```python
question = question.strip()[:500]  # Limit to 500 chars
```

---

### File Validation

**Extension check:** app.py:94-98

```python
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4', 'webm', 'ogg', 'flac', 'aac'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

**Size limit:** app.py:26, upload.html:204-208

```python
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
```

---

### Path Traversal Prevention

**Example attack prevented:**
```
# Attack attempt:
POST /upload
    filename: "../../etc/passwd"

# Sanitized to:
    filename: "etcpasswd"
```

**Validation:** app.py:210-214

```python
project_name = secure_filename(project_name)
project_path = Path(PROJECTS_FOLDER) / project_name

if not project_path.exists() or not project_path.is_dir():
    return "Project not found", 404
```

---

## Error Handling

### Pipeline Errors

**Implementation:** pipeline.py:431-439

```python
try:
    create_project_folder()
    transcribe()
    add_speakers()
    generate_summary()
    save_timing_report()
except Exception as e:
    logger.error(f"Pipeline failed: {e}")
    # Clean up partial project
    if self.project_dir.exists():
        shutil.rmtree(self.project_dir)
    raise
```

**Cleanup on failure:**
- Deletes incomplete project folder
- Prevents orphaned files
- Logs full stack trace

---

### API Error Responses

**Ollama Connection Errors:** app.py:455-461, pipeline.py:369-377

```python
try:
    response = requests.post(OLLAMA_URL, ...)
except requests.exceptions.Timeout:
    return {"error": "Request timed out"}, 504
except requests.exceptions.ConnectionError:
    return {"error": "Could not connect to Ollama"}, 503
except Exception as e:
    return {"error": str(e)}, 500
```

---

### Logging

**Configuration:** app.py:36-45

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()  # Also prints to console
    ]
)
```

**Log locations:**
- Console output (real-time)
- app.log file (persistent)

---

## Testing

### Manual Testing Checklist

#### Upload Flow
1. ✓ Upload valid audio file (MP3, WAV, etc.)
2. ✓ Try invalid file type → should reject
3. ✓ Try oversized file (>500MB) → should reject
4. ✓ Watch progress updates → should see all stages
5. ✓ Check project created in projects/
6. ✓ Verify all files: transcript, summary, timing

#### Project Viewing
1. ✓ View project from index page
2. ✓ Check transcript displays correctly
3. ✓ Check summary displays correctly
4. ✓ Download transcript → verify content
5. ✓ Download summary → verify content

#### Q&A Chat
1. ✓ Ask question about transcript
2. ✓ Verify answer is relevant
3. ✓ Try with Ollama stopped → should show error
4. ✓ Ask multiple questions → should all work

#### Error Handling
1. ✓ Upload with Ollama stopped → transcript should work, summary should fail gracefully
2. ✓ Upload with FFmpeg missing → should show clear error
3. ✓ Try path traversal in filename → should sanitize
4. ✓ Stop process mid-transcription → should clean up

---

### Setup Validation

**Script:** setup.py
**Run:** `python setup.py`

**Checks:**
1. Python version >= 3.8
2. FFmpeg installed
3. Ollama running
4. Phi-3 model available
5. All Python packages installed
6. GPU detection
7. Directory creation
8. .env file exists

---

## Common Issues & Solutions

### 1. "Could not connect to Ollama"

**Cause:** Ollama service not running
**Solution:**
```bash
ollama serve
ollama pull phi3:latest
```

**Code Reference:** app.py:458, pipeline.py:373

---

### 2. "FFmpeg not found"

**Cause:** FFmpeg not in PATH
**Solution:** Install FFmpeg (see setup.py:35-50)

**Code Reference:** pipeline.py:86

---

### 3. "GPU not detected"

**Cause:** PyTorch not CUDA-enabled
**Solution:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Code Reference:** config.py:72-76

---

### 4. Out of memory errors

**Cause:** Model too large for GPU
**Solution:** Use smaller model or CPU mode

```bash
# .env
WHISPER_MODEL=tiny.en
# or
WHISPER_DEVICE=cpu
```

**Code Reference:** config.py:33-35

---

## Development Notes

### Adding New Features

#### To add a new pipeline stage:

1. **Add method to AudioPipeline** (pipeline.py)
```python
def new_stage(self):
    self.update_status('New Stage', 85, 'Processing...', 'Detail')
    # Your code here
    self.log_time("new_stage")
```

2. **Call from run()** (pipeline.py:421)
```python
def run(self):
    # ... existing stages
    self.new_stage()
    # ... rest
```

3. **Update progress percentages** to ensure 0-100 coverage

---

#### To add a new API endpoint:

1. **Add route to app.py**
```python
@app.route('/api/new_endpoint', methods=['POST'])
def new_endpoint():
    data = request.get_json()
    # Process
    return jsonify({'result': 'success'})
```

2. **Add frontend call** (in templates/*.html)
```javascript
fetch('/api/new_endpoint', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({...})
})
```

3. **Update ARCHITECTURE.md** (this file)

---

### File Modification Guidelines

**When modifying:**
- `config.py` → Update this file's Configuration section
- `app.py` routes → Update API Endpoints section
- `pipeline.py` stages → Update Data Flow section
- Templates → Update template documentation
- Add imports → Update File Dependencies section

---

## Future Improvements

### Planned Features
- [ ] Database backend (SQLite/PostgreSQL)
- [ ] User authentication
- [ ] Better speaker diarization (pyannote.audio)
- [ ] Export to SRT/VTT subtitles
- [ ] Audio playback with timestamp sync
- [ ] Batch processing queue
- [ ] Docker containerization
- [ ] WebSocket for real-time updates
- [ ] Multi-language support

### Code Refactoring
- [ ] Move job state to Redis
- [ ] Separate API from web interface
- [ ] Add comprehensive test suite
- [ ] Add type hints throughout
- [ ] Migrate to async/await for I/O

---

## Maintenance

### Log Rotation

**Current:** Single app.log file (no rotation)

**To implement:**
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

---

### Cleanup Old Projects

**Manual:**
```bash
# Delete projects older than 30 days
find projects/ -type d -mtime +30 -exec rm -rf {} +
```

**Automated:** Add to app.py
```python
@app.route('/admin/cleanup')
def cleanup_old_projects():
    # Implementation here
    pass
```

---

## Quick Reference

### Start Application
```bash
# Windows
start.bat

# Linux/Mac
python app.py
```

### Access Points
- Web UI: http://localhost:5000
- Health Check: http://localhost:5000/health
- API Base: http://localhost:5000/api/

### Key Files to Edit
- Configuration: `.env`
- Routes: `app.py`
- Processing: `pipeline.py`
- UI: `templates/*.html`

### Log Files
- Application: `app.log`
- Pipeline: Console output
- Ollama: System logs

---

## Version History

**v2.0** - Current
- Model caching
- Server-Sent Events
- Phi-3 integration
- GPU optimization
- Enhanced security

**v1.0** - Initial
- Basic transcription
- Simple speaker detection
- Mistral summarization

---

**END OF ARCHITECTURE REFERENCE**

*Last updated: 2025-09-30*
*For questions or updates, modify this file and commit to version control.*
