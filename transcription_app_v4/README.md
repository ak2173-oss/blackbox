# 🎙️ Transcription Hub v2

**AI-powered audio transcription with GPU acceleration, speaker detection, and intelligent summarization using Phi-3**

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

### Core Capabilities
- 🚀 **GPU-Accelerated Transcription** - Automatic CUDA detection with faster-whisper
- 🎯 **Automatic Speaker Detection** - Identifies multiple speakers based on speech patterns
- 🤖 **AI Summarization** - Powered by Phi-3 via Ollama
- 💬 **Interactive Q&A** - Ask questions about your transcripts
- 📊 **Beautiful Web UI** - Modern, responsive interface with real-time progress
- ⚡ **Server-Sent Events** - Live processing updates without polling spam
- 🔒 **Security Hardened** - Input sanitization, file validation, and safe error handling

### New in v2.0
- ✅ Model caching for faster subsequent transcriptions
- ✅ Centralized configuration with environment variables
- ✅ Improved error handling and recovery
- ✅ Server-Sent Events for real-time progress (replaces polling)
- ✅ Enhanced UI with gradient designs and better UX
- ✅ Phi-3 integration for faster, more efficient AI responses
- ✅ GPU memory optimization
- ✅ Project search and filtering
- ✅ Detailed timing reports per stage

### New in v4.0
- ✅ **USB Audio Recorder Detection** - Auto-detect when audio recorder (D:/record) is plugged in
- ✅ **Batch Import** - Import multiple files from USB device with progress tracking and time estimates
- ✅ **Clickable Timestamps** - Click any transcript segment to jump audio playback to that time
- ✅ **Whisper Model Selection** - Choose model size (tiny/base/small/medium) during batch import
- ✅ **Automatic Whisper Unload** - Frees GPU VRAM after transcription for Ollama
- ✅ **Chat History Persistence** - Q&A conversations saved per project
- ✅ **HH:MM:SS Time Format** - All timestamps and durations display as 00:00:00
- ✅ **Improved Date Format** - Project dates shown as 28-Jan-2026 style
- ✅ **PLAUD-style Dark UI** - Clean black/gray accent theme
- ✅ **Siri-style Border Glow** - iOS 18-inspired animated conic-gradient border glow on device detection

### Siri Border Glow - Technical Details
The border glow animation uses CSS Houdini (`@property --siri-angle`) to animate a `conic-gradient` applied via `border-image`. Colors cycle through pink (`#f652bb`), blue (`#0855ff`), purple (`#5f2bf6`), and orange (`#ec882d`). A blurred `::before` pseudo-element provides a soft outer glow. The animation:
- **Triggers** when USB recorder is detected (auto or manual scan)
- **Persists** while the import modal is open and during file processing
- **Fades out** (0.6s transition) when the modal closes, device disconnects, or import completes
- Uses `pointer-events: none` and `z-index: 9999` so it never blocks interaction
- Implemented in `templates/index.html` via `showSiriAnimation()` / `hideSiriAnimation()` JS functions

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- FFmpeg (for audio processing)
- Ollama (for AI features)
- Optional: CUDA-capable GPU for faster processing

### Python Dependencies
```
faster-whisper>=0.10.0
ffmpeg-python>=0.2.0
Flask>=3.0.0
Flask-CORS>=4.0.0
torch>=2.0.0
requests>=2.31.0
python-dotenv>=1.0.0
```

## 🚀 Quick Start

### 1. Install Dependencies

#### Install FFmpeg
**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg  # Debian/Ubuntu
sudo yum install ffmpeg  # CentOS/RHEL
```

#### Install Ollama and Phi-3
```bash
# Install Ollama
# Visit https://ollama.com/download

# Pull Phi-3 model
ollama pull phi3:latest
```

#### Install Python Dependencies
```bash
cd transcription_app_v2
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` to customize settings:
```env
# Example configuration
WHISPER_MODEL=base.en
OLLAMA_MODEL=phi3:latest
MAX_UPLOAD_SIZE=524288000
```

### 3. Run the Application

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## 📖 Usage Guide

### Uploading Audio Files

1. Click **"Upload New Audio"** on the home page
2. Drag & drop your audio file or click to browse
3. Watch real-time progress as your file is processed
4. View the completed project with transcript and summary

**Supported formats:** MP3, WAV, M4A, MP4, WebM, OGG, FLAC, AAC
**Max file size:** 500MB (configurable)

### Viewing Projects

- Browse all projects on the home page
- Click **"View"** to see full transcript and AI summary
- Use the search bar to filter projects
- Download transcripts and summaries as text files

### Interactive Q&A

On any project page, use the chat interface to:
- Ask questions about the conversation
- Request specific information
- Get clarifications on key points
- Summarize portions of the transcript

Chat history is automatically saved and persists across sessions.

### USB Audio Recorder Import

1. Plug in your USB audio recorder (detected at D:/record)
2. A popup will appear showing available audio files
3. Select files to import (or "Select All")
4. Choose your preferred Whisper model size
5. Click "Import Selected" to begin batch processing
6. Watch real-time progress with time estimates

**Manual scan:** Click "Scan for Device" button if device isn't auto-detected.

### Clickable Timestamps

On any project page:
- Click any transcript segment to jump audio playback to that timestamp
- All times displayed in HH:MM:SS format (e.g., 00:05:23)
- Hover over segments to see "Click to play from" tooltip

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base.en` | Whisper model size (tiny, base, small, medium, large) |
| `WHISPER_DEVICE` | `auto` | Device for transcription (auto, cuda, cpu) |
| `OLLAMA_MODEL` | `phi3:latest` | Ollama model for AI features |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `MAX_UPLOAD_SIZE` | `524288000` | Max upload size in bytes (500MB) |
| `SPEAKER_GAP_THRESHOLD` | `2.0` | Seconds of silence to detect speaker change |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### Whisper Models

Choose based on your needs:

| Model | Size | Speed | Accuracy | GPU Memory |
|-------|------|-------|----------|------------|
| `tiny.en` | 39M | Fastest | Good | ~1GB |
| `base.en` | 74M | Fast | Better | ~1GB |
| `small.en` | 244M | Medium | Great | ~2GB |
| `medium.en` | 769M | Slow | Excellent | ~5GB |
| `large-v2` | 1550M | Slowest | Best | ~10GB |

## 🏗️ Architecture

```
transcription_app_v2/
├── app.py                 # Flask web application
├── pipeline.py            # Audio processing pipeline
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── README.md             # This file
├── templates/            # HTML templates
│   ├── index.html       # Project listing
│   ├── upload.html      # Upload interface
│   └── project.html     # Project viewer with chat
├── static/              # Static assets (CSS, JS)
├── uploads/             # Temporary upload storage
└── projects/            # Processed projects
    └── [project_name]/
        ├── audio/              # Original audio file
        ├── transcripts/        # JSON and text transcripts
        ├── summary/            # AI-generated summaries
        └── processing_time.json  # Timing report
```

## 🔧 Advanced Usage

### Running Pipeline Directly

Process an audio file without the web interface:

```bash
python pipeline.py path/to/audio.mp3
```

### API Endpoints

The application exposes several API endpoints:

- `GET /` - Home page with project listing
- `GET /project/<name>` - View specific project
- `POST /upload` - Upload audio file
- `GET /api/status/<job_id>` - Get job status (JSON)
- `GET /api/status/stream/<job_id>` - Stream job status (SSE)
- `POST /api/ask_question` - Ask questions about transcript
- `GET /download/<project>/<type>` - Download project files
- `GET /health` - System health check
- `GET /api/check_device` - Check if USB audio recorder is connected
- `POST /api/batch_import` - Start batch import from USB device
- `GET /api/batch_import_status` - Get batch import progress

### GPU Optimization

The application automatically detects and uses GPU if available:

```python
# Check GPU status
python -c "import torch; print(torch.cuda.is_available())"

# View GPU info
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

To force CPU mode, set in `.env`:
```env
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

## 🐛 Troubleshooting

### Common Issues

**1. "Could not connect to Ollama"**
```bash
# Make sure Ollama is running
ollama serve

# In a new terminal, pull the model
ollama pull phi3:latest
```

**2. "FFmpeg not found"**
```bash
# Verify FFmpeg installation
ffmpeg -version

# If not installed, see "Install Dependencies" section
```

**3. GPU not detected**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**4. Out of memory errors**
```env
# Use a smaller Whisper model
WHISPER_MODEL=tiny.en

# Or force CPU mode
WHISPER_DEVICE=cpu
```

### Logging

Check logs for detailed error information:
```bash
tail -f app.log
```

Set debug level in `.env`:
```env
LOG_LEVEL=DEBUG
```

## 🚦 Performance Tips

1. **Use GPU** - 5-10x faster transcription
2. **Choose appropriate model** - Balance speed vs accuracy
3. **Keep Ollama running** - Faster AI responses
4. **Batch processing** - Upload multiple files at once (future feature)
5. **SSD storage** - Faster file I/O for large audio files

## 📊 Benchmarks

Typical processing times (base.en model):

| Audio Length | CPU (8 cores) | GPU (RTX 3060) |
|--------------|---------------|----------------|
| 5 minutes | ~2 minutes | ~30 seconds |
| 15 minutes | ~6 minutes | ~90 seconds |
| 30 minutes | ~12 minutes | ~3 minutes |
| 60 minutes | ~25 minutes | ~6 minutes |

*Times include transcription, speaker detection, and AI summary generation.*

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add user authentication
- [ ] Implement database backend (SQLite/PostgreSQL)
- [ ] Support for more languages
- [ ] Better speaker diarization (pyannote.audio)
- [ ] Export to SRT/VTT for subtitles
- [x] ~~Audio playback with timestamp sync~~ (Implemented in v4.0)
- [x] ~~Batch processing queue~~ (Implemented in v4.0 via USB import)
- [ ] Docker containerization

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) - Efficient Whisper implementation
- [Ollama](https://ollama.com/) - Local LLM runner
- [Phi-3](https://azure.microsoft.com/en-us/products/phi-3) - Microsoft's compact LLM
- [Flask](https://flask.palletsprojects.com/) - Web framework

## 📧 Support

For issues and questions:
- Create an issue on GitHub
- Check existing issues for solutions
- Review logs in `app.log`

---

**Made with ❤️ using Python, Flask, and AI**