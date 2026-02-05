# Transcription App

Local audio transcription with AI-powered summaries.

## Quick Start

### Windows
Double-click `START-STOP/START_APP.bat` to launch the app.

Double-click `START-STOP/STOP_APP.bat` to shut down.

### Access
Open http://localhost:5000 in your browser.

## Folder Structure

```
blackbox/
├── START-STOP/          # Launch scripts (use these!)
│   ├── START_APP.bat    # Double-click to start
│   └── STOP_APP.bat     # Double-click to stop
├── transcription_app/   # Main application (latest version)
├── UI development/      # Design references
├── test files/          # Test audio files
├── test-audio/          # More test files
└── _archive/            # Old versions (v2, legacy scripts)
```

## Tech Stack
- **Transcription**: Whisper large-v3 (GPU-accelerated)
- **Summarization**: Qwen2.5:7b-instruct via Ollama
- **Context Window**: 32K tokens
- **Frontend**: Flask + Tailwind CSS
