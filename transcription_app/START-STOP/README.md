# 🚀 START/STOP Scripts

Quick launchers for Transcription Hub v2

## 📂 Files

| File | Purpose | How to Use |
|------|---------|------------|
| **START_APP.bat** | Start everything (Windows) | Double-click |
| **start_app.sh** | Start everything (WSL) | `./start_app.sh` |
| **STOP_APP.bat** | Stop everything (Windows) | Double-click |
| **stop_app.sh** | Stop everything (WSL) | `./stop_app.sh` |

## 🚀 Starting the App

### Windows Users (Easiest):
1. Double-click **`START_APP.bat`**
2. Wait for the app to start
3. Open browser to http://localhost:5000

### WSL Users:
```bash
./start_app.sh
```

## 🛑 Stopping the App

### Windows Users:
1. Double-click **`STOP_APP.bat`**
2. All processes will be stopped

### WSL Users:
```bash
./stop_app.sh
```

### Alternative:
- Press `Ctrl+C` in the terminal window running the app

## ℹ️ What Gets Started

- **Ollama Server** (http://localhost:11434)
  - LLM server for AI summarization
  - Uses GPU acceleration
  
- **Flask Web App** (http://localhost:5000)
  - Main transcription interface
  - Upload page, project viewer, chatbot

## ℹ️ What Gets Stopped

The STOP scripts will cleanly shutdown:
- Ollama server process
- Flask web application
- All related background processes

## 🔧 Troubleshooting

### "Processes won't stop"
Run the STOP script again - it has force-kill capability.

### "Can't start - port in use"
1. Run STOP script first
2. Then run START script

### "App starts but can't access"
Check if your firewall is blocking:
- Port 5000 (Flask)
- Port 11434 (Ollama)

## 📍 Location

These scripts should be run from the project root:
```
/mnt/c/Users/Agneya/Dev/blackbox/transcription_app_v2/START-STOP/
```

## 🔗 Quick Links

- Main App: http://localhost:5000
- Upload: http://localhost:5000/upload
- Ollama API: http://localhost:11434/api/tags
