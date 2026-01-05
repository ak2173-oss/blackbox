# 🚀 Quick Start Guide

## How to Launch the App

### Option 1: Windows (Easiest)
**Just double-click `START_APP.bat`** on your desktop or in this folder.

That's it! The launcher will:
1. Start Ollama server in WSL
2. Wait for it to be ready
3. Start the Flask web app
4. Open your browser automatically

### Option 2: WSL Terminal
```bash
cd /mnt/c/Users/Agneya/Dev/blackbox/transcription_app_v2
./start_app.sh
```

### Option 3: Manual (Advanced)
```bash
# Terminal 1 - Start Ollama
~/bin/ollama serve

# Terminal 2 - Start Flask
cd /mnt/c/Users/Agneya/Dev/blackbox/transcription_app_v2
source venv/bin/activate
export LD_LIBRARY_PATH=$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
python app.py
```

## Accessing the App

Once started, open your browser to:
- **Main App**: http://localhost:5000
- **Upload Page**: http://localhost:5000/upload

## Stopping the App

- **If using START_APP.bat**: Just close the terminal window
- **If using start_app.sh**: Press `Ctrl+C` in the terminal
- **Manual**: `pkill -f "ollama serve" && pkill -f "python.*app.py"`

## Troubleshooting

### App won't start
1. Make sure WSL is installed: `wsl --version`
2. Make sure Ollama is installed: `~/bin/ollama --version`
3. Check logs: `tail -f /tmp/ollama.log`

### GPU not working
- Verify CUDA: `nvidia-smi`
- Check cuDNN: `ls venv/lib/python3.12/site-packages/nvidia/cudnn/lib/`

### Port already in use
```bash
# Kill existing processes
pkill -f "ollama serve"
pkill -f "python.*app.py"
```

## What's Running

- **Ollama**: http://localhost:11434 (LLM server)
- **Flask**: http://localhost:5000 (Web app)
- **Transcription**: Whisper or Wav2Vec2 (GPU-accelerated)
- **Summarization**: Qwen2.5:7b-instruct (GPU-accelerated)

## Configuration

Edit `.env` file to change:
- Transcription engine (Whisper/Wav2Vec2)
- LLM model
- GPU settings
- Upload folder location
