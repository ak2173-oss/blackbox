#!/bin/bash
# Start Transcription App v2 with GPU-accelerated Ollama

set -e  # Exit on error

echo "================================================================"
echo " TRANSCRIPTION HUB V2 - STARTUP"
echo "================================================================"
echo ""

# Navigate to app directory
cd /mnt/c/Users/Agneya/Dev/blackbox/transcription_app_v2

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "ollama serve" 2>/dev/null || true
pkill -f "python.*app.py" 2>/dev/null || true
sleep 1

# Check if Ollama binary exists
if [ ! -f ~/bin/ollama ]; then
    echo "❌ ERROR: Ollama not found at ~/bin/ollama"
    echo "Please install Ollama first"
    exit 1
fi

# Start Ollama server in background
echo "🚀 Starting Ollama server..."
export LD_LIBRARY_PATH=~/bin/lib/ollama:$LD_LIBRARY_PATH
nohup ~/bin/ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "✓ Ollama started (PID: $OLLAMA_PID)"

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
for i in {1..15}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama is ready!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "❌ ERROR: Ollama failed to start. Check /tmp/ollama.log"
        exit 1
    fi
    sleep 1
done

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ ERROR: Virtual environment not found"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Start Flask app
echo ""
echo "🚀 Starting Flask application..."
echo ""
export LD_LIBRARY_PATH=$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
source venv/bin/activate

# Trap Ctrl+C to cleanup
trap 'echo ""; echo "🛑 Shutting down..."; pkill -f "ollama serve"; exit 0' INT TERM

# Start Flask (this will block until Ctrl+C)
python app.py
