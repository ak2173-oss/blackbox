#!/bin/bash
# Start Transcription App v2 with GPU-accelerated Ollama

# Kill any existing Ollama processes
pkill -f "ollama serve" 2>/dev/null

# Start Ollama server in background
export LD_LIBRARY_PATH=~/bin/lib/ollama:$LD_LIBRARY_PATH
nohup ~/bin/ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Started Ollama (PID: $OLLAMA_PID)"

# Wait for Ollama to be ready
sleep 3
echo "Waiting for Ollama to start..."
for i in {1..10}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama is ready"
        break
    fi
    sleep 1
done

# Start Flask app
cd /mnt/c/Users/Agneya/Dev/blackbox/transcription_app_v2
export LD_LIBRARY_PATH=$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
source venv/bin/activate
python app.py
