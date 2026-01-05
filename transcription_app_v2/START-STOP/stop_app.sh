#!/bin/bash
# Stop Transcription App v2

echo "================================================================"
echo " TRANSCRIPTION HUB V2 - SHUTDOWN"
echo "================================================================"
echo ""

echo "🛑 Stopping Ollama server..."
pkill -f "ollama serve" 2>/dev/null && echo "✓ Ollama stopped" || echo "  (Ollama was not running)"

echo "🛑 Stopping Flask application..."
pkill -f "python.*app.py" 2>/dev/null && echo "✓ Flask stopped" || echo "  (Flask was not running)"

sleep 1

# Check if any processes are still running
if pgrep -f "ollama serve" > /dev/null || pgrep -f "python.*app.py" > /dev/null; then
    echo ""
    echo "⚠️  Some processes are still running. Force stopping..."
    pkill -9 -f "ollama serve" 2>/dev/null
    pkill -9 -f "python.*app.py" 2>/dev/null
    sleep 1
    echo "✓ Force stopped all processes"
fi

echo ""
echo "================================================================"
echo " All services have been stopped"
echo "================================================================"
echo ""
