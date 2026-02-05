#!/bin/bash
# Stop Transcription App

echo "================================================================"
echo " TRANSCRIPTION APP - SHUTDOWN"
echo "================================================================"
echo ""

echo "🛑 Stopping Ollama server..."
pkill -f "ollama serve" 2>/dev/null && echo "  Sent stop signal..." || echo "  (Ollama was not running)"

echo "🛑 Stopping Flask application..."
pkill -f "python.*app.py" 2>/dev/null && echo "  Sent stop signal..." || echo "  (Flask was not running)"

sleep 2

# Verify processes stopped
echo ""
echo "🔍 Verifying shutdown..."

ollama_running=$(pgrep -f "ollama serve" | wc -l)
flask_running=$(pgrep -f "python.*app.py" | wc -l)

if [ "$ollama_running" -gt 0 ] || [ "$flask_running" -gt 0 ]; then
    echo "⚠️  Some processes did not stop gracefully. Force stopping..."
    pkill -9 -f "ollama serve" 2>/dev/null
    pkill -9 -f "python.*app.py" 2>/dev/null
    sleep 2

    # Verify again after force kill
    ollama_running=$(pgrep -f "ollama serve" | wc -l)
    flask_running=$(pgrep -f "python.*app.py" | wc -l)
fi

# Final verification report
echo ""
if [ "$ollama_running" -eq 0 ] && [ "$flask_running" -eq 0 ]; then
    echo "✅ VERIFIED: All processes stopped successfully"
    echo "   - Ollama: ✓ Stopped"
    echo "   - Flask:  ✓ Stopped"
    exit_code=0
else
    echo "❌ ERROR: Some processes are still running!"
    if [ "$ollama_running" -gt 0 ]; then
        echo "   - Ollama: ✗ Still running ($ollama_running process(es))"
    fi
    if [ "$flask_running" -gt 0 ]; then
        echo "   - Flask:  ✗ Still running ($flask_running process(es))"
    fi
    echo ""
    echo "Try running: sudo pkill -9 ollama && sudo pkill -9 python"
    exit_code=1
fi

echo ""
echo "================================================================"
exit $exit_code
