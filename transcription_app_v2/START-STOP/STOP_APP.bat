@echo off
REM Transcription Hub v2 - Stop Script
REM Double-click this file to stop all running processes

echo ================================================================
echo  TRANSCRIPTION HUB V2 - SHUTDOWN
echo ================================================================
echo.
echo Stopping all processes...
echo.

REM Stop processes in WSL
wsl -d Ubuntu bash -c "pkill -f 'ollama serve' 2>/dev/null; pkill -f 'python.*app.py' 2>/dev/null; echo '✓ All processes stopped'"

echo.
echo ================================================================
echo  All services have been stopped
echo ================================================================
echo.
pause
