@echo off
REM Transcription App - Windows Launcher
REM Double-click this file to start the app

echo ================================================================
echo  TRANSCRIPTION APP - LAUNCHER
echo ================================================================
echo.
echo Starting Ollama and Flask app in WSL...
echo.

REM Launch WSL with the start script
wsl -d Ubuntu bash -c "cd /mnt/c/Users/Agneya/Dev/blackbox/transcription_app/START-STOP && ./start_app.sh"

REM If WSL exits, pause to see error messages
pause
