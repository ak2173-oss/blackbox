@echo off
REM Transcription Hub v2 - Windows Launcher
REM Double-click this file to start the app

echo ================================================================
echo  TRANSCRIPTION HUB V2 - LAUNCHER
echo ================================================================
echo.
echo Starting Ollama and Flask app in WSL...
echo.

REM Launch WSL with the start script
wsl -d Ubuntu bash -c "cd /mnt/c/Users/Agneya/Dev/blackbox/transcription_app_v2 && ./start_app.sh"

REM If WSL exits, pause to see error messages
pause
