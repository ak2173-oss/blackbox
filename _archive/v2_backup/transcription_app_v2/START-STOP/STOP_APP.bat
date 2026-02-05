@echo off
REM Transcription Hub v2 - Stop Script
REM Double-click this file to stop all running processes

echo ================================================================
echo  TRANSCRIPTION HUB V2 - SHUTDOWN
echo ================================================================
echo.
echo Stopping all processes...
echo.

REM Run the stop script in WSL with full verification
wsl -d Ubuntu bash -c "cd /mnt/c/Users/Agneya/Dev/blackbox/transcription_app_v2/START-STOP && ./stop_app.sh"

echo.
pause
