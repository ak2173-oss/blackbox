@echo off
REM Transcription App - Stop Script
REM Double-click this file to stop all running processes

echo ================================================================
echo  TRANSCRIPTION APP - SHUTDOWN
echo ================================================================
echo.
echo Stopping all processes...
echo.

REM Run the stop script in WSL with full verification
wsl -d Ubuntu bash -c "cd /mnt/c/Users/Agneya/Dev/blackbox/transcription_app/START-STOP && ./stop_app.sh"

echo.
pause
