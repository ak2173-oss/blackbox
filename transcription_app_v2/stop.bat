@echo off
REM Stop script for Transcription App v2
REM Closes all running processes and frees GPU memory

echo ========================================
echo   Stopping Transcription Hub v2
echo ========================================
echo.

echo [1/3] Stopping Flask app...
taskkill /F /IM python.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Flask app stopped
) else (
    echo [!] No Flask app running
)

echo.
echo [2/3] Stopping Ollama...
taskkill /F /IM ollama.exe >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Ollama stopped
) else (
    echo [!] Ollama was not running
)

echo.
echo [3/3] Checking GPU memory...
nvidia-smi --query-gpu=memory.used --format=csv,noheader >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] GPU memory freed
) else (
    echo [!] NVIDIA GPU not detected or nvidia-smi not available
)

echo.
echo ========================================
echo   All processes stopped!
echo ========================================
echo.
echo To restart:
echo   1. Run: start.bat
echo   2. Or manually: python app.py
echo.
pause
