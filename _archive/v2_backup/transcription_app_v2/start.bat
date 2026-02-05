@echo off
REM Quick start script for Windows

echo ========================================
echo   Transcription Hub v2 - Starting
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found.
    echo Run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    echo.
)

REM Check if .env exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Copying from .env.example...
    copy .env.example .env
    echo Please edit .env to configure your settings.
    echo.
    pause
)

echo Starting application...
echo.
python app.py

pause