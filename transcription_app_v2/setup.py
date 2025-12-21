#!/usr/bin/env python3
"""
Setup script for Transcription Hub v2
Validates dependencies and configuration
"""

import sys
import subprocess
import shutil
from pathlib import Path

# Use simple ASCII characters instead of emojis for Windows compatibility
CHECK = "[OK]"
CROSS = "[X]"
WARN = "[!]"

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version < (3, 8):
        print(f"{CROSS} Python 3.8 or higher is required!")
        return False

    print(f"{CHECK} Python version OK")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print_header("Checking FFmpeg")

    if shutil.which("ffmpeg"):
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        version = result.stdout.split('\n')[0]
        print(f"{CHECK} FFmpeg found: {version}")
        return True
    else:
        print(f"{CROSS} FFmpeg not found!")
        print("\nPlease install FFmpeg:")
        print("  Windows: choco install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt install ffmpeg")
        return False

def check_ollama():
    """Check if Ollama is available"""
    print_header("Checking Ollama")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"{CHECK} Ollama is running with {len(models)} models")

            # Check for phi3
            phi3_found = any('phi3' in m['name'] for m in models)
            if phi3_found:
                print(f"{CHECK} Phi-3 model found")
            else:
                print(f"{WARN} Phi-3 not found. Run: ollama pull phi3:latest")

            return True
        else:
            print(f"{WARN} Ollama is not responding properly")
            return False
    except Exception as e:
        print(f"{CROSS} Cannot connect to Ollama")
        print("\nPlease install and start Ollama:")
        print("  1. Visit: https://ollama.com/download")
        print("  2. Install Ollama")
        print("  3. Run: ollama serve")
        print("  4. Pull model: ollama pull phi3:latest")
        return False

def check_dependencies():
    """Check Python dependencies"""
    print_header("Checking Python Dependencies")

    required = [
        "flask",
        "flask_cors",
        "faster_whisper",
        "torch",
        "requests",
        "dotenv"
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"{CHECK} {package}")
        except ImportError:
            print(f"{CROSS} {package}")
            missing.append(package)

    if missing:
        print(f"\n{CROSS} Missing packages: {', '.join(missing)}")
        print("\nRun: pip install -r requirements.txt")
        return False

    return True

def check_gpu():
    """Check for GPU availability"""
    print_header("Checking GPU")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"{CHECK} GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print(f"{WARN} No GPU detected, will use CPU")
            print("   (This is fine, but processing will be slower)")
            return True
    except ImportError:
        print(f"{WARN} PyTorch not installed, cannot check GPU")
        return True

def setup_directories():
    """Create necessary directories"""
    print_header("Setting Up Directories")

    dirs = ['uploads', 'projects', 'static']
    for dir_name in dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"{CHECK} {dir_name}/")

    return True

def setup_env_file():
    """Check/create .env file"""
    print_header("Checking Environment Configuration")

    env_file = Path('.env')
    env_example = Path('.env.example')

    if env_file.exists():
        print(f"{CHECK} .env file exists")
    else:
        if env_example.exists():
            print(f"{WARN} .env not found, copying from .env.example")
            shutil.copy(env_example, env_file)
            print(f"{CHECK} Created .env file")
            print("\nTIP: Please edit .env to customize your settings")
        else:
            print(f"{CROSS} .env.example not found!")
            return False

    return True

def main():
    print("\n" + "=" * 70)
    print("   TRANSCRIPTION HUB V2 - SETUP")
    print("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("FFmpeg", check_ffmpeg),
        ("Ollama", check_ollama),
        ("Python Dependencies", check_dependencies),
        ("GPU", check_gpu),
        ("Directories", setup_directories),
        ("Environment", setup_env_file),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"{CROSS} Error during {name} check: {e}")
            results.append((name, False))

    # Summary
    print_header("Setup Summary")

    all_passed = True
    for name, result in results:
        status = CHECK if result else CROSS
        print(f"{status} {name}")
        if not result:
            all_passed = False

    if all_passed:
        print("\n" + "=" * 70)
        print("SUCCESS: Setup complete! You're ready to go!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Edit .env to customize settings (optional)")
        print("  2. Run: python app.py")
        print("  3. Open: http://localhost:5000")
        print("\nFor detailed instructions, see README.md")
    else:
        print("\n" + "=" * 70)
        print("WARNING: Setup incomplete - please fix the issues above")
        print("=" * 70)
        print("\nFor help, check README.md")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())