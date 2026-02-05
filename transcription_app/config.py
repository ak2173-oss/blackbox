# config.py
# Centralized configuration for Transcription App v2

import os
from pathlib import Path

# Try to load environment variables with dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, will use system environment variables only
    pass

# Base paths
BASE_DIR = Path(__file__).parent
TEMPLATES_FOLDER = BASE_DIR / 'templates'
STATIC_FOLDER = BASE_DIR / 'static'

# Allow custom paths from environment variables (for G: drive support)
UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', str(BASE_DIR / 'uploads')))
PROJECTS_FOLDER = Path(os.getenv('PROJECTS_FOLDER', str(BASE_DIR / 'projects')))

# Flask configuration
class Config:
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_UPLOAD_SIZE', 500 * 1024 * 1024))  # 500MB default

    # Paths
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    PROJECTS_FOLDER = str(PROJECTS_FOLDER)

    # Transcription Engine Selection
    TRANSCRIPTION_ENGINE = os.getenv('TRANSCRIPTION_ENGINE', 'whisper')  # 'whisper' or 'parakeet'

    # Whisper configuration
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base.en')
    WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'auto')  # 'auto', 'cuda', or 'cpu'
    WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'auto')  # 'auto', 'float16', 'int8'
    WHISPER_CPU_THREADS = int(os.getenv('WHISPER_CPU_THREADS', os.cpu_count() or 4))
    WHISPER_NUM_WORKERS = int(os.getenv('WHISPER_NUM_WORKERS', 4))

    # Parakeet configuration (Alternative fast ASR models)
    PARAKEET_MODEL = os.getenv('PARAKEET_MODEL', 'facebook/wav2vec2-large-960h-lv60-self')  # HuggingFace model ID
    PARAKEET_CHUNK_LENGTH = int(os.getenv('PARAKEET_CHUNK_LENGTH', 30))  # seconds per chunk

    # Ollama configuration
    OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')
    OLLAMA_FALLBACK_MODEL = os.getenv('OLLAMA_FALLBACK_MODEL', 'llama3.1:8b')
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 180))  # 3 minutes
    OLLAMA_TEMPERATURE = float(os.getenv('OLLAMA_TEMPERATURE', 0.7))
    OLLAMA_CONTEXT_LENGTH = int(os.getenv('OLLAMA_CONTEXT_LENGTH', 32768))  # 32K default

    # Speaker detection
    SPEAKER_GAP_THRESHOLD = float(os.getenv('SPEAKER_GAP_THRESHOLD', 2.0))  # seconds
    SPEAKER_MERGE_THRESHOLD = float(os.getenv('SPEAKER_MERGE_THRESHOLD', 1.0))  # seconds

    # Processing settings
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1

    # File validation
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'mp4', 'webm', 'ogg', 'flac', 'aac'}

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')

    # Security
    CORS_ENABLED = os.getenv('CORS_ENABLED', 'False').lower() == 'true'
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')


def get_device_config():
    """Auto-detect best device and compute type for Whisper"""
    device = Config.WHISPER_DEVICE
    compute_type = Config.WHISPER_COMPUTE_TYPE

    if device == 'auto' or compute_type == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                compute_type = 'float16'
                print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                compute_type = 'int8'
                print("[OK] Using CPU (no CUDA available)")
        except ImportError:
            device = 'cpu'
            compute_type = 'int8'
            print("[OK] Using CPU (torch not available)")

    return device, compute_type


def validate_config():
    """Validate configuration and create necessary directories"""
    # Create directories
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    PROJECTS_FOLDER.mkdir(exist_ok=True)
    TEMPLATES_FOLDER.mkdir(exist_ok=True)
    STATIC_FOLDER.mkdir(exist_ok=True)

    # Validate Ollama connection
    try:
        import requests
        response = requests.get(f"{Config.OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            if Config.OLLAMA_MODEL in model_names or any(Config.OLLAMA_MODEL.split(':')[0] in name for name in model_names):
                print(f"[OK] Ollama connected: {Config.OLLAMA_MODEL}")
            else:
                print(f"[!] Warning: Model '{Config.OLLAMA_MODEL}' not found in Ollama")
                print(f"  Available models: {', '.join(model_names)}")
        else:
            print(f"[!] Warning: Could not connect to Ollama at {Config.OLLAMA_URL}")
    except Exception as e:
        print(f"[!] Warning: Ollama validation failed: {e}")

    print("[OK] Configuration validated")


if __name__ == '__main__':
    validate_config()
    device, compute = get_device_config()
    print(f"\nDevice: {device}, Compute: {compute}")