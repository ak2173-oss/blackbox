# 🚀 Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

✅ Python 3.8+ installed
✅ FFmpeg installed
✅ Ollama installed with Phi-3 model

---

## Step 1: Install Dependencies

```bash
cd transcription_app_v2
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit if needed (optional)
# nano .env
```

## Step 3: Verify Setup

```bash
python setup.py
```

This will check:
- ✅ Python version
- ✅ FFmpeg installation
- ✅ Ollama connection
- ✅ Phi-3 model
- ✅ Python packages
- ✅ GPU availability

## Step 4: Start the App

### Windows:
```bash
start.bat
```

### Mac/Linux:
```bash
python app.py
```

## Step 5: Open in Browser

Navigate to: **http://localhost:5000**

---

## 🎯 First Upload

1. Click **"Upload New Audio"**
2. Drag & drop an audio file (or click to browse)
3. Watch real-time progress
4. View your transcript and AI summary!

---

## 🔧 Troubleshooting

### "Ollama not found"
```bash
# Start Ollama
ollama serve

# In another terminal, pull Phi-3
ollama pull phi3:latest
```

### "FFmpeg not found"
```bash
# Windows (with Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

### "No GPU detected"
This is fine! The app works on CPU, just slower.

To enable GPU:
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 What to Expect

**Processing Times (base.en model):**
- 5 min audio → ~30 sec (GPU) / ~2 min (CPU)
- 15 min audio → ~90 sec (GPU) / ~6 min (CPU)
- 30 min audio → ~3 min (GPU) / ~12 min (CPU)

---

## 🎓 Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [IMPROVEMENTS.md](IMPROVEMENTS.md) to see what's new
- Customize settings in `.env`
- Try the Q&A feature on completed projects!

---

## 💡 Tips

1. **Keep Ollama running** for faster AI responses
2. **Use GPU** for 5-10x faster transcription
3. **Choose smaller Whisper models** (`tiny.en`) for faster processing
4. **Ask questions** using the chat interface on project pages

---

## 🆘 Need Help?

- Check `app.log` for detailed error messages
- Run `python setup.py` to validate your environment
- Review [README.md](README.md) troubleshooting section
- Ensure all prerequisites are installed

---

**That's it! You're ready to transcribe! 🎉**