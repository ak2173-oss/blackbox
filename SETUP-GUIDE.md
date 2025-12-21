# BlackBox Transcription - Setup Guide

## 🎯 Your Development Setup

You now have a **hybrid setup** that combines the best of both worlds:
- **Code on GitHub** - Version control, backup, portfolio
- **Data on G: Drive** - All audio files, outputs, and projects (offline, backed up)
- **Local working copy** - Fast development on C: drive

---

## 📁 Folder Structure

### Local Development (C: Drive)
```
C:\Users\Agneya\Dev\blackbox\
├── .git/                          ← Git repository
├── app.py                         ← V1 code (basic version)
├── optimized_pipeline.py
├── templates/
├── transcription_app_v2/          ← V2 code (production version)
│   ├── .env                      ← Configuration (points to G: drive)
│   ├── app.py
│   ├── pipeline.py
│   ├── config.py
│   ├── templates/
│   └── README.md
└── SETUP-GUIDE.md                 ← This file
```

### Data Storage (G: Drive - Google Drive offline)
```
G:\My Drive\Capstone\
├── BlackBox UI\
│   └── Capstone\                  ← Original backup copy
│       └── test2.wav, testWhisper.m4a (test files)
│
└── BlackBox-Data\                 ← Active data storage
    ├── uploads\                   ← Temporary uploaded files
    ├── projects\                  ← Processed transcriptions
    ├── test-audio\                ← Your test audio files
    └── backups\                   ← Optional manual backups
```

---

## 🚀 Daily Workflow

### 1. Start Working
```bash
# Open Windows Terminal
cd C:\Users\Agneya\Dev\blackbox

# Pull latest code from GitHub (if working from multiple computers)
git pull

# Navigate to v2
cd transcription_app_v2
```

### 2. Run the Application
```bash
# Make sure Ollama is running first
ollama serve

# In a new terminal
cd C:\Users\Agneya\Dev\blackbox\transcription_app_v2
python app.py
```

### 3. Use the App
- Open browser: http://localhost:5000
- Upload audio files (will be stored on G: drive)
- Process and view transcripts
- All outputs save to: `G:\My Drive\Capstone\BlackBox-Data\projects\`

### 4. Save Your Code Changes
```bash
# After making changes to code
cd C:\Users\Agneya\Dev\blackbox

# Check what changed
git status

# Add changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

---

## 📂 Where Things Are Stored

| Item | Location | Backed Up To |
|------|----------|--------------|
| **Code** | `C:\Users\Agneya\Dev\blackbox\` | GitHub (cloud) |
| **Test Audio** | `G:\My Drive\Capstone\BlackBox-Data\test-audio\` | Google Drive (cloud) |
| **Uploads** | `G:\My Drive\Capstone\BlackBox-Data\uploads\` | Google Drive (cloud) |
| **Projects** | `G:\My Drive\Capstone\BlackBox-Data\projects\` | Google Drive (cloud) |
| **Logs** | `C:\Users\Agneya\Dev\blackbox\transcription_app_v2\app.log` | Not backed up (temporary) |

---

## ⚙️ Configuration

All configuration is in `.env` file at:
```
C:\Users\Agneya\Dev\blackbox\transcription_app_v2\.env
```

**Key settings:**
```env
# Data storage on G: Drive
UPLOAD_FOLDER=G:/My Drive/Capstone/BlackBox-Data/uploads
PROJECTS_FOLDER=G:/My Drive/Capstone/BlackBox-Data/projects

# AI Model (change based on your GPU)
WHISPER_MODEL=base.en  # tiny.en, base.en, small.en, medium.en, large-v2

# Ollama model
OLLAMA_MODEL=phi3:latest  # or mistral:7b-instruct-q4_K_M
```

---

## 🔄 Redundancy Strategy

Your files are protected:

1. **Code**
   - ✅ Local: `C:\Users\Agneya\Dev\blackbox\`
   - ✅ Cloud: GitHub (https://github.com/ak2173-oss/blackbox)
   - ✅ Backup: `G:\My Drive\Capstone\BlackBox UI\Capstone\`

2. **Data (audio, transcripts, projects)**
   - ✅ Primary: `G:\My Drive\Capstone\BlackBox-Data\` (Google Drive - offline + cloud)
   - ✅ Optional: Manually copy important projects to `backups/` folder

3. **Git Version Control**
   - Every code change is tracked
   - Can restore any previous version
   - Full history on GitHub

---

## 🛠️ Common Tasks

### Add Test Audio Files
1. Navigate to: `G:\My Drive\Capstone\BlackBox-Data\test-audio\`
2. Copy your audio files there
3. Use the web UI to upload from that location

### View Old Projects
Projects are in: `G:\My Drive\Capstone\BlackBox-Data\projects\`

Each project folder contains:
- Original audio file
- Transcript (JSON and TXT)
- AI summary
- Processing time report

### Update Code from GitHub
```bash
cd C:\Users\Agneya\Dev\blackbox
git pull
```

### Clean Up Old Uploads
Temporary files in `G:\My Drive\Capstone\BlackBox-Data\uploads\` can be deleted manually.
Projects in `projects/` folder are permanent unless you delete them.

---

## 🐛 Troubleshooting

### G: Drive Not Accessible in WSL
If I (Claude) can't see your G: drive:
```bash
# Run in WSL terminal
sudo mkdir -p /mnt/g
sudo mount -t drvfs G: /mnt/g
```

### App Can't Find Uploads/Projects
Check `.env` file has correct paths:
```env
UPLOAD_FOLDER=G:/My Drive/Capstone/BlackBox-Data/uploads
PROJECTS_FOLDER=G:/My Drive/Capstone/BlackBox-Data/projects
```

### Ollama Not Connected
```bash
# Start Ollama
ollama serve

# Verify model is installed
ollama list

# Pull if needed
ollama pull phi3:latest
```

### Git Push Requires Password
Use a Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Generate new token (classic)
3. Check `repo` scope
4. Use token as password when pushing

---

## 📊 Performance

**With this setup:**
- ✅ Fast development (local C: drive)
- ✅ Data redundancy (Google Drive cloud + offline)
- ✅ Code version control (GitHub)
- ✅ No performance loss (G: drive is offline/local)

**Processing speed depends on:**
- GPU vs CPU (GPU is 10x faster)
- Whisper model size (tiny = fastest, large = most accurate)
- Audio length

---

## 🎓 For Your Capstone Presentation

**Highlight this setup as best practices:**
1. ✅ Version control with Git/GitHub
2. ✅ Separation of code and data
3. ✅ Cloud backup strategy
4. ✅ Environment-based configuration
5. ✅ Local-first performance, cloud-backed redundancy

---

## 📞 Need Help?

If something goes wrong:
1. Check `app.log` file for errors
2. Verify `.env` configuration
3. Test config with: `python config.py`
4. Check GitHub repo is up to date: `git status`

---

**Last Updated:** December 21, 2025
**GitHub Repo:** https://github.com/ak2173-oss/blackbox
