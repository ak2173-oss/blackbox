# process_audio.py
# Complete pipeline with organized folder structure

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


def create_project_folder(audio_file):
    """Create organized folder structure for this audio file"""

    # Get base name without extension
    audio_path = Path(audio_file)
    base_name = audio_path.stem  # e.g., "testWhisper" from "testWhisper.m4a"

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create folder name: audioname_timestamp
    folder_name = f"{base_name}_{timestamp}"
    project_dir = Path("projects") / folder_name

    # Create folder structure
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "audio").mkdir(exist_ok=True)
    (project_dir / "transcripts").mkdir(exist_ok=True)
    (project_dir / "summary").mkdir(exist_ok=True)

    # Copy audio file to project folder
    shutil.copy2(audio_file, project_dir / "audio" / audio_path.name)

    print(f"Created project folder: {project_dir}")
    return project_dir


def transcribe_audio(audio_file, output_dir):
    """Transcribe audio and save to specified directory"""
    from faster_whisper import WhisperModel

    print("\n1. TRANSCRIBING...")

    # Convert to WAV
    wav_file = "temp_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_file),
        "-ar", "16000", "-ac", "1", wav_file
    ]
    subprocess.run(cmd, capture_output=True)

    # Transcribe
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, info = model.transcribe(wav_file, beam_size=5)

    # Collect results
    results = []
    for segment in segments:
        results.append({
            "start": round(segment.start, 1),
            "end": round(segment.end, 1),
            "text": segment.text.strip()
        })

    # Save transcripts
    transcript_dir = output_dir / "transcripts"

    # Save JSON
    with open(transcript_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save text
    with open(transcript_dir / "transcript.txt", "w", encoding="utf-8") as f:
        f.write(" ".join([seg["text"] for seg in results]))

    # Clean up
    Path(wav_file).unlink(missing_ok=True)

    print(f"   Saved {len(results)} segments to {transcript_dir}")
    return results


def add_speakers(transcript, output_dir):
    """Add speaker identification"""
    print("\n2. IDENTIFYING SPEAKERS...")

    # Simple speaker detection based on gaps
    current_speaker = 0
    for i, seg in enumerate(transcript):
        if i > 0:
            gap = seg['start'] - transcript[i - 1]['end']
            if gap > 2.0:  # 2 second gap = speaker change
                current_speaker = 1 - current_speaker
        seg['speaker'] = f'Speaker {current_speaker}'

    # Merge consecutive segments from same speaker
    merged = []
    for seg in transcript:
        if merged and merged[-1]['speaker'] == seg['speaker']:
            if seg['start'] - merged[-1]['end'] < 1.0:
                merged[-1]['text'] += ' ' + seg['text']
                merged[-1]['end'] = seg['end']
                continue
        merged.append(seg)

    # Save with speakers
    transcript_dir = output_dir / "transcripts"

    with open(transcript_dir / "transcript_with_speakers.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    with open(transcript_dir / "transcript_with_speakers.txt", "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in merged:
            if seg['speaker'] != current_speaker:
                f.write(f"\n[{seg['speaker']}]\n")
                current_speaker = seg['speaker']
            f.write(f"{seg['text']} ")

    print(f"   Identified {len(set(seg['speaker'] for seg in merged))} speakers")
    return merged


def generate_title(transcript_text):
    """Generate a short descriptive title for the conversation"""
    import requests

    prompt = f"""Based on this transcript, generate a SHORT descriptive title (3-6 words) that captures the main topic.
The title should be suitable for a folder name (no special characters).

TRANSCRIPT:
{transcript_text[:2000]}

Respond with ONLY the title, nothing else."""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b-instruct-q4_K_M",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}  # Lower temp for consistent titles
            },
            timeout=60
        )

        if response.status_code == 200:
            title = response.json()['response'].strip()
            # Clean up title for folder name
            title = title.replace('"', '').replace(':', '-').replace('/', '-')
            title = title.replace('\\', '-').replace('*', '').replace('?', '')
            title = title.replace('<', '').replace('>', '').replace('|', '')
            # Limit length and remove extra spaces
            title = ' '.join(title.split())[:50]
            return title
    except:
        pass

    return None


def generate_summary(transcript_with_speakers, output_dir):
    """Generate AI summary and title"""
    print("\n3. GENERATING SUMMARY...")

    import requests

    # Convert to text for LLM
    transcript_text = ""
    current_speaker = None
    for seg in transcript_with_speakers:
        if seg['speaker'] != current_speaker:
            transcript_text += f"\n\n{seg['speaker']}:\n"
            current_speaker = seg['speaker']
        transcript_text += seg['text'] + " "

    # Generate title first
    print("   Generating title...")
    title = generate_title(transcript_text)

    # Create prompt for summary
    prompt = f"""Analyze this meeting transcript and provide:

1. SUMMARY: Brief overview (2-3 paragraphs)
2. ACTION ITEMS: Specific tasks mentioned
3. KEY DECISIONS: Any decisions made
4. MAIN TOPICS: Primary discussion points
5. FOLLOW-UP: Open questions

TRANSCRIPT:
{transcript_text[:8000]}

Format with clear headers."""

    try:
        # Send to Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b-instruct-q4_K_M",
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        if response.status_code == 200:
            summary = response.json()['response']

            # Save summary
            summary_dir = output_dir / "summary"

            with open(summary_dir / "summary.txt", "w", encoding="utf-8") as f:
                f.write(f"MEETING SUMMARY\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write("=" * 50 + "\n\n")
                f.write(summary)

            with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump({
                    "generated_at": datetime.now().isoformat(),
                    "model": "mistral:7b-instruct-q4_K_M",
                    "title": title,
                    "summary": summary
                }, f, indent=2, ensure_ascii=False)

            print(f"   Summary saved to {summary_dir}")
            return summary, title

    except Exception as e:
        print(f"   Warning: Could not generate summary ({e})")
        return None, None


def create_index(project_dir):
    """Create an index.html file for easy viewing"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{project_dir.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
        .speaker {{ color: #0066cc; font-weight: bold; }}
        pre {{ background: #f4f4f4; padding: 15px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>📁 {project_dir.name}</h1>

    <div class="section">
        <h2>📊 Files</h2>
        <ul>
            <li><a href="audio/">Audio Files</a></li>
            <li><a href="transcripts/transcript.txt">Transcript (Plain)</a></li>
            <li><a href="transcripts/transcript_with_speakers.txt">Transcript (With Speakers)</a></li>
            <li><a href="summary/summary.txt">AI Summary</a></li>
        </ul>
    </div>

    <div class="section">
        <h2>📝 Quick View</h2>
        <p>Open the files above for full content.</p>
    </div>

    <p><small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small></p>
</body>
</html>"""

    with open(project_dir / "index.html", "w") as f:
        f.write(html)

    print(f"\n📂 Project saved to: {project_dir}")
    print(f"   Open {project_dir}/index.html to view")


def main(audio_file):
    """Process audio file with organized output"""

    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        return

    print(f"\n{'=' * 60}")
    print(f"PROCESSING: {audio_file}")
    print(f"{'=' * 60}")

    # First, create a temporary project folder
    temp_project_dir = create_project_folder(audio_file)

    # Run pipeline
    transcript = transcribe_audio(audio_file, temp_project_dir)
    transcript_with_speakers = add_speakers(transcript, temp_project_dir)
    summary, title = generate_summary(transcript_with_speakers, temp_project_dir)

    # Rename folder with title if we got one
    if title:
        # Create new folder name with title and date
        timestamp = datetime.now().strftime("%Y%m%d")
        clean_title = title.replace(' ', '_')
        new_folder_name = f"{clean_title}_{timestamp}"
        new_project_dir = temp_project_dir.parent / new_folder_name

        # Rename the folder
        if new_project_dir.exists():
            # If folder exists, add time to make unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_folder_name = f"{clean_title}_{timestamp}"
            new_project_dir = temp_project_dir.parent / new_folder_name

        temp_project_dir.rename(new_project_dir)
        project_dir = new_project_dir
        print(f"\n📁 Folder renamed to: {project_dir.name}")
    else:
        project_dir = temp_project_dir

    # Create index file
    create_index(project_dir)

    print(f"\n{'=' * 60}")
    print(f"✅ COMPLETE!")
    print(f"{'=' * 60}")

    return project_dir


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # List available audio files
        audio_files = list(Path(".").glob("*.m4a")) + list(Path(".").glob("*.mp3")) + list(Path(".").glob("*.wav"))

        if audio_files:
            print("\nAvailable audio files:")
            for i, f in enumerate(audio_files):
                print(f"  {i + 1}. {f.name}")

            choice = input("\nEnter number to process (or filename): ").strip()

            if choice.isdigit() and 1 <= int(choice) <= len(audio_files):
                audio_file = audio_files[int(choice) - 1]
            else:
                audio_file = choice
        else:
            print("No audio files found in current directory")
            sys.exit(1)

    main(audio_file)