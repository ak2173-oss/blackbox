# optimized_pipeline.py
# Complete optimized pipeline with timing and progress tracking

import os
import sys
import json
import time
import shutil
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

# Safe GPU detection
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class AudioPipeline:
    def __init__(self, audio_file):
        self.audio_file = Path(audio_file)
        self.start_time = time.time()
        self.timings = {}
        self.project_dir = None

    def log_time(self, stage):
        """Log time for each stage"""
        elapsed = time.time() - self.start_time
        self.timings[stage] = round(elapsed, 1)
        logging.info(f"[{elapsed:.1f}s] Completed: {stage}")

    def create_project_folder(self):
        """Create organized project folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.audio_file.stem}_{timestamp}"
        self.project_dir = Path("projects") / folder_name

        # Create folder structure
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "audio").mkdir(exist_ok=True)
        (self.project_dir / "transcripts").mkdir(exist_ok=True)
        (self.project_dir / "summary").mkdir(exist_ok=True)

        # Copy audio file
        shutil.copy2(self.audio_file, self.project_dir / "audio" / self.audio_file.name)

        logging.info(f"Created project: {self.project_dir}")
        self.log_time("folder_creation")

    def transcribe(self):
        """Optimized transcription with GPU/CPU detection"""
        logging.info("Starting transcription...")

        # Convert to WAV
        wav_file = "temp_audio.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(self.audio_file),
            "-ar", "16000", "-ac", "1", wav_file,
            "-threads", str(os.cpu_count())
        ]
        subprocess.run(cmd, capture_output=True)
        self.log_time("audio_conversion")

        # Safe GPU detection
        device = "cpu"
        compute_type = "int8"

        if HAS_TORCH:
            try:
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                    logging.info("Using GPU acceleration")
                else:
                    logging.info("Using CPU (no CUDA)")
            except:
                logging.info("Using CPU (torch error)")
        else:
            logging.info("Using CPU (no torch)")

        # Load Whisper with optimizations
        logging.info(f"Loading Whisper with {os.cpu_count()} threads...")
        model = WhisperModel(
            "base.en",
            device=device,
            compute_type=compute_type,
            cpu_threads=os.cpu_count(),
            num_workers=4
        )

        # Transcribe with optimizations
        logging.info("Transcribing...")
        segments, info = model.transcribe(
            wav_file,
            beam_size=1,  # Faster
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500
            )
        )

        # Save results
        results = []
        for segment in segments:
            results.append({
                "start": round(segment.start, 1),
                "end": round(segment.end, 1),
                "text": segment.text.strip()
            })

        # Save to project folder
        transcript_dir = self.project_dir / "transcripts"
        with open(transcript_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(transcript_dir / "transcript.txt", "w", encoding="utf-8") as f:
            f.write(" ".join([seg["text"] for seg in results]))

        # Clean up
        Path(wav_file).unlink(missing_ok=True)

        logging.info(f"Transcribed {len(results)} segments")
        self.log_time("transcription")
        return results

    def add_speakers(self, transcript):
        """Simple speaker detection"""
        logging.info("Adding speaker labels...")

        current_speaker = 0
        for i, seg in enumerate(transcript):
            if i > 0:
                gap = seg['start'] - transcript[i - 1]['end']
                if gap > 2.0:
                    current_speaker = 1 - current_speaker
            seg['speaker'] = f'Speaker {current_speaker}'

        # Merge consecutive segments
        merged = []
        for seg in transcript:
            if merged and merged[-1]['speaker'] == seg['speaker']:
                if seg['start'] - merged[-1]['end'] < 1.0:
                    merged[-1]['text'] += ' ' + seg['text']
                    merged[-1]['end'] = seg['end']
                    continue
            merged.append(seg.copy())

        # Save
        transcript_dir = self.project_dir / "transcripts"
        with open(transcript_dir / "transcript_with_speakers.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        with open(transcript_dir / "transcript_with_speakers.txt", "w", encoding="utf-8") as f:
            current = None
            for seg in merged:
                if seg['speaker'] != current:
                    f.write(f"\n[{seg['speaker']}]\n")
                    current = seg['speaker']
                f.write(f"{seg['text']} ")

        logging.info(f"Identified {len(set(seg['speaker'] for seg in merged))} speakers")
        self.log_time("speaker_detection")
        return merged

    def generate_summary(self, transcript_with_speakers):
        """Generate summary with Ollama"""
        logging.info("Generating summary with Mistral...")

        # Convert to text
        transcript_text = ""
        current_speaker = None
        for seg in transcript_with_speakers:
            if seg['speaker'] != current_speaker:
                transcript_text += f"\n\n{seg['speaker']}:\n"
                current_speaker = seg['speaker']
            transcript_text += seg['text'] + " "

        # Generate title
        title_prompt = f"""Generate a SHORT descriptive title (3-6 words) for this conversation:
{transcript_text[:1500]}

Reply with ONLY the title, nothing else."""

        try:
            # Get title
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral:7b-instruct-q4_K_M",
                    "prompt": title_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=60
            )

            if response.status_code == 200:
                title = response.json()['response'].strip()
                title = title.replace('"', '').replace(':', '-')[:50]
                logging.info(f"Generated title: {title}")
            else:
                title = "Untitled"

            self.log_time("title_generation")

            # Get summary
            summary_prompt = f"""Analyze this transcript and provide:
1. SUMMARY: Brief overview (2-3 paragraphs)
2. ACTION ITEMS: Tasks mentioned
3. KEY DECISIONS: Decisions made
4. MAIN TOPICS: Discussion points

TRANSCRIPT:
{transcript_text[:8000]}"""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral:7b-instruct-q4_K_M",
                    "prompt": summary_prompt,
                    "stream": False
                },
                timeout=180
            )

            if response.status_code == 200:
                summary = response.json()['response']

                # Save summary
                summary_dir = self.project_dir / "summary"
                with open(summary_dir / "summary.txt", "w", encoding="utf-8") as f:
                    f.write(summary)

                with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "title": title,
                        "summary": summary,
                        "generated_at": datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)

                # Rename folder with title
                if title and title != "Untitled":
                    clean_title = title.replace(' ', '_').replace('/', '-').replace('\\', '-')
                    new_name = f"{clean_title}_{datetime.now().strftime('%Y%m%d')}"
                    new_dir = self.project_dir.parent / new_name
                    if not new_dir.exists():
                        self.project_dir.rename(new_dir)
                        self.project_dir = new_dir
                        logging.info(f"Renamed to: {new_dir}")

            self.log_time("summary_generation")

        except Exception as e:
            logging.error(f"Summary error: {e}")

    def save_timing_report(self):
        """Save timing report"""
        total_time = time.time() - self.start_time

        report = {
            "total_seconds": round(total_time, 1),
            "total_minutes": round(total_time / 60, 1),
            "stages": self.timings,
            "audio_file": self.audio_file.name,
            "processed_at": datetime.now().isoformat()
        }

        # Save to project
        with open(self.project_dir / "processing_time.json", "w") as f:
            json.dump(report, f, indent=2)

        # Create readable report
        with open(self.project_dir / "processing_time.txt", "w") as f:
            f.write(f"Processing Report\n")
            f.write(f"================\n")
            f.write(f"Audio File: {self.audio_file.name}\n")
            f.write(f"Total Time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)\n\n")
            f.write(f"Stage Timings:\n")
            for stage, time_elapsed in self.timings.items():
                f.write(f"  {stage}: {time_elapsed:.1f}s\n")

        logging.info(f"\n{'=' * 50}")
        logging.info(f"PIPELINE COMPLETE")
        logging.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        logging.info(f"Project: {self.project_dir}")
        logging.info(f"{'=' * 50}\n")

    def run(self):
        """Run complete pipeline"""
        try:
            self.create_project_folder()
            transcript = self.transcribe()
            transcript_with_speakers = self.add_speakers(transcript)
            self.generate_summary(transcript_with_speakers)
            self.save_timing_report()
            return str(self.project_dir)
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        pipeline = AudioPipeline(audio_file)
        project_dir = pipeline.run()
        print(f"Success: {project_dir}")
    else:
        print("Usage: python optimized_pipeline.py <audio_file>")