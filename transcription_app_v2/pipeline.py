# pipeline.py - Improved Audio Processing Pipeline
# GPU-enabled, better error handling, phi3 integration

import os
import sys
import json
import time
import shutil
import subprocess
import requests
from pathlib import Path
from datetime import datetime
import logging

# Import config
from config import Config, get_device_config

logger = logging.getLogger(__name__)

# WSL Ollama helper
def call_ollama(model, prompt, temperature=0.7):
    """Call Ollama - works in both WSL and native Linux"""
    import platform
    import subprocess as sp

    # Check if we're in WSL
    is_wsl = 'microsoft' in platform.uname().release.lower()

    if is_wsl:
        # Use Windows Ollama executable directly
        ollama_exe = "/mnt/c/Users/Agneya/AppData/Local/Programs/Ollama/ollama.exe"
        if Path(ollama_exe).exists():
            try:
                # Call ollama run with the prompt
                result = sp.run(
                    [ollama_exe, "run", model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    logger.error(f"Ollama error: {result.stderr}")
                    return None
            except Exception as e:
                logger.error(f"Ollama execution error: {e}")
                return None

    # Fallback to HTTP API
    try:
        response = requests.post(
            f"{Config.OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=180
        )
        if response.status_code == 200:
            return response.json()['response']
    except Exception as e:
        logger.error(f"Ollama HTTP error: {e}")

    return None


class AudioPipeline:
    """Improved audio processing pipeline with GPU support and phi3"""

    def __init__(self, audio_file, job_id=None, status_callback=None, whisper_model=None):
        self.audio_file = Path(audio_file)
        self.job_id = job_id or 'default'
        self.status_callback = status_callback
        self.whisper_model = whisper_model  # Use pre-loaded model
        self.start_time = time.time()
        self.timings = {}
        self.project_dir = None

        if not self.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

    def update_status(self, step, progress, message, detail=''):
        """Update processing status"""
        if self.status_callback:
            self.status_callback(
                self.job_id,
                status='processing',
                step=step,
                progress=progress,
                message=message,
                detail=detail
            )
        logger.info(f"[{step}] {message}: {detail}")

    def log_time(self, stage):
        """Log time for each stage"""
        elapsed = time.time() - self.start_time
        self.timings[stage] = round(elapsed, 1)
        logger.info(f"[{elapsed:.1f}s] Completed: {stage}")

    def create_project_folder(self):
        """Create organized project folder"""
        self.update_status('Folder Creation', 20, 'Creating project folder...', '')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.audio_file.stem}_{timestamp}"
        self.project_dir = Path(Config.PROJECTS_FOLDER) / folder_name

        # Create folder structure
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "audio").mkdir(exist_ok=True)
        (self.project_dir / "transcripts").mkdir(exist_ok=True)
        (self.project_dir / "summary").mkdir(exist_ok=True)

        # Copy audio file (use copy instead of copy2 for G: drive compatibility)
        try:
            shutil.copy(self.audio_file, self.project_dir / "audio" / self.audio_file.name)
            self.update_status('Folder Creation', 25, 'Project folder created', f"Path: {self.project_dir}")
        except Exception as e:
            logger.error(f"Failed to copy audio file: {e}")
            raise

        self.log_time("folder_creation")

    def transcribe(self):
        """GPU-enabled transcription with better error handling"""
        self.update_status('Transcription', 30, 'Converting audio...', 'Preparing audio for transcription')

        # Convert to WAV
        wav_file = self.project_dir / "temp_audio.wav"
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(self.audio_file),
                "-ar", str(Config.AUDIO_SAMPLE_RATE),
                "-ac", str(Config.AUDIO_CHANNELS),
                wav_file.name,
                "-threads", str(Config.WHISPER_CPU_THREADS),
                "-loglevel", "error"
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_dir,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")

            self.update_status('Transcription', 35, 'Audio converted', f"Sample rate: {Config.AUDIO_SAMPLE_RATE}Hz")
            self.log_time("audio_conversion")

        except subprocess.TimeoutExpired:
            raise Exception("Audio conversion timed out (>10 minutes)")
        except Exception as e:
            raise Exception(f"Audio conversion failed: {e}")

        # Get device config
        device, compute_type = get_device_config()

        self.update_status(
            'Transcription',
            40,
            'Transcribing with Whisper...',
            f"Model: {Config.WHISPER_MODEL}, Device: {device}, Compute: {compute_type}"
        )

        try:
            # Use pre-loaded model if available, otherwise load it
            if self.whisper_model is None:
                from faster_whisper import WhisperModel
                logger.info("Loading Whisper model (should be pre-cached)...")
                self.whisper_model = WhisperModel(
                    Config.WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=Config.WHISPER_CPU_THREADS,
                    num_workers=Config.WHISPER_NUM_WORKERS
                )

            # Transcribe with optimizations
            segments, info = self.whisper_model.transcribe(
                str(wav_file),
                beam_size=1,  # Faster beam search
                best_of=1,
                temperature=0.0,  # Deterministic
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=500
                )
            )

            self.update_status('Transcription', 50, 'Processing segments...', f"Language: {info.language}")

            # Collect results
            results = []
            segment_count = 0
            for segment in segments:
                results.append({
                    "start": round(segment.start, 1),
                    "end": round(segment.end, 1),
                    "text": segment.text.strip()
                })
                segment_count += 1

                # Update progress every 10 segments
                if segment_count % 10 == 0:
                    self.update_status(
                        'Transcription',
                        50 + min(20, segment_count // 10),
                        'Processing segments...',
                        f"Processed {segment_count} segments"
                    )

            # Save results
            transcript_dir = self.project_dir / "transcripts"
            with open(transcript_dir / "transcript.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            with open(transcript_dir / "transcript.txt", "w", encoding="utf-8") as f:
                f.write(" ".join([seg["text"] for seg in results]))

            # Clean up temp file
            wav_file.unlink(missing_ok=True)

            self.update_status('Transcription', 70, 'Transcription complete', f"Found {len(results)} segments")
            logger.info(f"Transcribed {len(results)} segments")
            self.log_time("transcription")

            return results

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            raise Exception(f"Transcription failed: {e}")

    def add_speakers(self, transcript):
        """Simple speaker detection with configurable threshold"""
        self.update_status('Speaker Detection', 72, 'Detecting speakers...', 'Analyzing speech patterns')

        current_speaker = 0
        speaker_changes = 0

        for i, seg in enumerate(transcript):
            if i > 0:
                gap = seg['start'] - transcript[i - 1]['end']
                if gap > Config.SPEAKER_GAP_THRESHOLD:
                    current_speaker = 1 - current_speaker
                    speaker_changes += 1
            seg['speaker'] = f'Speaker {current_speaker}'

        # Merge consecutive segments
        merged = []
        for seg in transcript:
            if merged and merged[-1]['speaker'] == seg['speaker']:
                if seg['start'] - merged[-1]['end'] < Config.SPEAKER_MERGE_THRESHOLD:
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

        speakers_count = len(set(seg['speaker'] for seg in merged))
        self.update_status(
            'Speaker Detection',
            75,
            'Speakers identified',
            f"{speakers_count} speakers, {speaker_changes} changes"
        )
        logger.info(f"Identified {speakers_count} speakers with {speaker_changes} changes")
        self.log_time("speaker_detection")

        return merged

    def generate_summary(self, transcript_with_speakers):
        """Generate summary with Phi-3"""
        self.update_status('AI Summary', 80, 'Generating summary with Phi-3...', 'Preparing transcript')

        # Convert to text
        transcript_text = ""
        current_speaker = None
        for seg in transcript_with_speakers:
            if seg['speaker'] != current_speaker:
                transcript_text += f"\n\n{seg['speaker']}:\n"
                current_speaker = seg['speaker']
            transcript_text += seg['text'] + " "

        # Limit context size (phi3 has smaller context window than mistral)
        max_context = 4000  # characters
        if len(transcript_text) > max_context:
            transcript_text = transcript_text[:max_context] + "\n\n[Transcript truncated for summary generation]"

        # Generate title
        self.update_status('AI Summary', 82, 'Generating title...', 'Using Phi-3 model')
        title_prompt = f"""Based on this conversation transcript, generate a short descriptive title (3-6 words maximum).

Transcript:
{transcript_text[:1500]}

Respond with ONLY the title, nothing else."""

        title = "Untitled"
        try:
            response_text = call_ollama(Config.OLLAMA_MODEL, title_prompt, temperature=0.3)

            if response_text:
                title = response_text.strip()
                title = title.replace('"', '').replace(':', '-')[:50]
                # Remove special characters for folder name
                title = ''.join(c for c in title if c.isalnum() or c in ' -_')
                self.update_status('AI Summary', 85, 'Title generated', f"Title: {title}")
                logger.info(f"Generated title: {title}")
            else:
                logger.warning("Title generation failed")

            self.log_time("title_generation")

        except Exception as e:
            logger.error(f"Title generation error: {e}")

        # Generate summary
        self.update_status('AI Summary', 87, 'Generating detailed summary...', 'This may take 1-2 minutes')

        summary_prompt = f"""You are analyzing a meeting transcript. Provide a structured summary with direct quotes as evidence.

TRANSCRIPT:
{transcript_text}

Create a comprehensive summary using this EXACT format:

## 📋 OVERVIEW
[2-3 sentence summary of the meeting]

## 🔑 KEY POINTS
• [Main topic 1]
  Quote: "[relevant quote from transcript]" - [Speaker]

• [Main topic 2]
  Quote: "[relevant quote from transcript]" - [Speaker]

• [Main topic 3]
  Quote: "[relevant quote from transcript]" - [Speaker]

## ✅ ACTION ITEMS & DECISIONS
• [Action item or decision]
  Quote: "[supporting quote]" - [Speaker]

## 👥 PARTICIPANTS
• [Speaker 0]: [Brief description of their role/contributions based on what they said]
• [Speaker 1]: [Brief description of their role/contributions based on what they said]

IMPORTANT: Always include direct quotes from the transcript to support each point. Use the exact speaker names from the transcript."""

        try:
            summary = call_ollama(Config.OLLAMA_MODEL, summary_prompt, temperature=Config.OLLAMA_TEMPERATURE)

            if summary:
                # Save summary
                summary_dir = self.project_dir / "summary"
                with open(summary_dir / "summary.txt", "w", encoding="utf-8") as f:
                    f.write(summary)

                with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "title": title,
                        "summary": summary,
                        "model": Config.OLLAMA_MODEL,
                        "generated_at": datetime.now().isoformat()
                    }, f, indent=2, ensure_ascii=False)

                # Rename folder with title
                if title and title != "Untitled":
                    clean_title = title.replace(' ', '_')
                    new_name = f"{clean_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    new_dir = self.project_dir.parent / new_name

                    # Ensure unique name
                    counter = 1
                    while new_dir.exists():
                        new_name = f"{clean_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter}"
                        new_dir = self.project_dir.parent / new_name
                        counter += 1

                    self.project_dir.rename(new_dir)
                    self.project_dir = new_dir
                    self.update_status('AI Summary', 95, 'Summary complete', f"Project: {new_dir.name}")
                    logger.info(f"Renamed to: {new_dir}")

                self.log_time("summary_generation")
            else:
                logger.error("Summary generation failed - no response from Ollama")
                self.update_status('AI Summary', 95, 'Summary failed', 'Continuing without summary')

        except Exception as e:
            logger.error(f"Summary error: {e}")
            self.update_status('AI Summary', 95, 'Summary error', str(e))

    def save_timing_report(self):
        """Save timing report"""
        self.update_status('Finalizing', 98, 'Saving timing report...', '')

        total_time = time.time() - self.start_time

        report = {
            "total_seconds": round(total_time, 1),
            "total_minutes": round(total_time / 60, 1),
            "stages": self.timings,
            "audio_file": self.audio_file.name,
            "processed_at": datetime.now().isoformat(),
            "config": {
                "whisper_model": Config.WHISPER_MODEL,
                "ollama_model": Config.OLLAMA_MODEL,
                "device": get_device_config()[0]
            }
        }

        # Save to project
        with open(self.project_dir / "processing_time.json", "w") as f:
            json.dump(report, f, indent=2)

        # Create readable report
        with open(self.project_dir / "processing_time.txt", "w") as f:
            f.write(f"Processing Report\n")
            f.write(f"================\n")
            f.write(f"Audio File: {self.audio_file.name}\n")
            f.write(f"Total Time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)\n")
            f.write(f"Whisper Model: {Config.WHISPER_MODEL}\n")
            f.write(f"LLM Model: {Config.OLLAMA_MODEL}\n")
            f.write(f"Device: {report['config']['device']}\n\n")
            f.write(f"Stage Timings:\n")
            for stage, time_elapsed in self.timings.items():
                f.write(f"  {stage}: {time_elapsed:.1f}s\n")

        logger.info(f"\n{'=' * 50}")
        logger.info(f"PIPELINE COMPLETE")
        logger.info(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
        logger.info(f"Project: {self.project_dir}")
        logger.info(f"{'=' * 50}\n")

    def run(self):
        """Run complete pipeline with error handling"""
        try:
            self.create_project_folder()
            transcript = self.transcribe()
            transcript_with_speakers = self.add_speakers(transcript)
            self.generate_summary(transcript_with_speakers)
            self.save_timing_report()
            return str(self.project_dir)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            # Clean up on failure
            if self.project_dir and self.project_dir.exists():
                try:
                    shutil.rmtree(self.project_dir)
                except:
                    pass
            raise


if __name__ == "__main__":
    # Test pipeline directly
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        pipeline = AudioPipeline(audio_file)
        project_dir = pipeline.run()
        print(f"✓ Success: {project_dir}")
    else:
        print("Usage: python pipeline.py <audio_file>")