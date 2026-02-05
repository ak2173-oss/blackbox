# pipeline.py - Improved Audio Processing Pipeline
# GPU-enabled, retrieval-based summarization with qwen2.5

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


class AudioPipeline:
    """Improved audio processing pipeline with GPU support and qwen2.5:7b-instruct"""

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

    def transcribe_with_parakeet(self):
        """Transcribe using Wav2Vec2 model (alternative to Whisper)"""
        self.update_status('Transcription', 30, 'Loading Wav2Vec2 model...', f"Model: {Config.PARAKEET_MODEL}")

        try:
            import torch
            import torchaudio
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            # Get device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.update_status('Transcription', 35, 'Initializing Wav2Vec2...', f"Device: {device}")

            # Load model and processor
            processor = Wav2Vec2Processor.from_pretrained(Config.PARAKEET_MODEL)
            model = Wav2Vec2ForCTC.from_pretrained(Config.PARAKEET_MODEL).to(device)
            model.eval()

            self.update_status('Transcription', 40, 'Loading audio...', f"File: {self.audio_file.name}")

            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(str(self.audio_file))

            # Resample if needed (Wav2Vec2 expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            self.update_status('Transcription', 50, 'Transcribing with Wav2Vec2...', 'Processing audio chunks')

            # Process in chunks to avoid memory issues
            chunk_length = Config.PARAKEET_CHUNK_LENGTH * sample_rate  # Convert seconds to samples
            total_samples = waveform.shape[1]
            results = []

            for start_idx in range(0, total_samples, chunk_length):
                end_idx = min(start_idx + chunk_length, total_samples)
                chunk = waveform[:, start_idx:end_idx]

                # Process chunk
                inputs = processor(chunk.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)

                with torch.no_grad():
                    logits = model(inputs.input_values.to(device)).logits

                # Decode
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]

                # Calculate timestamps
                start_time = start_idx / sample_rate
                end_time = end_idx / sample_rate

                if transcription.strip():
                    results.append({
                        "start": round(start_time, 1),
                        "end": round(end_time, 1),
                        "text": transcription.strip()
                    })

                # Update progress
                progress = 50 + int((end_idx / total_samples) * 20)
                self.update_status('Transcription', progress, 'Processing chunks...', f"{end_idx}/{total_samples} samples")

            # Save results
            transcript_dir = self.project_dir / "transcripts"
            with open(transcript_dir / "transcript.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            with open(transcript_dir / "transcript.txt", "w", encoding="utf-8") as f:
                f.write(" ".join([seg["text"] for seg in results]))

            self.update_status('Transcription', 70, 'Transcription complete', f"Found {len(results)} segments")
            logger.info(f"Wav2Vec2 transcribed {len(results)} segments")
            self.log_time("transcription")

            return results

        except ImportError as e:
            raise Exception(f"Wav2Vec2 dependencies missing. Install: pip install transformers torch torchaudio. Error: {e}")
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription error: {e}", exc_info=True)
            raise Exception(f"Wav2Vec2 transcription failed: {e}")

    def transcribe(self):
        """GPU-enabled transcription with better error handling"""
        # Route to appropriate transcription engine
        if Config.TRANSCRIPTION_ENGINE.lower() == 'parakeet':
            return self.transcribe_with_parakeet()

        # Default: Use Whisper
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

    def _select_key_chunks(self, transcript_with_speakers, max_chunks=10):
        """Select most important chunks from transcript using retrieval strategy"""
        # Score segments by importance (length, speaker changes, position)
        scored_segments = []
        for i, seg in enumerate(transcript_with_speakers):
            score = 0
            # Longer segments are more important
            score += len(seg['text']) / 10
            # First and last segments are important
            if i < 3 or i >= len(transcript_with_speakers) - 3:
                score += 50
            # Segments with speaker changes
            if i > 0 and seg['speaker'] != transcript_with_speakers[i-1]['speaker']:
                score += 20

            scored_segments.append((score, i, seg))

        # Sort by score and select top-k
        scored_segments.sort(reverse=True, key=lambda x: x[0])
        top_segments = sorted(scored_segments[:max_chunks], key=lambda x: x[1])  # Re-sort by original order

        return [seg[2] for seg in top_segments]

    def _format_timestamp(self, seconds):
        """Convert seconds to MM:SS format"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _call_ollama_with_fallback(self, prompt, temperature=0.7):
        """Call Ollama with fallback model support"""
        models = [Config.OLLAMA_MODEL, Config.OLLAMA_FALLBACK_MODEL]

        for model in models:
            try:
                response = requests.post(
                    f"{Config.OLLAMA_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_ctx": Config.OLLAMA_CONTEXT_LENGTH
                        }
                    },
                    timeout=Config.OLLAMA_TIMEOUT
                )

                if response.status_code == 200:
                    return response.json()['response'], model

            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue

        return None, None

    def generate_summary(self, transcript_with_speakers):
        """Generate summary using retrieval approach with timestamp citations"""
        self.update_status('AI Summary', 80, 'Generating summary with Qwen2.5...', 'Selecting key segments')

        # Select top chunks instead of using entire transcript
        key_chunks = self._select_key_chunks(transcript_with_speakers, max_chunks=15)

        # Format chunks with timestamps
        chunks_text = ""
        for chunk in key_chunks:
            timestamp = self._format_timestamp(chunk['start'])
            chunks_text += f"[{timestamp}] {chunk['speaker']}: {chunk['text']}\n\n"

        # Generate title from first few chunks
        self.update_status('AI Summary', 82, 'Generating title...', 'Using retrieval strategy')
        title_prompt = f"""Based on these key excerpts from a conversation, generate a short descriptive title (3-6 words maximum).

KEY EXCERPTS:
{chunks_text[:1500]}

Respond with ONLY the title, nothing else."""

        title = "Untitled"
        try:
            response_text, model_used = self._call_ollama_with_fallback(title_prompt, temperature=0.3)

            if response_text:
                title = response_text.strip()
                title = title.replace('"', '').replace(':', '-')[:50]
                # Remove special characters for folder name
                title = ''.join(c for c in title if c.isalnum() or c in ' -_')
                self.update_status('AI Summary', 85, 'Title generated', f"Title: {title} (Model: {model_used})")
                logger.info(f"Generated title: {title} using {model_used}")
            else:
                logger.warning("Title generation failed with all models")

            self.log_time("title_generation")

        except Exception as e:
            logger.error(f"Title generation error: {e}")

        # Generate summary with timestamp citations
        self.update_status('AI Summary', 87, 'Generating detailed summary...', 'Using retrieval with citations')

        summary_prompt = f"""You are analyzing a conversation. Below are the most important excerpts with timestamps. Provide a structured summary with timestamp citations.

KEY EXCERPTS:
{chunks_text}

Create a comprehensive summary using this EXACT format:

## 📋 OVERVIEW
[2-3 sentence summary of the conversation]

## 🔑 KEY POINTS
• [Main topic 1]
  Citation: [MM:SS] "[relevant quote]" - [Speaker]

• [Main topic 2]
  Citation: [MM:SS] "[relevant quote]" - [Speaker]

• [Main topic 3]
  Citation: [MM:SS] "[relevant quote]" - [Speaker]

## ✅ ACTION ITEMS & DECISIONS
• [Action item or decision]
  Citation: [MM:SS] "[supporting quote]" - [Speaker]

## 👥 PARTICIPANTS
• [Speaker 0]: [Brief description based on excerpts]
• [Speaker 1]: [Brief description based on excerpts]

IMPORTANT:
- Include timestamp citations in [MM:SS] format
- Use direct quotes from the excerpts
- Reference the exact timestamps provided"""

        try:
            summary, model_used = self._call_ollama_with_fallback(
                summary_prompt,
                temperature=Config.OLLAMA_TEMPERATURE
            )

            if summary:
                # Save summary
                summary_dir = self.project_dir / "summary"
                with open(summary_dir / "summary.txt", "w", encoding="utf-8") as f:
                    f.write(summary)

                with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "title": title,
                        "summary": summary,
                        "model": model_used,
                        "approach": "retrieval_with_citations",
                        "chunks_used": len(key_chunks),
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