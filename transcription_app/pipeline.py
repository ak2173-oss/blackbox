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

    def __init__(self, audio_file, job_id=None, status_callback=None, whisper_model=None, reload_ollama_callback=None):
        self.audio_file = Path(audio_file)
        self.job_id = job_id or 'default'
        self.status_callback = status_callback
        self.whisper_model = whisper_model  # Use pre-loaded model
        self.reload_ollama_callback = reload_ollama_callback  # Callback to reload Ollama before summary
        self.start_time = time.time()
        self.timings = {}
        self.project_dir = None
        self.debug_log = []  # Collect debug entries for saving

        if not self.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

    def add_external_debug(self, step, message, detail=''):
        """Add debug entry from external source (e.g., app.py memory management)"""
        self._add_debug_entry(step, message, detail, include_gpu=True)

    def _get_gpu_memory(self):
        """Get current GPU memory usage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                used, total = result.stdout.strip().split(', ')
                return f"{used}MiB / {total}MiB"
        except:
            pass
        return "N/A"

    def _add_debug_entry(self, step, message, detail='', include_gpu=False):
        """Add entry to debug log"""
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        entry = {
            "timestamp": timestamp,
            "elapsed_sec": round(elapsed, 2),
            "step": step,
            "message": message,
            "detail": detail
        }

        if include_gpu:
            entry["gpu_memory"] = self._get_gpu_memory()

        self.debug_log.append(entry)

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

        # Add to debug log (include GPU info for key steps)
        include_gpu = step in ['Transcription', 'AI Summary', 'Memory Management']
        self._add_debug_entry(step, message, detail, include_gpu=include_gpu)

        logger.info(f"[{step}] {message}: {detail}")

    def log_time(self, stage):
        """Log time for each stage"""
        elapsed = time.time() - self.start_time
        self.timings[stage] = round(elapsed, 1)
        logger.info(f"[{elapsed:.1f}s] Completed: {stage}")

    def create_project_folder(self):
        """Create organized project folder"""
        self.update_status('Folder Creation', 20, 'Creating project folder...', '')

        timestamp = datetime.now().strftime("%d-%b-%Y")
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
                self._add_debug_entry("Transcription", "Loading Whisper model", f"Model: {Config.WHISPER_MODEL}", include_gpu=True)
                logger.info("Loading Whisper model (should be pre-cached)...")
                self.whisper_model = WhisperModel(
                    Config.WHISPER_MODEL,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=Config.WHISPER_CPU_THREADS,
                    num_workers=Config.WHISPER_NUM_WORKERS
                )
                self._add_debug_entry("Transcription", "Whisper model loaded", f"Device: {device}, Compute: {compute_type}", include_gpu=True)
            else:
                self._add_debug_entry("Transcription", "Using pre-loaded Whisper model", "Model was cached", include_gpu=True)

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

            # Unload Whisper from GPU to free VRAM for Ollama
            self._add_debug_entry("Memory Management", "Unloading Whisper model", "Freeing GPU memory", include_gpu=True)
            del self.whisper_model
            self.whisper_model = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass
            self._add_debug_entry("Memory Management", "Whisper model unloaded", "GPU memory freed", include_gpu=True)
            logger.info("Whisper model unloaded from GPU")

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
        title_prompt = f"""You are a professional meeting analyst. Based on these excerpts, generate a concise, descriptive title.

EXCERPTS:
{chunks_text[:1500]}

INSTRUCTIONS:
- Create a title of 3-6 words that captures the main topic or purpose
- Use action-oriented language when appropriate (e.g., "Planning Q4 Marketing Strategy")
- For informal recordings, describe the core subject (e.g., "Product Feedback Discussion")
- Avoid generic titles like "Meeting Notes" or "Discussion"

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

        summary_prompt = f"""You are a professional meeting analyst creating actionable documentation from audio transcripts. Your summaries help teams track decisions, assignments, and follow-ups.

TRANSCRIPT EXCERPTS (with timestamps):
{chunks_text}

RECORDING TYPE DETECTION:
First, identify what type of recording this is:
- Formal meeting (structured agenda, multiple participants)
- Interview (Q&A format, interviewer/interviewee)
- Brainstorm/Discussion (free-flowing ideas, collaborative)
- Voice memo (single speaker, notes to self)
- Phone call (two parties, may be informal)

Then generate a summary using this format:

## 📋 OVERVIEW
**Type:** [Recording type from above]
**Duration context:** [Brief/Extended based on content density]

[2-3 sentence executive summary: What was this about? What was accomplished? What's the significance?]

## 🎯 KEY DECISIONS
[List decisions that were made. For each include WHO decided, WHAT was decided, and any CONDITIONS or DEADLINES mentioned]

• **[Decision topic]**
  - Decision: [What was decided]
  - Made by: [Speaker/Role if identifiable]
  - Conditions: [Any caveats, dependencies, or deadlines mentioned]
  - Citation: [MM:SS] "[supporting quote]" - [Speaker]

[If no clear decisions were made, write: "No explicit decisions captured in this recording."]

## ✅ ACTION ITEMS
[Extract specific tasks or commitments. Be specific about WHO owns WHAT.]

| Owner | Action Item | Due Date | Dependencies | Status |
|-------|-------------|----------|--------------|--------|
| [Name/Speaker] | [Specific task] | [Date if mentioned, else "TBD"] | [Blockers/prerequisites] | Assigned |

Citation: [MM:SS] "[quote where action was assigned]" - [Speaker]

[If no action items, write: "No specific action items identified. This appears to be [informational/exploratory/etc.]."]

## 💬 DISCUSSION SUMMARY
[Key topics discussed, organized by theme]

### [Topic 1]
- Main points: [Bullet points of key arguments/information]
- Challenges raised: [Problems or concerns mentioned]
- Citation: [MM:SS] "[key quote]" - [Speaker]

### [Topic 2]
[Same structure]

## ❓ OPEN QUESTIONS & UNRESOLVED ITEMS
[Questions raised but not answered, items needing follow-up, parking lot items]

• [Question/unresolved item]
  - Context: [Why this matters]
  - Suggested owner: [Who should follow up, if clear]
  - Citation: [MM:SS] "[quote]" - [Speaker]

[If none: "All raised questions were addressed during the recording."]

## 📅 NEXT STEPS & FOLLOW-UPS
[Forward-looking items: scheduled meetings, milestones, check-ins]

• [Next step]
  - When: [Date/timeframe if mentioned]
  - Who: [Responsible party]
  - Purpose: [What this will accomplish]

## 👥 PARTICIPANTS & ROLES
[Identify speakers and their apparent roles based on context]

• **[Speaker 0]**: [Role/perspective - e.g., "Project lead, drove discussion on timelines"]
• **[Speaker 1]**: [Role/perspective - e.g., "Technical expert, provided implementation details"]

[If speaker roles are unclear: "Speaker roles could not be determined from context."]

---
CRITICAL INSTRUCTIONS:
1. **Never invent information** - If something wasn't mentioned, say so explicitly
2. **Use timestamps** - Every major point should have a [MM:SS] citation
3. **Be specific** - Include names, numbers, dates when mentioned
4. **Handle uncertainty** - Use "[unclear]" for inaudible portions, "[Speaker X]" when identity is unknown
5. **Adapt to recording type** - A voice memo won't have "decisions"; an interview may not have "action items"
6. **Preserve nuance** - Note disagreements, concerns, or tentative language ("might", "considering")
7. **Prioritize actionability** - The reader should know exactly what to do after reading this"""

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
                    new_name = f"{clean_title}_{datetime.now().strftime('%d-%b-%Y')}"
                    new_dir = self.project_dir.parent / new_name

                    # Ensure unique name
                    counter = 1
                    while new_dir.exists():
                        new_name = f"{clean_title}_{datetime.now().strftime('%d-%b-%Y')}_{counter}"
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

    def save_debug_log(self):
        """Save comprehensive debug log for troubleshooting"""
        debug_file = self.project_dir / "debug_log.txt"
        json_file = self.project_dir / "debug_log.json"

        # Get final GPU state
        final_gpu = self._get_gpu_memory()

        # Save JSON version for programmatic access
        debug_data = {
            "job_id": self.job_id,
            "audio_file": self.audio_file.name,
            "started_at": datetime.fromtimestamp(self.start_time).isoformat(),
            "completed_at": datetime.now().isoformat(),
            "total_seconds": round(time.time() - self.start_time, 2),
            "config": {
                "whisper_model": Config.WHISPER_MODEL,
                "whisper_device": get_device_config()[0],
                "whisper_compute_type": get_device_config()[1],
                "ollama_model": Config.OLLAMA_MODEL,
                "ollama_url": Config.OLLAMA_URL,
                "transcription_engine": Config.TRANSCRIPTION_ENGINE
            },
            "final_gpu_memory": final_gpu,
            "entries": self.debug_log
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2)

        # Save human-readable version
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("TRANSCRIPTION DEBUG LOG\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Job ID:       {self.job_id}\n")
            f.write(f"Audio File:   {self.audio_file.name}\n")
            f.write(f"Started:      {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Completed:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Time:   {time.time() - self.start_time:.1f} seconds\n\n")

            f.write("-" * 70 + "\n")
            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Whisper Model:    {Config.WHISPER_MODEL}\n")
            f.write(f"Device:           {get_device_config()[0]}\n")
            f.write(f"Compute Type:     {get_device_config()[1]}\n")
            f.write(f"Ollama Model:     {Config.OLLAMA_MODEL}\n")
            f.write(f"Ollama URL:       {Config.OLLAMA_URL}\n")
            f.write(f"Final GPU Memory: {final_gpu}\n\n")

            f.write("-" * 70 + "\n")
            f.write("PROCESSING STEPS\n")
            f.write("-" * 70 + "\n\n")

            for entry in self.debug_log:
                elapsed = entry.get('elapsed_sec', 0)
                step = entry.get('step', '')
                message = entry.get('message', '')
                detail = entry.get('detail', '')
                gpu = entry.get('gpu_memory', '')

                f.write(f"[{elapsed:7.2f}s] [{step}]\n")
                f.write(f"           {message}\n")
                if detail:
                    f.write(f"           Detail: {detail}\n")
                if gpu:
                    f.write(f"           GPU: {gpu}\n")
                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("END OF LOG\n")
            f.write("=" * 70 + "\n")

        logger.info(f"Debug log saved to: {debug_file}")

    def run(self):
        """Run complete pipeline with error handling"""
        try:
            # Log initial state
            self._add_debug_entry("Pipeline", "Starting pipeline", f"Audio: {self.audio_file.name}", include_gpu=True)

            self.create_project_folder()
            transcript = self.transcribe()
            transcript_with_speakers = self.add_speakers(transcript)

            # Reload Ollama before summary generation (if callback provided)
            if self.reload_ollama_callback:
                self._add_debug_entry("Memory Management", "Reloading Ollama for summary", "Callback triggered", include_gpu=True)
                self.reload_ollama_callback()
                self._add_debug_entry("Memory Management", "Ollama reload complete", "Ready for summary", include_gpu=True)

            self.generate_summary(transcript_with_speakers)
            self.save_timing_report()
            self.save_debug_log()
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