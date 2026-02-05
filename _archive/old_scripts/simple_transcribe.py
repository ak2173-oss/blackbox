# fast_transcribe.py
# Optimized version with better CPU usage

import subprocess
import json
from pathlib import Path
from faster_whisper import WhisperModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)


def transcribe_audio(audio_file, model_size="base.en"):
    """Optimized transcription"""
    print(f"Transcribing: {audio_file}")

    # Convert to WAV
    wav_file = "temp_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_file),
        "-ar", "16000", "-ac", "1", wav_file
    ]
    subprocess.run(cmd, capture_output=True)

    # Load model with optimizations
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=8,  # Use more CPU threads
        num_workers=4  # Parallel processing
    )

    # Transcribe with optimizations
    segments, info = model.transcribe(
        wav_file,
        beam_size=1,  # Faster beam search
        best_of=1,  # Less sampling
        temperature=0.0,  # Deterministic
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500  # Faster VAD
        )
    )

    # Save results quickly
    results = []
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for segment in segments:
        results.append({
            "start": round(segment.start, 1),
            "end": round(segment.end, 1),
            "text": segment.text.strip()
        })
        print(f".", end="", flush=True)  # Progress indicator

    # Save files
    with open(output_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "transcript.txt", "w", encoding="utf-8") as f:
        f.write(" ".join([seg["text"] for seg in results]))

    # Cleanup
    Path(wav_file).unlink(missing_ok=True)

    print(f"\n✓ Complete: {len(results)} segments")
    return results


if __name__ == "__main__":
    import sys

    audio_file = sys.argv[1] if len(sys.argv) > 1 else "testWhisper.m4a"
    transcribe_audio(audio_file)