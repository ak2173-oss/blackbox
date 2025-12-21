# simple_speaker_detection.py
# Simple speaker detection based on gaps and alternation patterns
# No complex dependencies required!

import json
from pathlib import Path


def add_speakers_by_gaps(transcript_file="output/transcript.json", gap_threshold=2.0):
    """
    Simple speaker detection based on pause patterns.
    Assumes speakers change when there's a significant pause.
    """
    print("=" * 50)
    print("SIMPLE SPEAKER DETECTION")
    print("=" * 50)

    # Load transcript
    print("\nLoading transcript...")
    with open(transcript_file, "r", encoding="utf-8") as f:
        segments = json.load(f)
    print(f"Found {len(segments)} segments")

    # Detect speaker changes based on gaps
    print(f"\nDetecting speakers (gap threshold: {gap_threshold}s)...")

    current_speaker = 0
    speaker_changes = 0

    for i, seg in enumerate(segments):
        if i == 0:
            seg['speaker'] = f'Speaker {current_speaker}'
        else:
            # Check gap from previous segment
            gap = seg['start'] - segments[i - 1]['end']

            # Long pause might indicate speaker change
            if gap > gap_threshold:
                current_speaker = 1 - current_speaker  # Alternate between 0 and 1
                speaker_changes += 1
                print(f"  Speaker change detected at {seg['start']:.1f}s (gap: {gap:.1f}s)")

            seg['speaker'] = f'Speaker {current_speaker}'

    print(f"Detected {speaker_changes} speaker changes")

    # Merge consecutive segments from same speaker
    print("\nMerging consecutive segments...")
    merged = []
    for seg in segments:
        if merged and merged[-1]['speaker'] == seg['speaker']:
            # Check if close enough to merge (within 1 second)
            if seg['start'] - merged[-1]['end'] < 1.0:
                merged[-1]['text'] += ' ' + seg['text']
                merged[-1]['end'] = seg['end']
                continue
        merged.append(seg.copy())

    print(f"Merged {len(segments)} segments into {len(merged)} speaker turns")

    # Save results
    print("\nSaving results...")

    # Save JSON
    with open("output/transcript_with_speakers.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # Save readable text
    with open("output/transcript_with_speakers.txt", "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in merged:
            if seg['speaker'] != current_speaker:
                f.write(f"\n[{seg['speaker']}]\n")
                current_speaker = seg['speaker']
            f.write(f"{seg['text']} ")

    # Save timeline
    with open("output/transcript_speakers_timeline.txt", "w", encoding="utf-8") as f:
        for seg in merged:
            f.write(f"[{seg['start']:.1f}-{seg['end']:.1f}s] {seg['speaker']}: {seg['text']}\n")

    print("\n" + "=" * 50)
    print("✅ COMPLETE!")
    print("=" * 50)
    print("\nFiles created:")
    print("  - output/transcript_with_speakers.json")
    print("  - output/transcript_with_speakers.txt")
    print("  - output/transcript_speakers_timeline.txt")

    # Show sample
    print("\nSample output:")
    print("-" * 30)
    for seg in merged[:5]:
        print(f"{seg['speaker']}: {seg['text'][:60]}...")

    # Statistics
    speakers_count = len(set(seg['speaker'] for seg in merged))
    print(f"\nStatistics:")
    print(f"  Total speakers: {speakers_count}")
    print(f"  Total turns: {len(merged)}")

    return merged


if __name__ == "__main__":
    # Run the simple speaker detection
    # Adjust gap_threshold if needed (smaller = more speaker changes)
    add_speakers_by_gaps(gap_threshold=2.0)