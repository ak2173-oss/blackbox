# summarize_transcript.py
# Uses Ollama with Mistral to summarize the transcript

import json
import requests
from pathlib import Path


def summarize_with_ollama(transcript_file="output/transcript_with_speakers.json"):
    """
    Send transcript to Ollama for summary and action items
    """
    print("=" * 50)
    print("LLM SUMMARIZATION WITH MISTRAL")
    print("=" * 50)

    # Load transcript with speakers
    print("\nLoading transcript...")
    with open(transcript_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Convert to text format for LLM
    transcript_text = ""
    current_speaker = None
    for seg in segments:
        if seg['speaker'] != current_speaker:
            transcript_text += f"\n\n{seg['speaker']}:\n"
            current_speaker = seg['speaker']
        transcript_text += seg['text'] + " "

    print(f"Transcript length: {len(transcript_text)} characters")

    # Create prompt for LLM
    prompt = f"""Please analyze this meeting transcript and provide:

1. SUMMARY: A brief 2-3 paragraph overview of the discussion
2. ACTION ITEMS: List specific tasks that need to be done (with owner if mentioned)
3. KEY DECISIONS: Any decisions that were made
4. MAIN TOPICS: The primary topics discussed
5. FOLLOW-UP NEEDED: Any open questions or items needing clarification

TRANSCRIPT:
{transcript_text[:8000]}

Please structure your response with clear headers for each section."""

    # Send to Ollama
    print("\nSending to Mistral (this may take 30-60 seconds)...")

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral:7b-instruct-q4_K_M",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=180  # 3 minutes timeout for larger model
        )

        if response.status_code == 200:
            result = response.json()
            summary = result['response']

            # Save summary
            print("\nSaving summary...")

            # Save as text
            with open("output/summary.txt", "w", encoding="utf-8") as f:
                f.write("MEETING ANALYSIS - MISTRAL 7B\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Transcript segments: {len(segments)}\n")
                f.write(f"Speakers: {', '.join(set(seg['speaker'] for seg in segments))}\n")
                f.write("-" * 50 + "\n\n")
                f.write(summary)

            # Save as JSON with metadata
            summary_data = {
                "transcript_file": transcript_file,
                "model": "mistral:7b-instruct-q4_K_M",
                "summary": summary,
                "num_segments": len(segments),
                "speakers": list(set(seg['speaker'] for seg in segments))
            }

            with open("output/summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            print("\n" + "=" * 50)
            print("✅ SUMMARY COMPLETE!")
            print("=" * 50)
            print("\nFiles created:")
            print("  - output/summary.txt")
            print("  - output/summary.json")

            print("\n" + "-" * 30)
            print("SUMMARY PREVIEW:")
            print("-" * 30)
            # Show first 500 characters
            if len(summary) > 500:
                print(summary[:500] + "...")
            else:
                print(summary)

            return summary

        else:
            print(f"Error: Ollama returned status {response.status_code}")
            print("Make sure Ollama is running: ollama serve")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to Ollama")
        print("\nPlease make sure Ollama is running.")
        print("Try opening a new terminal and running: ollama serve")
    except requests.exceptions.Timeout:
        print("\n❌ Error: Request timed out")
        print("The model might be taking too long. Try running again.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    summarize_with_ollama()