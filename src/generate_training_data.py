import json
import time
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY         = os.getenv("GROQ_API_KEY")
INPUT_FILE      = "clean_ayurveda_data.md"
OUTPUT_FILE     = "ayurveda_dataset.json"
WORDS_PER_CHUNK = 2500      # Words sent per API call
PAIRS_PER_CHUNK = 20        # Q&A pairs to request per chunk
MAX_RETRIES     = 3         # Retry on errors
RETRY_DELAY     = 30        # Seconds to wait on rate limit
# ─────────────────────────────────────────────────────────────────────────────

client = Groq(api_key=API_KEY)

# ── Helpers ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, words_per_chunk: int) -> list:
    """Split text into chunks of roughly `words_per_chunk` words."""
    words = text.split()
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]

def generate_qa_pairs(chunk: str, chunk_num: int, total: int):
    """Call Groq API and return a list of {instruction, output} dicts."""
    prompt = f"""You are an expert Ayurveda Q&A dataset creator.
Read the following Ayurvedic textbook excerpt and generate exactly {PAIRS_PER_CHUNK} high-quality Question and Answer pairs.

Rules:
- Output ONLY a valid JSON array. No markdown, no extra text, no code fences.
- Each object must have exactly two keys: "instruction" (the question) and "output" (the answer).
- Answers must be factual and based strictly on the text provided.
- Vary the question types: definitions, comparisons, causes, treatments, herbs, doshas, dosages, contraindications, etc.
- Do NOT wrap output in ```json or any other markdown.

Textbook Excerpt:
{chunk}
"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  → Chunk {chunk_num}/{total} | Attempt {attempt}...")
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Best free model on Groq
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=0.7,
            )
            content = response.choices[0].message.content or ""
            # Extract the JSON array even if model adds extra text around it
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if not match:
                print(f"  ⚠ No JSON array found in response for chunk {chunk_num}. Skipping.")
                return [], 0
            raw = match.group(0)
            pairs = json.loads(raw)
            tokens_used = response.usage.total_tokens if response.usage else 0
            return pairs, tokens_used

        except json.JSONDecodeError:
            print(f"  ⚠ JSON parse error on chunk {chunk_num}. Skipping.")
            return [], 0
        except Exception as e:
            err = str(e)
            print(f"  ⚠ Error: {err[:150]}...")
            if "429" in err or "rate" in err.lower():
                match = re.search(r'retry in (\d+)', err)
                wait = int(match.group(1)) + 5 if match else RETRY_DELAY
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES:
                print(f"  ⏳ Waiting {RETRY_DELAY}s before retry...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  ✗ Chunk {chunk_num} failed after {MAX_RETRIES} attempts. Skipping.")
                return [], 0
    return [], 0

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Read the markdown file
    print(f"📖 Reading '{INPUT_FILE}'...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        book_content = f.read()
    print(f"   File size: {len(book_content):,} characters")

    # 2. Split into chunks
    chunks = chunk_text(book_content, WORDS_PER_CHUNK)
    print(f"   Split into {len(chunks)} chunks ({WORDS_PER_CHUNK} words each)\n")

    # 3. Load existing results if resuming
    all_pairs = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                all_pairs = json.load(f)
                print(f"▶ Resuming — {len(all_pairs)} pairs already saved.\n")
            except json.JSONDecodeError:
                all_pairs = []

    # 4. Process each chunk
    total_tokens_used = 0
    start_chunk = len(all_pairs) // PAIRS_PER_CHUNK  # Resume from where we left off

    for i, chunk in enumerate(chunks[start_chunk:], start=start_chunk + 1):
        pairs, tokens = generate_qa_pairs(chunk, i, len(chunks))
        total_tokens_used += tokens
        all_pairs.extend(pairs)

        # Save after every chunk so progress is never lost
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_pairs, f, indent=2, ensure_ascii=False)

        print(f"  ✔ {len(pairs)} pairs added | Total: {len(all_pairs)} | Tokens used: {total_tokens_used:,}")
        time.sleep(2)  # Groq is generous — 2s is enough

    # 5. Done
    print(f"\n🎉 Finished!")
    print(f"   Total Q&A pairs generated : {len(all_pairs)}")
    print(f"   Total tokens used          : {total_tokens_used:,}")
    print(f"   Saved to                   : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()