"""
fill_gaps.py — Regenerates pairs for skipped chunks using Groq API.
"""
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY         = os.getenv("GROQ_API_KEY")  # Your Groq key
INPUT_FILE      = "clean_ayurveda_data.md"
OUTPUT_FILE     = "ayurveda_dataset.json"
WORDS_PER_CHUNK = 1200  # Smaller chunks to fit within llama-3.3-70b request limit
PAIRS_PER_CHUNK = 10    # Keep low to stay within 6,000 TPM
SKIPPED_CHUNKS  = [17, 18, 28, 32, 33, 34, 39, 41]  # Manually tracked missing chunks
DELAY_SECONDS   = 35    # 35s between requests keeps us within 6,000 TPM limit
# ─────────────────────────────────────────────────────────────────────────────

client = Groq(api_key=API_KEY)

def chunk_text(text, words_per_chunk):
    words = text.split()
    return [" ".join(words[i:i+words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

def generate_for_chunk(chunk, chunk_num, total):
    prompt = f"""You are an expert Ayurveda Q&A dataset creator.
Read the following Ayurvedic textbook excerpt and generate exactly {PAIRS_PER_CHUNK} high-quality Question and Answer pairs.

Rules:
- Output ONLY a valid JSON array. No markdown, no extra text, no code fences.
- Each object must have exactly two keys: "instruction" (the question) and "output" (the answer).
- Answers must be factual and based strictly on the text provided.
- Vary the question types: definitions, causes, treatments, herbs, doshas, dosages, contraindications.
- Do NOT wrap output in ```json or any other markdown.

Textbook Excerpt:
{chunk}
"""
    for attempt in range(1, 4):
        try:
            print(f"  → Chunk {chunk_num}/{total} | Attempt {attempt}...")
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.7,
            )
            content = response.choices[0].message.content or ""
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if not match:
                print(f"  ⚠ No JSON found. Retrying...")
                time.sleep(5)
                continue
            pairs = json.loads(match.group(0))
            print(f"  ✔ Generated {len(pairs)} pairs for chunk {chunk_num}")
            return pairs
        except json.JSONDecodeError:
            print(f"  ⚠ JSON parse error. Retrying...")
            time.sleep(5)
        except Exception as e:
            err = str(e)
            print(f"  ⚠ Error: {err[:120]}")
            if "429" in err or "rate" in err.lower():
                match2 = re.search(r'retry in (\d+)', err)
                wait = int(match2.group(1)) + 5 if match2 else 60
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif "413" in err:
                print(f"  ✗ Request too large. Skipping chunk {chunk_num}.")
                return []
            else:
                time.sleep(10)
    print(f"  ✗ Chunk {chunk_num} failed after 3 attempts. Skipping.")
    return []

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        book = f.read()
    chunks = chunk_text(book, WORDS_PER_CHUNK)

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)

    print(f"📂 Loaded {len(all_pairs)} existing pairs")
    print(f"📚 Book split into {len(chunks)} chunks at {WORDS_PER_CHUNK} words each")
    print(f"🔄 Regenerating {len(SKIPPED_CHUNKS)} chunks: {SKIPPED_CHUNKS}\n")

    for chunk_num in SKIPPED_CHUNKS:
        # Map original chunk number (2500 words) to new chunk index (1200 words)
        # Original chunk N covers words [(N-1)*2500 : N*2500]
        # Find the matching new chunk index
        original_start_word = (chunk_num - 1) * 2500
        new_chunk_idx = original_start_word // WORDS_PER_CHUNK

        if new_chunk_idx >= len(chunks):
            print(f"⚠ Chunk {chunk_num} out of range. Skipping.")
            continue

        chunk = chunks[new_chunk_idx]
        print(f"🔄 Chunk {chunk_num} (original) → new chunk idx {new_chunk_idx}...")
        new_pairs = generate_for_chunk(chunk, chunk_num, len(SKIPPED_CHUNKS))

        if new_pairs:
            insert_pos = (chunk_num - 1) * 20  # Keep insert position based on original 20 pairs/chunk
            for i, pair in enumerate(new_pairs):
                all_pairs.insert(insert_pos + i, pair)
            print(f"  ✅ Inserted {len(new_pairs)} pairs | Total: {len(all_pairs)}")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_pairs, f, indent=2, ensure_ascii=False)

        print(f"  ⏳ Waiting {DELAY_SECONDS}s before next chunk...")
        time.sleep(DELAY_SECONDS)

    print(f"\n🎉 Done! Final dataset: {len(all_pairs)} pairs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
