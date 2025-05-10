"""
generate_transcriptions.py
──────────────────────────
Creates a synthetic primary-care corpus with columns:

    gender · age · smoker · drinker · blood_pressure
    chief_complaint · transcription

All numeric / enum settings live in the CONFIG dict.
"""

import json, asyncio, random, time
from pathlib import Path
from typing import List, Dict
import os
import pandas as pd           # optional downstream use
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

openai_key = os.getenv("OPENAI_API_KEY")

# ╔════════════════════════════════╗
# ║            CONFIG              ║
# ╚════════════════════════════════╝
CONFIG: Dict = {
    # API & models
    "OPENAI_API_KEY":       openai_key,
    "MODEL_REFINE":         "gpt-4o",
    "MODEL_TX":             "gpt-4o",

    # Dataset size
    "TARGET_ROWS":          1_032, #1_032

    # Concurrency & checkpointing
    "BATCH_SIZE":           10,         # concurrent LLM calls
    "CHECKPOINT_ROWS":      100,
    "CHECKPOINT_DIR":       "checkpoints",

    # Retry settings
    "RETRY_ATTEMPTS":       6,
    "RETRY_MIN_WAIT":       1,          # seconds
    "RETRY_MAX_WAIT":       20,         # seconds

    # Sampling parameters
    "TEMP_REFINE":          0.9,
    "PENALTY_REFINE":       1.0,
    "TEMP_TX":              0.75,

    # Transcript length (words) — used only in prompt text
    "TX_WORDS_MIN":         600,
    "TX_WORDS_MAX":         1200,
}

# ╔════════════════════════════════╗
# ║        STATIC INPUTS           ║
# ╚════════════════════════════════╝
with open("chief_complaints.json", encoding="utf-8") as f:
    BASE_COMPLAINTS: List[str] = json.load(f)

FIELDS = [
    "gender", "age", "smoker", "drinker",
    "blood_pressure", "chief_complaint", "transcription"
]
UNIQUE_KEY = tuple(FIELDS)

# ╔════════════════════════════════╗
# ║        RANDOM HELPERS          ║
# ╚════════════════════════════════╝
rand_bp      = lambda: f"{random.randint(100,180)}/{random.randint(60,110)}"
rand_age     = lambda: random.randint(18, 90)
rand_gender  = lambda: random.choice(["Male", "Female", "Non-binary"])
rand_yes_no  = lambda: random.choice(["Yes", "No"])
rand_base_cc = lambda: random.choice(BASE_COMPLAINTS)

# ╔════════════════════════════════╗
# ║          OPENAI CLIENT         ║
# ╚════════════════════════════════╝
client = AsyncOpenAI(api_key=CONFIG["OPENAI_API_KEY"])

# ───── prompt templates ───── #
PROMPT_REFINE = """
You are refining a primary-care chief complaint.

Given:
  gender: {gender}
  age: {age}
  baseline_complaint: "{base}"

Return ONLY a JSON object on ONE line:
{{"chief_complaint": "< ≤8 words, stays on topic >"}}
"""

PROMPT_TX = f"""
You are transcribing a 1-mic recording of a real-time office visit
between a family-practice physician and an adult patient.

Guidelines:
• Write continuous dialogue, no labels like “Patient:” or “Doctor:”.
• Natural, conversational English; occasional pauses (“uh-huh”, “okay”).
• {CONFIG['TX_WORDS_MIN']} – {CONFIG['TX_WORDS_MAX']} words total.
• Include history-taking, brief exam cues, initial plan.
• No PHI, no brand drug names, no mention of this being synthetic.

Facts you MUST weave in naturally:
- gender: {{gender}}
- age: {{age}}
- chief complaint: "{{cc}}"
- smoker: {{smoker}}, drinker: {{drinker}}, BP: {{bp}}

Important: You will be creative, producing a unique narrative based on the facts presented, 
you will trive your best to create a natural encouter that no one never heard about before.

Return ONLY a JSON object on ONE line:
{{{{"transcription": "full text here"}}}}
!Important: You will not use facing and closing like ``` JSON or anything similar, 
only a valid plain JSON
"""

# ╔════════════════════════════════╗
# ║         LLM FUNCTIONS          ║
# ╚════════════════════════════════╝
retry_decorator = retry(
    stop=stop_after_attempt(CONFIG["RETRY_ATTEMPTS"]),
    wait=wait_random_exponential(
        min=CONFIG["RETRY_MIN_WAIT"],
        max=CONFIG["RETRY_MAX_WAIT"]
    ),
)

@retry_decorator
async def refine_complaint(gender: str, age: int, base: str) -> str:
    prompt = PROMPT_REFINE.format(gender=gender, age=age, base=base)
    resp = await client.chat.completions.create(
        model=CONFIG["MODEL_REFINE"],
        temperature=CONFIG["TEMP_REFINE"],
        presence_penalty=CONFIG["PENALTY_REFINE"],
        messages=[{"role": "system", "content": prompt}]
    )
    return json.loads(resp.choices[0].message.content)["chief_complaint"]

@retry_decorator
async def make_transcript(rec: dict) -> str:
    prompt = PROMPT_TX.format(
        gender=rec["gender"], age=rec["age"], cc=rec["chief_complaint"],
        smoker=rec["smoker"], drinker=rec["drinker"], bp=rec["blood_pressure"]
    )
    resp = await client.chat.completions.create(
        model=CONFIG["MODEL_TX"],
        temperature=CONFIG["TEMP_TX"],
        messages=[{"role": "system", "content": prompt}]
    )
    return json.loads(resp.choices[0].message.content)["transcription"]

# ╔════════════════════════════════╗
# ║        JSONL UTILITIES         ║
# ╚════════════════════════════════╝
def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# ╔════════════════════════════════╗
# ║       RECORD GENERATOR         ║
# ╚════════════════════════════════╝
async def generate_one() -> dict:
    rec = {
        "gender":         rand_gender(),
        "age":            rand_age(),
        "smoker":         rand_yes_no(),
        "drinker":        rand_yes_no(),
        "blood_pressure": rand_bp(),
    }
    rec["chief_complaint"] = await refine_complaint(
        rec["gender"], rec["age"], rand_base_cc()
    )
    rec["transcription"]   = await make_transcript(rec)
    return rec

# ╔════════════════════════════════╗
# ║            MAIN LOOP           ║
# ╚════════════════════════════════╝
async def main():
    out_dir = Path(CONFIG["CHECKPOINT_DIR"])
    out_dir.mkdir(exist_ok=True)
    rows, seen = [], set()
    chunk_idx  = 0
    start_time = time.time()

    while len(rows) < CONFIG["TARGET_ROWS"]:
        batch_size = min(CONFIG["BATCH_SIZE"], CONFIG["TARGET_ROWS"] - len(rows))
        batch_jobs = [generate_one() for _ in range(batch_size)]

        for rec in await asyncio.gather(*batch_jobs):
            key = tuple(rec[f] for f in UNIQUE_KEY)
            if key in seen:
                continue
            rows.append(rec)
            seen.add(key)

        # checkpoint logic
        if len(rows) // CONFIG["CHECKPOINT_ROWS"] > chunk_idx:
            chunk_idx += 1
            chunk_path = out_dir / f"patients_{chunk_idx:03d}.jsonl"
            write_jsonl(chunk_path, rows[-CONFIG["CHECKPOINT_ROWS"]:])
            print(f"✓ saved {chunk_path}  ({len(rows)}/{CONFIG['TARGET_ROWS']})")

    # final merged file
    master_path = Path("synthetic_patients.jsonl")
    write_jsonl(master_path, rows)
    elapsed = time.time() - start_time
    print(f"\nGenerated {len(rows)} rows → {master_path}  in {elapsed/60:.1f} min.")

if __name__ == "__main__":
    asyncio.run(main())
