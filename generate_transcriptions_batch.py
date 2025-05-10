"""
batch_transcripts.py
────────────────────
Generate 1 032 high-quality single-mic encounter transcripts with the
**OpenAI Batch API** (GPT-4o, bulk pricing).

Flow
----
1. Build demographic rows locally (no LLM calls).
2. Write `batch_input.jsonl` – one /v1/chat/completions request per row.
3. Upload file, launch batch job (`completion_window = "24h"` for 50 % discount).
4. Poll until status == "completed".
5. Download `batch_output.jsonl`, merge transcripts back into the dataframe.
6. Emit `synthetic_patients.jsonl` (1 032 rows, ready for next pipeline step).

All tunable numbers live in the CONFIG dict.
"""

from __future__ import annotations

import json, random, time
from pathlib import Path
from typing import Dict, List
import os
import pandas as pd
from openai import OpenAI, OpenAIError

openai_key = os.getenv("OPENAI_API_KEY")

# ╔════════════════════════════════╗
# ║            CONFIG              ║
# ╚════════════════════════════════╝
CONFIG: Dict = {
    # API / model
    "API_KEY":         openai_key,
    "MODEL_TX":        "gpt-4o",      # full-strength model
    "TX_TEMP":         0.75,

    # Dataset
    "TARGET_ROWS":     1_032,
    "TX_MIN_WORDS":    600,
    "TX_MAX_WORDS":    1200,

    # Batch settings
    "BATCH_POLL_SEC":  30,            # polling interval (sec)
    "COMPLETION_WIN":  "24h",         # cheapest bulk tier
    "OUT_DIR":         "batch_files", # dir for input / output files
}

# ╔════════════════════════════════╗
# ║        STATIC INPUTS           ║
# ╚════════════════════════════════╝
BASE_CC: List[str] = json.load(open("chief_complaints.json", encoding="utf-8"))

# quick random helpers
rand_bp   = lambda: f"{random.randint(100,180)}/{random.randint(60,110)}"
rand_age  = lambda: random.randint(18, 90)
rand_gen  = lambda: random.choice(["Male", "Female", "Non-binary"])
rand_yn   = lambda: random.choice(["Yes", "No"])
rand_cc   = lambda: random.choice(BASE_CC)

client = OpenAI(api_key=CONFIG["API_KEY"])
OUT_DIR = Path(CONFIG["OUT_DIR"]).resolve()
OUT_DIR.mkdir(exist_ok=True)

# ───── prompt template ───── #
_PROMPT_TX_TEMPLATE = """
You are transcribing a 1-mic recording of a family-practice visit.

Guidelines:
• Continuous dialogue, no speaker labels.
• Natural English; occasional fillers (“uh-huh”, “okay”).
• {min_w} – {max_w} words total.
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
!Return rule important: You will not use facing and closing like ``` JSON or anything similar, 
in your response, only a valid plain JSON.
""".strip().format(min_w=CONFIG["TX_MIN_WORDS"], max_w=CONFIG["TX_MAX_WORDS"])

# ╔═════════════════════════════════╗
# ║ 1. build local demographic rows ║
# ╚═════════════════════════════════╝
def build_rows(n: int) -> pd.DataFrame:
    rows = []

    for i in range(CONFIG["TARGET_ROWS"]):
        gender  = rand_gen()
        age     = rand_age()
        smoker  = rand_yn()
        drinker = rand_yn()
        bp      = rand_bp()
        cc      = rand_cc()

        rows.append({
            "row_id":         f"row{i:04d}",
            "gender":         gender,
            "age":            age,
            "smoker":         smoker,
            "drinker":        drinker,
            "blood_pressure": bp,
            "chief_complaint": cc,
        })
    return pd.DataFrame(rows)

df = build_rows(CONFIG["TARGET_ROWS"])

# ╔═════════════════════════════════╗
# ║ 2. write batch_input.jsonl      ║
# ╚═════════════════════════════════╝
batch_in = OUT_DIR / "batch_input.jsonl"
with batch_in.open("w", encoding="utf-8") as fh:
    for r in df.to_dict("records"):
        prompt = _PROMPT_TX_TEMPLATE.format(
            gender=r["gender"], age=r["age"], cc=r["chief_complaint"],
            smoker=r["smoker"], drinker=r["drinker"], bp=r["blood_pressure"]
        )
        body = {
            "model": CONFIG["MODEL_TX"],
            "temperature": CONFIG["TX_TEMP"],
            "messages": [{"role": "system", "content": prompt}],
        }
        fh.write(json.dumps({
            "custom_id": r["row_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }) + "\n")
print(f"✓ wrote {batch_in} ({CONFIG['TARGET_ROWS']} requests)")

# ╔═════════════════════════════════╗
# ║ 3. upload file & launch batch   ║
# ╚═════════════════════════════════╝
file_obj = client.files.create(file=batch_in, purpose="batch")
batch_job = client.batches.create( 
    input_file_id=file_obj.id,
    endpoint="/v1/chat/completions",
    completion_window=CONFIG["COMPLETION_WIN"],
)
print("Batch id =", batch_job.id)

# ╔════════════════════════════════╗
# ║ 4. poll until completed        ║
# ╚════════════════════════════════╝
while True:
    batch_job = client.batches.retrieve(batch_job.id)
    status = batch_job.status
    if status in {"failed", "expired"}:
        raise RuntimeError(f"Batch {status}: {batch_job}")
    if status == "completed":
        print("✓ Batch completed")
        break
    print(f"...{status}, waiting {CONFIG['BATCH_POLL_SEC']} s")
    time.sleep(CONFIG["BATCH_POLL_SEC"])

# ╔════════════════════════════════╗
# ║ 5. download output + merge     ║
# ╚════════════════════════════════╝
output_file = client.files.retrieve(batch_job.output_file_id)
out_path = OUT_DIR / "batch_output.jsonl"
client.files.download_to_file(output_file.id, str(out_path))
print("✓ downloaded", out_path)

tx_map: Dict[str, str] = {}
with out_path.open(encoding="utf-8") as fh:
    for line in fh:
        obj = json.loads(line)
        if obj.get("response", {}).get("status_code") == 200:
            content = json.loads(obj["response"]["body"])
            raw_json = content["choices"][0]["message"]["content"]
            tx_map[obj["custom_id"]] = json.loads(raw_json)["transcription"]

if len(tx_map) != CONFIG["TARGET_ROWS"]:
    missing = CONFIG["TARGET_ROWS"] - len(tx_map)
    raise RuntimeError(f"{missing} rows missing in batch output")

df["transcription"] = df["row_id"].map(tx_map)
final_path = "synthetic_patients.jsonl"
df.to_json(final_path, orient="records", lines=True, force_ascii=False)
print("✓ Final file →", final_path)
