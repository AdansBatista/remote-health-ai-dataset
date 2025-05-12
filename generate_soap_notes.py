"""
generate_soap_notes.py.py
────────────────────────
Create SOAP notes by pairing lines from:
  batch_files/batch_input.jsonl   (facts in prompt)
  batch_files/batch_output.jsonl  (transcription)

No merge dataframe needed – we extract data on the fly, match by custom_id,
and run a second Batch job to get JSON-structured SOAP notes.

Requires:
  pip install openai python-dotenv regex
"""

from __future__ import annotations
import json, re, time, os, sys
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# ──────────────── EDIT IF NEEDED ───────────────── #
BATCH_DIR      = Path("batch_files")          # where input/output jsonl live
INPUT_TX_FILE  = BATCH_DIR / "batch_input.jsonl"
OUTPUT_TX_FILE = BATCH_DIR / "batch_output.jsonl"

SOAP_DIR       = Path("soap_batch")           # new folder for SOAP job
SOAP_IN_FILE   = SOAP_DIR / "soap_input.jsonl"
SOAP_OUT_FILE  = SOAP_DIR / "soap_output.jsonl"
MASTER_OUT     = "patients_with_soap.jsonl"
# ───────────────────────────────────────────────── #

# 1 ▸ load key
load_dotenv(); api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("❌  OPENAI_API_KEY missing in .env")
print("🔑 key ok")

client = OpenAI(api_key=api_key)
SOAP_DIR.mkdir(exist_ok=True, parents=True)

# 2 ▸ regex helpers
re_gender   = re.compile(r"gender:\s*(\w+)", re.I)
re_age      = re.compile(r"age:\s*(\d+)", re.I)
re_smoker   = re.compile(r"smoker:\s*(Yes|No)", re.I)
re_drinker  = re.compile(r"drinker:\s*(Yes|No)", re.I)
re_bp       = re.compile(r"BP:\s*([\d/]+)", re.I)
re_cc       = re.compile(r'chief complaint:\s*"([^"]+)"', re.I)

def parse_facts(prompt: str) -> Dict[str, str]:
    return {
        "gender":  re_gender.search(prompt).group(1),
        "age":     re_age.search(prompt).group(1),
        "smoker":  re_smoker.search(prompt).group(1),
        "drinker": re_drinker.search(prompt).group(1),
        "bp":      re_bp.search(prompt).group(1),
        "cc":      re_cc.search(prompt).group(1),
    }

# 3 ▸ load original facts/prompts
facts_map: Dict[str, Dict[str,str]] = {}
with INPUT_TX_FILE.open(encoding="utf-8") as fh:
    for ln in fh:
        j = json.loads(ln)
        cid = j["custom_id"]
        prompt_text = j["body"]["messages"][0]["content"]
        facts_map[cid] = parse_facts(prompt_text)

# 4 ▸ load transcriptions
tx_map: Dict[str, str] = {}
grab_tx = re.compile(r'{"transcription"\s*:\s*"(.*)"}', re.S)   # ← add this regex

with OUTPUT_TX_FILE.open(encoding="utf-8") as fh:
    for ln in fh:
        j = json.loads(ln)
        if j["response"]["status_code"] == 200:
            raw      = j["response"]["body"]
            content  = raw["choices"][0]["message"]["content"]

            # REPLACE the next two lines ↓↓↓
            # tx = json.loads(content)["transcription"]
            # tx_map[j["custom_id"]] = tx

            # WITH this safer extractor ↓↓↓
            m = grab_tx.match(content)
            if not m:
                raise ValueError(f"Bad JSON in transcription for {j['custom_id']}")
            tx_map[j["custom_id"]] = m.group(1)

missing = set(facts_map) - set(tx_map)
if missing:
    sys.exit(f"❌ {len(missing)} transcriptions missing, cannot proceed")

print(f"✓ paired {len(tx_map)} records")

# 5 ▸ build soap_input.jsonl
template = """
    You are an **expert clinical documentation specialist**.  
    Using the transcript and patient facts below, produce a SOAP note.
    
    Return **ONE valid JSON object on a single line**:
    
    {{
      "subjective": "...",
      "objective": "...",
      "assessment": "...",
      "plan": "..."
    }}
    • Base strictly on the facts + transcription — **no extra keys**.
    
    Facts:
    - gender: {gender}
    - age: {age}
    - smoker: {smoker}, drinker: {drinker}, BP: {bp}
    - chief complaint: "{cc}"
    
    Transcription:
    \"\"\"{tx}\"\"\"
    
    Important: **Do NOT** wrap your response with markdown fences (```), headings, or any explanatory text.  
    Return only the plain JSON object shown above.
""".strip()

with SOAP_IN_FILE.open("w", encoding="utf-8") as fh:
    for cid, facts in facts_map.items():
        prompt = template.format(tx=tx_map[cid], **facts)
        fh.write(json.dumps({
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.7,
                "messages": [{"role": "system", "content": prompt}],
            }
        }) + "\n")
print("✓ wrote", SOAP_IN_FILE)

# 6 ▸ launch batch
file_obj = client.files.create(file=SOAP_IN_FILE, purpose="batch")
batch = client.batches.create(
    input_file_id=file_obj.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
print("Batch id", batch.id)

# 7 ▸ poll
while True:
    batch = client.batches.retrieve(batch.id)
    if batch.status in {"failed","expired"}:
        sys.exit(f"❌ batch {batch.status}")
    if batch.status == "completed":
        break
    print("…", batch.status, "waiting 30 s"); time.sleep(30)

# 8 ▸ download soap_output.jsonl
with SOAP_OUT_FILE.open("wb") as fh:
    for chunk in client.files.content(batch.output_file_id).iter_bytes():
        fh.write(chunk)
print("✓ downloaded", SOAP_OUT_FILE)

# 9 ▸ merge & save master
with SOAP_OUT_FILE.open(encoding="utf-8") as fh, open(MASTER_OUT,"w",encoding="utf-8") as out:
    for ln in fh:
        j = json.loads(ln)
        if j["response"]["status_code"] != 200: continue
        cid = j["custom_id"]
        body = j["response"]["body"]  # already a dict
        soap = json.loads(body["choices"][0]["message"]["content"])
        record = {**facts_map[cid], "transcription": tx_map[cid], **soap, "row_id": cid}
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✓ Final dataset →", MASTER_OUT)
