"""
augment_soap_sections.py
────────────────────────
Paraphrase every S / O / A / P section K-times **in five different styles**
using the OpenAI Batch API.

Input
-----
soap_batch/soap_output.jsonl   ← 1 032 lines (existing SOAP notes)

Output
------
soap_batch/augment_input.jsonl   (batch requests)
soap_batch/augment_output.jsonl  (batch responses)
soap_augmented.jsonl             (flattened dataset)

For each original note you get:
    5 formats × K paraphrases × 4 sections
    (default K = 2 ⇒ 40 new lines per patient)

Run:
    pip install openai python-dotenv
    python augment_soap_sections.py
"""

from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# ╔════════════════════════════════╗
# ║            CONFIG              ║
# ╚════════════════════════════════╝
CONFIG: Dict = {
    # OpenAI
    "MODEL":           "gpt-4o",
    "TEMP":            0.85,
    "COMPL_WIN":       "24h",
    "POLL_SEC":        30,

    # Paraphrase counts
    "K":               2,                      # ← variations per format (5 × K total)

    # File paths
    "SOAP_OUT_FILE":   Path("soap_batch/soap_output.jsonl"),
    "BATCH_DIR":       Path("soap_batch"),
    "AUG_INPUT":       Path("soap_batch/augment_input.jsonl"),
    "AUG_OUTPUT":      Path("soap_batch/augment_output.jsonl"),
    "FINAL_DATA":      "soap_augmented.jsonl",
}

# ╔════════════════════════════════╗
# ║  Paraphrase-format matrix      ║
# ╚════════════════════════════════╝
FORMAT_VARIANTS: List[tuple[str, str]] = [
    ("simple",   "Rephrase in plain prose; keep structure similar."),
    ("bullets",  "Rewrite as bullet points (-). Each line ≤ 20 words."),
    ("narrative","Expand into a brief narrative paragraph (3-4 sentences)."),
    ("brev_bp",  "Bullet list, line ≤ 70 characters; be concise."),
    ("detailed", "Add clinical nuance and detail while preserving facts."),
]

# ╔════════════════════════════════╗
# ║ 0.  key + client               ║
# ╚════════════════════════════════╝
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("❌  OPENAI_API_KEY missing in .env")
client = OpenAI(api_key=api_key)

CONFIG["BATCH_DIR"].mkdir(exist_ok=True)

# ╔════════════════════════════════╗
# ║ 1.  load SOAP sections         ║
# ╚════════════════════════════════╝
sections: Dict[str, Dict[str, str]] = {}
with CONFIG["SOAP_OUT_FILE"].open(encoding="utf-8") as fh:
    for ln in fh:
        j = json.loads(ln)
        if j["response"]["status_code"] != 200:
            continue
        cid = j["custom_id"]
        body = j["response"]["body"]
        soap = json.loads(body["choices"][0]["message"]["content"])
        sections[cid] = soap
print(f"✓ extracted {len(sections)} SOAP notes")

# ╔════════════════════════════════╗
# ║ 2.  build batch_input.jsonl    ║
# ╚════════════════════════════════╝
template = """
Return {k} alternative phrasings of the **{label}** section as a JSON array
of strings. Preserve clinical meaning; vary wording and sentence structure.

Original:
\"\"\"{text}\"\"\"
""".strip()

with CONFIG["AUG_INPUT"].open("w", encoding="utf-8") as fh:
    for cid, soap in sections.items():                # each patient
        for label, text in soap.items():              # S O A P
            for slug, extra in FORMAT_VARIANTS:
                prompt = template.format(
                    k=CONFIG["K"],
                    label=label.capitalize(),
                    text=text
                ) + "\n\n" + extra

                fh.write(json.dumps({
                    "custom_id": f"{cid}_{label}_{slug}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": CONFIG["MODEL"],
                        "temperature": CONFIG["TEMP"],
                        "messages": [{"role": "system", "content": prompt}],
                    }
                }) + "\n")
print("✓ wrote", CONFIG["AUG_INPUT"])

# ╔════════════════════════════════╗
# ║ 3.  launch batch               ║
# ╚════════════════════════════════╝
file_obj = client.files.create(file=CONFIG["AUG_INPUT"], purpose="batch")
batch = client.batches.create(
    input_file_id=file_obj.id,
    endpoint="/v1/chat/completions",
    completion_window=CONFIG["COMPL_WIN"],
)
print("Batch id", batch.id)

# ╔════════════════════════════════╗
# ║ 4.  poll                       ║
# ╚════════════════════════════════╝
while True:
    batch = client.batches.retrieve(batch.id)
    if batch.status in {"failed", "expired"}:
        sys.exit(f"❌ batch {batch.status}")
    if batch.status == "completed":
        break
    print("…", batch.status, "sleeping", CONFIG["POLL_SEC"], "s")
    time.sleep(CONFIG["POLL_SEC"])

# ╔════════════════════════════════╗
# ║ 5.  download output            ║
# ╚════════════════════════════════╝
with CONFIG["AUG_OUTPUT"].open("wb") as fh:
    for chunk in client.files.content(batch.output_file_id).iter_bytes():
        fh.write(chunk)
print("✓ downloaded", CONFIG["AUG_OUTPUT"])

# ╔════════════════════════════════╗
# ║ 6.  flatten & emit final file  ║
# ╚════════════════════════════════╝
with CONFIG["AUG_OUTPUT"].open(encoding="utf-8") as fh, \
     CONFIG["FINAL_DATA"].open("w", encoding="utf-8") as out:

    for ln in fh:
        j = json.loads(ln)
        if j["response"]["status_code"] != 200:
            continue

        cid, section, slug = j["custom_id"].split("_", 2)
        paras = json.loads(j["response"]["body"]["choices"][0]["message"]["content"])

        for i, alt in enumerate(paras, 1):
            out.write(json.dumps({
                "row_id":  cid,
                "section": section,      # subjective / objective / ...
                "format":  slug,         # simple / bullets / narrative / ...
                "alt_idx": f"{i:02d}",
                "text":    alt.strip()
            }, ensure_ascii=False) + "\n")

print("✓ Final augmented file →", CONFIG["FINAL_DATA"])
