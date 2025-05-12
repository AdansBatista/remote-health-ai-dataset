#!/usr/bin/env python
"""
merge_soap_clean.py
───────────────────
Builds a tidy JSON-Lines file with patient facts + SOAP sections.

Inputs
-------
batch_files/batch_input.jsonl      ← original requests (facts in prompt)
soap_batch/soap_output.jsonl       ← batch responses (SOAP JSON)

Output
------
data/synthetic_patients_with_soap.jsonl
"""

import json, re, pathlib

# ---------- helpers ------------------------------------------------------
fact_re = {
    "gender":  re.compile(r"gender:\s*(\w+)", re.I),
    "age":     re.compile(r"age:\s*(\d+)", re.I),
    "smoker":  re.compile(r"smoker:\s*(Yes|No)", re.I),
    "drinker": re.compile(r"drinker:\s*(Yes|No)", re.I),
    "bp":      re.compile(r"BP:\s*([\d/]+)", re.I),
    "cc":      re.compile(r'chief complaint:\s*"([^"]+)"', re.I),
}

def extract_facts(prompt:str)->dict:
    return {k: rx.search(prompt).group(1) for k,rx in fact_re.items()}

# ---------- load requests ------------------------------------------------
req_facts = {}
with open("batch_files/batch_input.jsonl", encoding="utf-8") as fh:
    for line in fh:
        j   = json.loads(line)
        cid = j["custom_id"]
        prompt = j["body"]["messages"][0]["content"]
        facts  = extract_facts(prompt)
        req_facts[cid] = {
            "row_id": cid,
            "gender": facts["gender"],
            "age": int(facts["age"]),
            "smoker": facts["smoker"],
            "drinker": facts["drinker"],
            "blood_pressure": facts["bp"],
            "chief_complaint": facts["cc"],
        }

# ---------- load responses ----------------------------------------------
soap_map = {}
with open("soap_batch/soap_output.jsonl", encoding="utf-8") as fh:
    for line in fh:
        j = json.loads(line)
        if j["response"]["status_code"] != 200:
            continue
        cid   = j["custom_id"]
        body  = j["response"]["body"]
        soap  = json.loads(body["choices"][0]["message"]["content"])
        soap_map[cid] = soap

# ---------- merge + write -----------------------------------------------
out_path = pathlib.Path("data")
out_path.mkdir(exist_ok=True)
outfile = out_path / "synthetic_patients_with_soap.jsonl"

with outfile.open("w", encoding="utf-8") as fh:
    for cid, base in req_facts.items():
        if cid not in soap_map:
            continue
        record = {**base, **soap_map[cid]}
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✓ wrote {outfile}  ({sum(1 for _ in open(outfile))} lines)")
