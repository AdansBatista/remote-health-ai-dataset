# generate_complaints.py  – robust version
import json, re, pandas as pd
from openai import OpenAI
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential

openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

PROMPT = """
List the 200 most common chief complaints seen in family-practice / primary-care
clinics in the USA and Canada.  Return ONLY a JSON array of short phrases,
each ≤ 6 words.  Do NOT add code fences, numbering, or any extra text.
"""

def clean_json(raw: str) -> str:
    """Remove ``` fences or text before/after a JSON array."""
    raw = raw.strip()

    # strip ```…``` blocks
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)   # opening fence
        raw = re.sub(r"\s*```$", "", raw)            # closing fence

    # keep only the first [...] you find
    m = re.search(r"\[.*\]", raw, re.S)
    return m.group(0) if m else ""

@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=15))
def fetch_complaints():
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[{"role": "system", "content": PROMPT}]
    )
    cleaned = clean_json(resp.choices[0].message.content)
    return json.loads(cleaned)          # will raise JSONDecodeError if still bad

complaints = fetch_complaints()

# 1-based index for human-friendly Excel view
indexed = [{"idx": i + 1, "chief_complaint": c} for i, c in enumerate(complaints)]

with open("chief_complaints.json", "w") as f:
    json.dump(complaints, f, indent=2)

pd.DataFrame(indexed).to_excel("chief_complaints.xlsx", index=False)
print("Saved chief_complaints.[json|xlsx] with", len(complaints), "items")
