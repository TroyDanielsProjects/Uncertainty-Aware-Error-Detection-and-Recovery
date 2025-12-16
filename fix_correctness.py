import sqlite3
import json
import re
from llama_cpp import Llama

# --------------------
# Config
# --------------------
DB_PATH = "db/results.sqlite"
MODEL_PATH = "models/mistral-7b-instruct.Q4_K_M.gguf"
BATCH_SIZE = 16
CTX = 4096

# --------------------
# Load model (Metal / MPS)
# --------------------
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX,
    n_threads=8,
    n_gpu_layers=-1,   # 🔥 ALL layers on Metal
    verbose=False,
)

# --------------------
# JSON Parsing (robust)
# --------------------
def parse_json_safe(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found:\n{text}")
    return json.loads(match.group())


# --------------------
# Local LLM Grader
# --------------------
def local_grade(question, full_response):
    prompt = f"""[INST]
You are a strict math grader.

Question:
{question}

Student's FULL response:
{full_response}

Task:
Determine whether the student's final numerical answer is mathematically correct.

Rules:
- Ignore formatting, verbosity, markdown, repetition
- Ignore explanation quality
- Judge ONLY mathematical correctness
- If the correct final number appears anywhere, mark correct

Respond ONLY in JSON:
{{
  "correct": true or false,
  "reason": "brief justification"
}}
[/INST]
"""

    out = llm(
        prompt,
        max_tokens=256,
        temperature=0,
        stop=["</s>"],
    )

    return parse_json_safe(out["choices"][0]["text"])


# --------------------
# DB Setup
# --------------------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Add old_correct column if missing
cur.execute("PRAGMA table_info(Results)")
cols = [c[1] for c in cur.fetchall()]
if "old_correct" not in cols:
    print("Adding old_correct column")
    cur.execute("ALTER TABLE Results ADD COLUMN old_correct BOOLEAN")
    cur.execute("UPDATE Results SET old_correct = is_correct")
    conn.commit()

# --------------------
# Fetch rows
# --------------------
cur.execute("""
    SELECT result_id, question_text, full_trace_text
    FROM Results
""")
rows = cur.fetchall()

print(f"Regrading {len(rows)} rows using local MPS LLM")

# --------------------
# Regrade
# --------------------
for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i + BATCH_SIZE]

    for result_id, question, full_trace in batch:
        try:
            verdict = local_grade(question, full_trace)

            cur.execute("""
                UPDATE Results
                SET
                    is_correct = ?,
                    eval_method = 'Local_MPS_LLM',
                    gpt_eval_reason = ?
                WHERE result_id = ?
            """, (
                int(verdict["correct"]),
                verdict.get("reason", ""),
                result_id
            ))

        except Exception as e:
            print(f"[WARN] result_id {result_id} failed: {e}")
            cur.execute("""
                UPDATE Results
                SET
                    eval_method = 'Local_MPS_LLM_ERROR',
                    gpt_eval_reason = ?
                WHERE result_id = ?
            """, (str(e), result_id))

    conn.commit()
    print(f"Regraded {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

conn.close()
print("✅ Regrading complete using local MPS LLM")
