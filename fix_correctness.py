import sqlite3
import json
import re
from llama_cpp import Llama

# --------------------
# Config
# --------------------
DB_PATH = "db/results.sqlite"
MODEL_PATH = "models/mistral-7b-instruct.Q4_K_M.gguf"
BATCH_SIZE = 4
CTX = 4096

# --------------------
# Load model (Metal / MPS)
# --------------------
print(f"Loading model from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX,
    n_threads=8,
    n_gpu_layers=-1, # Metal/MPS offloading
    verbose=False,
)

# --------------------
# Robust JSON Parsing (Fixes Invalid \escape & Extra data)
# --------------------
def parse_json_safe(text: str):
    """
    Robustly parses JSON, handling:
    1. 'Extra data': via Regex extraction.
    2. 'Invalid \escape': via automatic backslash sanitization fallback.
    """
    # 1. Regex to find the JSON object (ignores text before/after)
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return {}
    
    json_str = match.group(1)

    # 2. Try standard parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # 3. Fallback: Fix LaTeX backslashes (e.g., "\boxed" -> "\\boxed")
        try:
            # Escape all backslashes to make them valid JSON text
            fixed_str = json_str.replace("\\", "\\\\")
            return json.loads(fixed_str)
        except:
            # If it still fails, return empty to prevent script crash
            return {}

def run_llm_completion(prompt):
    """Runs the LLM with low temperature for deterministic output."""
    out = llm(
        prompt,
        max_tokens=64,  # Keep it short
        temperature=0,
        stop=["</s>", "```", "\n\n"]
    )
    return parse_json_safe(out["choices"][0]["text"])

# --------------------
# Two-Step Grader Logic (Extract -> Compare)
# --------------------
def step1_extract_value(full_trace):
    """Step 1: Blind extraction of the student's answer."""
    # Slice the last 2500 chars to ensure the final answer is in context
    trace_snippet = full_trace[-2500:] 
    
    prompt = f"""[INST] 
Extract the final answer from the text below.
- Prioritize values in \\boxed{{...}}
- Return strictly the number/string.

Text:
{trace_snippet} 

Respond ONLY in JSON:
{{ "val": "extracted_value" }}
[/INST]""" 
    
    res = run_llm_completion(prompt)
    return res.get("val", None)

def step2_verify_match(gold_val, student_val):
    """Step 2: Strict comparison without context noise."""
    prompt = f"""[INST] 
Are these two values mathematically equivalent?
- Ignore formatting (e.g. 1000 = 1,000)
- Ignore units (e.g. $5 = 5)

A (Gold): "{gold_val}"
B (Student): "{student_val}"

Respond ONLY in JSON:
{{ "match": true/false }}
[/INST]"""
    
    res = run_llm_completion(prompt)
    return res.get("match", False)

def local_grade(question, gold_answer, full_trace):
    # 1. Extract
    extracted = step1_extract_value(full_trace)
    
    if extracted is None:
        return {
            "correct": False,
            "reason": "Extraction Failed (No JSON returned)"
        }

    # 2. Compare
    is_match = step2_verify_match(str(gold_answer), str(extracted))

    return {
        "correct": is_match,
        "reason": f"Extracted: {extracted}"
    }

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
    SELECT result_id, question_text, full_trace_text, gold_answer
    FROM Results
""")
rows = cur.fetchall()

print(f"Regrading {len(rows)} rows using Two-Step Local LLM")

# --------------------
# Regrade Loop
# --------------------
for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i + BATCH_SIZE]

    for result_id, question, full_trace, gold_answer in batch:
        try:
            # Run the two-step grader
            verdict = local_grade(question, gold_answer, full_trace) 

            cur.execute("""
                UPDATE Results
                SET
                    is_correct = ?,
                    eval_method = 'Local_MPS_CoT',
                    gpt_eval_reason = ?
                WHERE result_id = ?
            """, (
                int(verdict["correct"]),
                verdict["reason"],
                result_id
            ))

        except Exception as e:
            print(f"[WARN] result_id {result_id} failed: {e}")
            cur.execute("""
                UPDATE Results
                SET
                    eval_method = 'Local_MPS_ERROR',
                    gpt_eval_reason = ?
                WHERE result_id = ?
            """, (str(e), result_id))

    conn.commit()
    print(f"Regraded {min(i + BATCH_SIZE, len(rows))}/{len(rows)}")

conn.close()
print("✅ Regrading complete using Two-Step Logic")