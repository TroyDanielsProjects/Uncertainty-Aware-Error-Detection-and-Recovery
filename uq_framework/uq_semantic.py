
import re
import numpy as np
from collections import Counter
from typing import List

# Optional OpenAI fallback for grading
try:
    from openai import OpenAI
    # Assumes OPENAI_API_KEY is set in the environment.
    # Set CLIENT = OpenAI() to enable.
    CLIENT = None
except ImportError:
    CLIENT = None


# -------------------------
#  Final-answer extraction
# -------------------------
def extract_final_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # GSM8K format
    if "####" in text:
        return text.split("####")[-1].strip()

    # LaTeX format: \boxed{18}
    m = re.search(r"\\boxed\{([^}]+)\\}", text)
    if m:
        return m.group(1).strip()

    # Natural language format: “The answer is X”
    m = re.search(
        r"(?:Final Answer|The answer|Result)\s*(?:is|:|)\s*([0-9,]+(?:\.[0-9]+)?)",
        text, re.I
    )
    if m:
        return m.group(1).replace(",", "").strip()

    # Last numeric fallback
    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()

    return ""


# -------------------------
#  LLM correctness grader
# -------------------------
def llm_judge_correctness(question: str, gold: str, pred: str, agent=None) -> bool:
    
    # 1. Basic exact match / numeric equivalence
    if gold.strip() == pred.strip():
        return True
    try:
        # Allow for small floating point differences
        if abs(float(gold.replace(',','')) - float(pred.replace(',',''))) < 1e-5:
            return True
    except (ValueError, TypeError):
        pass

    # If basic check fails, proceed to LLM grading if available
    
    if not agent and not CLIENT:
        # Cannot perform LLM judge, fallback to the basic check result (False)
        return False

    # print(f"Using LLM Judge for Q: {question[:30]}... Gold: {gold}, Pred: {pred}")

    prompt = f"""
    You are a strict math grader.

    Question: {question}
    Gold Answer: {gold}
    Student Answer: {pred}

    Did the student get the correct answer? Be lenient with formatting, but strict with the value.
    Reply "YES" or "NO" only.
    """

    # Local model path (if an agent is provided, e.g., self-judge)
    if agent:
        try:
            # Assuming agent has a 'solve' method returning a list of Trace-like objects
            resp = agent.solve(prompt, n_samples=1)
            if resp and resp[0].get("text"):
                return "YES" in resp[0].get("text").upper()
        except Exception:
            pass

    # OpenAI fallback
    if CLIENT:
        try:
            r = CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            out = r.choices[0].message.content.upper()
            return "YES" in out
        except Exception as e:
            print(f"OpenAI Grader failed: {e}")
            pass

    return False


# -------------------------
#  Semantic uncertainty
# -------------------------
# Note: Input type expects List[Trace] (objects with a get method)
def compute_semantic_entropy(traces: List[object]) -> float:
    answers = [extract_final_answer(t.get("text")) for t in traces]
    answers = [a for a in answers if a] # Filter empty answers
    if not answers:
        return 0.0

    # Calculate entropy of the distribution of answers
    C = Counter(answers)
    p = np.array([v / len(answers) for v in C.values()])
    
    # Stable calculation of -sum(p * log(p))
    plogp = np.where(p > 0, p * np.log(p), 0)
    return -np.sum(plogp)


def compute_semantic_divergence(traces: List[object]) -> float:
    answers = [extract_final_answer(t.get("text")) for t in traces]
    answers = [a for a in answers if a]
    if not answers:
        return 0.0

    # Ratio of unique answers to total answers (Consistency metric)
    return len(set(answers)) / len(answers)
