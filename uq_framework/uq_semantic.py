import re
import numpy as np
from collections import Counter
from typing import List

try:
    from openai import OpenAI
    CLIENT = OpenAI()
except:
    CLIENT = None

def extract_final_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""

    if "####" in text:
        return text.split("####")[-1].strip()

    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"(Final Answer|The answer|Result) is:?\s*([0-9,]+(?:\.\d+)?)",
                  text, re.I)
    if m:
        return m.group(2).replace(",", "").strip()

    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()

    return ""

def llm_judge_correctness(q: str, gold: str, pred: str) -> bool:
    if not CLIENT:
        return False

    prompt = (
        f"Question: {q}\nGold Answer: {gold}\nModel Output: {pred}\n\n"
        "Is the model output correct? Reply YES or NO."
    )
    try:
        r = CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return "YES" in r.choices[0].message.content.strip().upper()
    except:
        return False

def compute_semantic_entropy(traces: List[dict]) -> float:
    a = [extract_final_answer(t.get("text")) for t in traces]
    a = [x for x in a if x]
    if not a:
        return 0.0
    c = Counter(a)
    p = np.array([v / len(a) for v in c.values()])
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def compute_semantic_divergence(traces: List[dict]) -> float:
    a = [extract_final_answer(t.get("text")) for t in traces]
    a = [x for x in a if x]
    if not a:
        return 0.0
    return len(set(a)) / len(a)
