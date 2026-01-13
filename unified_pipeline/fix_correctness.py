"""
fix_correctness.py

Append-only correctness regrading using:
  - deterministic extraction (baseline)
  - optional LLM adjudication (local or API)

Grades:
  - original trajectory
  - all regenerated trajectories

Adds entries to:
  rec["correctness_runs"]

Never deletes or overwrites existing keys.
"""

import argparse
import json
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


# =============================
# IO
# =============================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =============================
# Subset selection
# =============================
def select_subset(df: pd.DataFrame, strategy: str, seed: int):
    if strategy == "all":
        return df
    if strategy == "none":
        return df.iloc[[]]
    if strategy == "incorrect_only":
        return df[df.get("is_exact", False) == False]
    if strategy.startswith("random_"):
        frac = float(strategy.split("_")[1].replace("pct", "")) / 100
        return df.sample(frac=frac, random_state=seed)
    return df.iloc[[]]


# =============================
# Deterministic extraction
# =============================
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_final_answer_v2(text: str) -> str:
    if not text:
        return ""
    nums = _NUM_RE.findall(text.replace(",", ""))
    if nums:
        return nums[-1]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def try_parse_number(s: str) -> Optional[float]:
    try:
        return float(re.sub(r"[^\d\.\-\+eE]", "", s))
    except Exception:
        return None


def deterministic_correct(pred: str, gold: str, tol: float) -> bool:
    pn, gn = try_parse_number(pred), try_parse_number(gold)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= tol
    return pred.strip().lower() == gold.strip().lower()


# =============================
# LLM GRADER (LOCAL OR API)
# =============================

class LLMGrader:
    def __init__(self, mode: str, **kwargs):
        self.mode = mode
        if mode == "local":
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=kwargs["model_path"],
                n_ctx=kwargs.get("ctx", 4096),
                n_threads=kwargs.get("threads", 8),
                n_gpu_layers=kwargs.get("gpu_layers", -1),
                verbose=False,
            )
        elif mode == "api":
            import openai
            self.client = openai.OpenAI(api_key=kwargs["api_key"])
            self.model = kwargs["model"]
        else:
            raise ValueError(f"Unknown grader mode: {mode}")

    # -------- FIX: robust JSON parsing --------
    def _parse_json_safe(self, text: str) -> Dict[str, Any]:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if not m:
            return {}

        raw = m.group(1)

        # Attempt 1: direct
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt 2: escape backslashes (LaTeX, boxed, etc.)
        try:
            return json.loads(raw.replace("\\", "\\\\"))
        except json.JSONDecodeError:
            pass

        # Attempt 3: truncate to last brace
        try:
            last = raw[: raw.rfind("}") + 1]
            return json.loads(last)
        except Exception:
            return {}

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if self.mode == "local":
            out = self.llm(prompt, max_tokens=128, temperature=0)
            text = out["choices"][0]["text"]
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = resp.choices[0].message.content

        return self._parse_json_safe(text)

    def grade(self, gold: str, trace_text: str) -> Dict[str, Any]:
        # Step 1: extract
        p1 = f"""
Extract the final answer from the text below.
Return ONLY JSON: {{ "val": "..." }}

Text:
{trace_text[-2500:]}
"""
        r1 = self._call_llm(p1)
        val = r1.get("val")

        if val is None:
            return {"is_correct": False, "reason": "LLM extraction failed"}

        # Step 2: verify
        p2 = f"""
Are these two values mathematically equivalent?

Gold: "{gold}"
Student: "{val}"

Return ONLY JSON: {{ "match": true/false }}
"""
        r2 = self._call_llm(p2)

        return {
            "is_correct": bool(r2.get("match", False)),
            "extracted": val,
        }


# =============================
# Main
# =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--subset_strategy", default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=1e-9)

    # grading mode
    ap.add_argument("--grader", choices=["none", "local", "api"], default="none")
    ap.add_argument("--local_model_path")
    ap.add_argument("--api_key")
    ap.add_argument("--api_model", default="gpt-4o-mini")

    args = ap.parse_args()

    records = load_jsonl(args.input_json)
    df = pd.DataFrame(records)
    subset = select_subset(df, args.subset_strategy, args.seed)
    subset_idx = set(subset.index.tolist())

    grader = None
    if args.grader != "none":
        grader = LLMGrader(
            mode=args.grader,
            model_path=args.local_model_path,
            api_key=args.api_key,
            model=args.api_model,
        )

    run_id = str(int(time.time()))
    out = []

    for i, rec in enumerate(records):
        rec = dict(rec)
        if i not in subset_idx:
            out.append(rec)
            continue

        gold = str(rec.get("gold") or rec.get("gold_answer") or "")
        trajs = rec.get("trajectories") or []

        per = []
        for j, t in enumerate(trajs):
            text = t.get("text", "")
            ans = extract_final_answer_v2(text)

            det_ok = deterministic_correct(ans, gold, args.tol)
            llm_res = grader.grade(gold, text) if grader else None

            per.append({
                "traj_id": t.get("traj_id", f"idx_{j}"),
                "source": t.get("source"),
                "deterministic_answer": ans,
                "deterministic_correct": det_ok,
                "llm": llm_res,
            })

        rec.setdefault("correctness_runs", []).append({
            "run_id": run_id,
            "method": "deterministic+llm" if grader else "deterministic",
            "n": len(per),
            "per_trajectory": per,
            "any_correct": any(
                (p["llm"]["is_correct"] if p["llm"] else p["deterministic_correct"])
                for p in per
            ),
        })

        out.append(rec)

    write_jsonl(args.output_json, out)


if __name__ == "__main__":
    main()
