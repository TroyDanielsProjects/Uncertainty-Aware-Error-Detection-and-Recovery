# unified_pipeline/compute_semantic_and_consistency.py
"""
compute_semantic_and_consistency.py

Append-only: computes and appends:
  - semantic metrics (ANSWER-level variability)
  - consistency metrics (TRACE-level coherence via embeddings cosine similarity)

Reads existing `trajectories` (must include original + regenerated).
Does NOT regenerate anything.
Never deletes keys or overwrites existing values; appends run objects to:
  - rec["semantic_runs"]
  - rec["consistency_runs"]

Usage:
python unified_pipeline/compute_semantic_and_consistency.py \
  --input_json  path/to/results_with_traj.jsonl \
  --output_json path/to/results_with_sem_cons.jsonl \
  --subset_strategy all
"""

import argparse
import json
import math
import re
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# Optional: sklearn fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos_sim
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# -----------------------------
# IO
# -----------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Subset selection
# -----------------------------
def _parse_random_strategy(strategy: str) -> Optional[float]:
    if not strategy.startswith("random_"):
        return None
    tail = strategy[len("random_") :]
    if tail.endswith("pct"):
        return float(tail[:-3]) / 100.0
    return float(tail)


def select_subset(df: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    if strategy == "all":
        return df
    if strategy == "none":
        return df.iloc[[]]
    if strategy == "incorrect_only":
        return df[df.get("is_exact", False) == False]
    frac = _parse_random_strategy(strategy)
    if frac is not None:
        frac = max(0.0, min(1.0, frac))
        if len(df) == 0 or frac == 0.0:
            return df.iloc[[]]
        n = max(1, int(round(frac * len(df))))
        return df.sample(n=n, random_state=seed)
    return df.iloc[[]]


# -----------------------------
# Answer extraction v2 (cheap)
# -----------------------------
_ANS_PATTERNS = [
    re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE),
    re.compile(r"final answer\s*[:\-]\s*([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]\s*([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE),
]
_NUM_FALLBACK = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_final_answer_v2(text: str) -> str:
    if not text:
        return ""
    for pat in _ANS_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    nums = _NUM_FALLBACK.findall(text)
    if nums:
        return nums[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# -----------------------------
# Semantic metrics (ANSWER-level)
# -----------------------------
def _shannon_entropy(counts, total):
    ent = 0.0
    for c in counts:
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def compute_semantic(answers: List[str]) -> Dict[str, Any]:
    canon = [a.strip().lower() for a in answers if a is not None]
    canon = [a for a in canon if a != ""]
    if not canon:
        return {
            "n": 0,
            "semantic_entropy": None,
            "num_clusters": 0,
            "majority_frac": None,
            "pairwise_agreement": None,
        }

    counter = Counter(canon)
    total = len(canon)
    entropy = _shannon_entropy(counter.values(), total)
    num_clusters = len(counter)
    majority_frac = max(counter.values()) / total

    if total <= 1:
        pairwise = 1.0
    else:
        agree = sum(c * (c - 1) / 2 for c in counter.values())
        total_pairs = total * (total - 1) / 2
        pairwise = agree / total_pairs

    return {
        "n": total,
        "semantic_entropy": float(entropy),
        "num_clusters": int(num_clusters),
        "majority_frac": float(majority_frac),
        "pairwise_agreement": float(pairwise),
    }


# -----------------------------
# Consistency metrics (TRACE-level)
# -----------------------------
def cosine_sim_matrix_texts(texts: List[str], device: str) -> np.ndarray:
    if _HAS_ST:
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        embs = model.encode(texts, convert_to_tensor=True)
        sim = st_util.cos_sim(embs, embs).cpu().numpy()
        return sim

    if _HAS_SK:
        vec = TfidfVectorizer(max_features=5000)
        X = vec.fit_transform(texts)
        sim = sk_cos_sim(X, X)
        return sim

    # ultra-fallback: identical-string similarity
    n = len(texts)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim[i, j] = 1.0 if texts[i] == texts[j] else 0.0
    return sim


def compute_consistency(traces: List[str], traj_ids: List[str], device: str) -> Dict[str, Any]:
    traces = [t if t is not None else "" for t in traces]
    if len(traces) == 0:
        return {"n": 0, "coherence_mean": None, "similarity_matrix": None, "per_trajectory_coherence": []}

    sim = cosine_sim_matrix_texts(traces, device=device)
    n = sim.shape[0]

    per = []
    for i in range(n):
        row = sim[i].tolist()
        coherence_i = (sum(row) - 1.0) / (n - 1) if n > 1 else 0.0
        per.append({"traj_id": traj_ids[i], "coherence": float(coherence_i)})

    coherence_mean = float(np.mean([p["coherence"] for p in per])) if per else None
    return {
        "n": int(n),
        "coherence_mean": coherence_mean,
        "similarity_matrix": [[float(round(x, 6)) for x in row] for row in sim.tolist()],
        "per_trajectory_coherence": per,
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--subset_strategy", default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="mps", help="sentence-transformers device if available; e.g. cpu|mps|cuda")
    ap.add_argument("--answer_field_priority", default="final_answer_v2,final_answer",
                    help="comma-separated priority list for trajectory answer field")
    args = ap.parse_args()

    records = load_jsonl(args.input_json)
    df = pd.DataFrame(records)
    subset_df = select_subset(df, args.subset_strategy, seed=args.seed)
    subset_idx = set(subset_df.index.tolist())

    answer_fields = [x.strip() for x in args.answer_field_priority.split(",") if x.strip()]

    run_id = f"{int(time.time())}"
    out = []

    for i, rec in enumerate(records):
        rec = dict(rec)

        if i not in subset_idx:
            out.append(rec)
            continue

        trajs = rec.get("trajectories") or []
        if not trajs:
            out.append(rec)
            continue

        traj_ids = [t.get("traj_id", f"idx_{j}") for j, t in enumerate(trajs)]
        texts = [t.get("text", "") for t in trajs]

        # derive answers (append-only: do not overwrite; add final_answer_v2 if missing)
        answers = []
        for t in trajs:
            if "final_answer_v2" not in t:
                t["final_answer_v2"] = extract_final_answer_v2(t.get("text", ""))
            chosen = None
            for fld in answer_fields:
                val = t.get(fld)
                if isinstance(val, str) and val.strip() != "":
                    chosen = val
                    break
            if chosen is None:
                chosen = extract_final_answer_v2(t.get("text", ""))
            answers.append(chosen)

        semantic_obj = compute_semantic(answers)
        semantic_obj.update({"run_id": run_id, "answer_field_used_priority": answer_fields})

        consistency_obj = compute_consistency(texts, traj_ids, device=args.device)
        consistency_obj.update({"run_id": run_id, "backend": "sentence-transformers" if _HAS_ST else ("tfidf" if _HAS_SK else "string")})

        rec.setdefault("semantic_runs", []).append(semantic_obj)
        rec.setdefault("consistency_runs", []).append(consistency_obj)

        out.append(rec)

    write_jsonl(args.output_json, out)


if __name__ == "__main__":
    main()
