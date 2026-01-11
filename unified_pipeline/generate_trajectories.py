# unified_pipeline/generate_trajectories.py
"""
generate_trajectories.py

Append-only: adds/extends per-record `trajectories` in a JSONL results file.
- Always includes the ORIGINAL trace as a trajectory (from existing `trace_txt`).
- Optionally appends N regenerated trajectories using ModelSetup.solve (token-level data included if available).
- Never deletes keys or overwrites existing values. Only adds new keys/subkeys.

Usage:
python unified_pipeline/generate_trajectories.py \
  --input_json  path/to/results.jsonl \
  --output_json path/to/results_with_traj.jsonl \
  --config_json path/to/trial_config.json \
  --subset_strategy random_10pct \
  --k 3
"""

import argparse
import json
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import torch

from unified_pipeline.model_setup.model_setup import ModelSetup


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
    """
    Accept:
      - random_10pct  -> 0.10
      - random_2.5pct -> 0.025
      - random_0.1    -> 0.1
    """
    if not strategy.startswith("random_"):
        return None
    tail = strategy[len("random_") :]
    if tail.endswith("pct"):
        pct = float(tail[:-3])
        return pct / 100.0
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
        # sample at least 1 if frac > 0 and df non-empty (nice for tiny dev sets)
        n = max(1, int(round(frac * len(df))))
        return df.sample(n=n, random_state=seed)
    # fallback
    return df.iloc[[]]


# -----------------------------
# Prompt recovery (safe)
# -----------------------------
def recover_prompt_from_trace(trace_txt: str) -> str:
    """
    Try to recover the prompt used during original generation.
    For GSM8K-like traces, the prompt is typically:
      'Question: ...\nLet's think step by step.'
    """
    if not trace_txt:
        return ""
    marker = "Let's think step by step."
    if marker in trace_txt:
        return trace_txt.split(marker)[0].strip() + "\n" + marker
    return trace_txt.strip()


# -----------------------------
# Answer extraction (v2, improved but cheap)
# -----------------------------
_ANS_PATTERNS = [
    re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE),     # GSM8K canonical
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
    # fallback: last number
    nums = _NUM_FALLBACK.findall(text)
    if nums:
        return nums[-1].strip()
    # final fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# -----------------------------
# Trace packing
# -----------------------------
def _get_attr(obj: Any, name: str, default=None):
    return getattr(obj, name, default)


def pack_trace(
    *,
    text: str,
    source: str,
    tokens: Optional[List[str]] = None,
    entropies=None,
    top1_logprobs=None,
    top2_logprobs=None,
) -> Dict[str, Any]:
    return {
        "traj_id": str(uuid.uuid4()),
        "source": source,  # "original" or "regenerated"
        "text": text,
        # token-level data (may be None for the original if not stored)
        "tokens": tokens,
        "entropies": entropies,
        "top1_logprobs": top1_logprobs,
        "top2_logprobs": top2_logprobs,
        # do not overwrite any existing final_answer; add a v2 extraction
        "final_answer_v2": extract_final_answer_v2(text),
    }


# -----------------------------
# Main augmentation
# -----------------------------
def ensure_original_trajectory(rec: Dict[str, Any]) -> None:
    """
    Adds original trajectory from existing trace_txt.
    Append-only: if already present, do nothing.
    """
    trajs = rec.setdefault("trajectories", [])
    if any(t.get("source") == "original" for t in trajs):
        return
    trace_txt = rec.get("trace_txt") or ""
    if not trace_txt:
        return
    trajs.append(
        pack_trace(
            text=trace_txt,
            source="original",
            tokens=rec.get("tokens"),  # if you ever stored these upstream
            entropies=rec.get("entropies"),
            top1_logprobs=rec.get("top1_logprobs"),
            top2_logprobs=rec.get("top2_logprobs"),
        )
    )


def count_regenerated(trajs: List[Dict[str, Any]]) -> int:
    return sum(1 for t in trajs if t.get("source") == "regenerated")


def append_regenerated_trajectories(
    rec: Dict[str, Any],
    model: ModelSetup,
    k: int,
    *,
    explanation_override: str,
    prompt_from: str,
    force: bool,
) -> None:
    trajs = rec.setdefault("trajectories", [])
    existing = count_regenerated(trajs)
    if (not force) and existing >= k:
        return

    # generate as many as needed to reach k total regenerated
    need = k if force else (k - existing)
    if need <= 0:
        return

    # prompt source
    if prompt_from == "prompt":
        prompt = rec.get("prompt") or ""
    else:
        prompt = recover_prompt_from_trace(rec.get("trace_txt") or "")

    if not prompt:
        return

    for _ in range(need):
        try:
            trace = model.solve(prompt, explanation=explanation_override)
        except TypeError:
            # in case solve signature differs; fallback to solve(prompt)
            trace = model.solve(prompt)

        if not trace:
            continue

        text = _get_attr(trace, "text", "") or ""
        if not text:
            continue

        trajs.append(
            pack_trace(
                text=text,
                source="regenerated",
                tokens=_get_attr(trace, "tokens", None),
                entropies=_get_attr(trace, "entropies", None),
                top1_logprobs=_get_attr(trace, "top1_logprobs", None),
                top2_logprobs=_get_attr(trace, "top2_logprobs", None),
            )
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--config_json", required=True)
    ap.add_argument("--subset_strategy", default="incorrect_only")
    ap.add_argument("--k", type=int, default=3, help="number of regenerated trajectories to store (target count)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_regen", action="store_true", help="append k more regenerated trajs even if already present")
    ap.add_argument("--prompt_from", choices=["trace", "prompt"], default="trace")
    ap.add_argument("--explanation_override", default="", help="pass to ModelSetup.solve(explanation=...) to avoid None issues")
    args = ap.parse_args()

    records = load_jsonl(args.input_json)
    df = pd.DataFrame(records)
    subset_df = select_subset(df, args.subset_strategy, seed=args.seed)
    subset_idx = set(subset_df.index.tolist())

    with open(args.config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = ModelSetup(
        model_name=cfg["model_name"],
        task=cfg["task"],
        dtype=torch.float16 if str(cfg.get("dtype", "float16")) == "float16" else torch.float32,
        uq_methods=[],
        mech_interp_ident_methods=[],
        entropy_neurons=0,
        data_size=0,
        semantic_runs=0,
        include_prefill=cfg.get("prefill_included", True),
        device=cfg.get("device", "mps"),
        explanation=cfg.get("use_explanation", False),
    )

    out = []
    for i, rec in enumerate(records):
        rec = dict(rec)

        # Always keep original trajectory if trace exists (even if not selected)
        ensure_original_trajectory(rec)

        if i in subset_idx:
            append_regenerated_trajectories(
                rec,
                model,
                args.k,
                explanation_override=args.explanation_override,
                prompt_from=args.prompt_from,
                force=args.force_regen,
            )

        out.append(rec)

    write_jsonl(args.output_json, out)


if __name__ == "__main__":
    main()
