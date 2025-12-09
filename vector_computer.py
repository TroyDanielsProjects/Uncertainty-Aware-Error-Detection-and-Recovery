"""
Utility functions for computing single-trace uncertainty metrics.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np

# --- Configuration Loading ---
_ENTROPY_INDICES: Optional[np.ndarray] = None

def load_indices(custom_path: Optional[str] = None) -> None:
    global _ENTROPY_INDICES
    default_path = os.path.join(os.path.dirname(__file__), "config/entropy_neurons_default.json")
    p = custom_path or os.getenv("UQ_NEURON_CONFIG", default_path)
    if p and os.path.exists(p):
        try:
            with open(p, "r") as f:
                data = json.load(f)
            _ENTROPY_INDICES = np.array(data, dtype=int)
        except Exception:
            _ENTROPY_INDICES = np.array([], dtype=int)
    else:
        _ENTROPY_INDICES = np.array([], dtype=int)

# --- 1. Entropic Metrics ---
def compute_logit_gap_vector(t: Any) -> np.ndarray:
    t1 = np.array(t.get("top1_logprobs", []), dtype=float)
    t2 = np.array(t.get("top2_logprobs", []), dtype=float)
    if t1.size and t2.size and t1.size == t2.size:
        t2_safe = np.where(np.isinf(t2), -1e6, t2)
        return t1 - t2_safe
    return np.array([], dtype=float)

def compute_entropy_metrics(t: Any) -> Dict[str, float]:
    e = t.get("entropies", [])
    g = compute_logit_gap_vector(t)
    if not e:
        return {"avg_entropy": 0.0, "min_logit_gap": 0.0}
    return {
        "avg_entropy": float(np.mean(e)),
        "min_logit_gap": float(np.min(g)) if g.size else 0.0,
    }

# --- 2. Mechanistic Metrics ---
def get_token_mechanistic_scores(a: Optional[np.ndarray]) -> np.ndarray:
    global _ENTROPY_INDICES
    if _ENTROPY_INDICES is None: load_indices()
    
    if a is None or not isinstance(a, np.ndarray) or a.size == 0 or _ENTROPY_INDICES.size == 0:
        return np.array([])
    
    try:
        if a.ndim == 1: a = a.reshape(1, -1)
        max_dim = a.shape[-1]
        valid_indices = _ENTROPY_INDICES[_ENTROPY_INDICES < max_dim]
        if valid_indices.size == 0: return np.zeros(a.shape[0])
        
        s = np.mean(a[:, valid_indices], axis=1)
        return np.tanh(s * 0.5)
    except Exception:
        return np.array([])

# --- 3. Heuristic Metrics ---
def analyze_text_confidence(text: str) -> float:
    if not text: return 0.0
    hedges = ["might", "perhaps", "possibly", "unclear", "maybe", "assume", "unlikely"]
    words = text.lower().split()
    if not words: return 0.0
    count = sum(1 for w in words if w in hedges)
    return min(1.0, count / (len(words) / 5 + 1))

def extract_final_answer(text: str) -> str:
    if not isinstance(text, str): return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    m = re.search(r"\\boxed\\{([^}]+)\\}", text)
    if m: return m.group(1).strip()
    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
    if nums: return nums[-1].replace(",", "").strip()
    return ""

# --- 4. Data Structures ---
@dataclass
class UncertaintyVector:
    avg_entropy: float
    min_logit_gap: float
    heuristic_score: float
    mechanistic_score: float

class UncertaintyVectorComputer:
    def compute_vector(self, trace: Any) -> UncertaintyVector:
        ent = compute_entropy_metrics(trace)
        h = analyze_text_confidence(trace.get("text", ""))
        
        acts = trace.get("activations")
        if acts is not None and not isinstance(acts, np.ndarray):
            try: acts = np.array(acts)
            except: acts = None
            
        mech_scores = get_token_mechanistic_scores(acts)
        m = float(np.mean(mech_scores)) if mech_scores.size > 0 else 0.0

        return UncertaintyVector(
            avg_entropy=ent.get("avg_entropy", 0.0),
            min_logit_gap=ent.get("min_logit_gap", 0.0),
            heuristic_score=h,
            mechanistic_score=m,
        )

    def extract_token_metrics(self, t: Any) -> List[Dict[str, Any]]:
        tokens = t.get("tokens", [])
        entropies = t.get("entropies", [])
        top1 = t.get("top1_logprobs", [])
        top2 = t.get("top2_logprobs", [])
        gap_arr = compute_logit_gap_vector(t)
        
        acts = t.get("activations")
        if acts is not None and not isinstance(acts, np.ndarray):
            try: acts = np.array(acts)
            except: acts = None
        mech_arr = get_token_mechanistic_scores(acts)

        out = []
        length = min(len(tokens), len(entropies))
        for i in range(length):
            out.append({
                "position": i,
                "token": tokens[i],
                "entropy": float(entropies[i]),
                "top1_logprob": float(top1[i]) if i < len(top1) else 0.0,
                "top2_logprob": float(top2[i]) if i < len(top2) else -9999.0,
                "logit_gap": float(gap_arr[i]) if i < len(gap_arr) else 0.0,
                "mechanistic_score": float(mech_arr[i]) if (mech_arr.size > i) else 0.0
            })
        return out