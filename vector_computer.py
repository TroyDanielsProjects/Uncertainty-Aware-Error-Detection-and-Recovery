"""
Utility functions and classes for computing holistic uncertainty vectors.

This module consolidates the pieces of the uncertainty quantification
framework required by the main pipeline. It provides functions to
compute scalar summaries and granular token-level metrics.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np
from collections import Counter

# ---------------------------------------------------------------------------
# Helper functions for entropic metrics
# ---------------------------------------------------------------------------

def compute_logit_gap_vector(t: Any) -> np.ndarray:
    """
    Compute the per‑token logit gap (difference between the top‑1 and
    top‑2 log probabilities) from a trace.
    """
    t1 = np.array(t.get("top1_logprobs", []), dtype=float)
    t2 = np.array(t.get("top2_logprobs", []), dtype=float)
    
    # Handle missing or mismatched lengths gracefully
    if t1.size and t2.size and t1.size == t2.size:
        # Replace ‑inf in t2 with a large negative value so that the gap
        # remains finite.
        t2_safe = np.where(np.isinf(t2), -1e6, t2)
        return t1 - t2_safe
    return np.array([], dtype=float)


def compute_entropy_metrics(t: Any) -> Dict[str, float]:
    """
    Aggregate entropic metrics from a trace.
    """
    e = t.get("entropies", [])
    g = compute_logit_gap_vector(t)
    if not e:
        return {
            "avg_entropy": 0.0,
            "min_logit_gap": 0.0,
        }
    e_arr = np.array(e, dtype=float)
    metrics = {
        "avg_entropy": float(np.mean(e_arr)),
        "min_logit_gap": float(np.min(g)) if g.size else 0.0,
    }
    return metrics


# ---------------------------------------------------------------------------
# Stage‑wise analysis utilities
# ---------------------------------------------------------------------------

def _identify_stage_boundaries(text: str) -> List[tuple[int, int, str]]:
    """
    Identify boundaries in a chain‑of‑thought style answer.
    """
    pattern = re.compile(
        r"(?:^|\n)(?:Step\s+\d+[:.]|\d+\.|First,|Second,|Third,|Finally,)", re.I
    )
    matches = list(pattern.finditer(text))
    if not matches:
        # Fallback: treat each paragraph as a stage if multiple paragraphs
        if text.count("\n") > 2:
            ps = list(re.finditer(r"(?:^|\n)(.+?)(?=\n|$)", text, re.DOTALL))
            return [(m.start(), m.end(), f"Paragraph {i + 1}") for i, m in enumerate(ps)]
        return [(0, len(text), "Full Trace")]
    out: List[tuple[int, int, str]] = []
    for i, m in enumerate(matches):
        s = m.start()
        # skip the leading newline if present
        if text[s] == "\n":
            s += 1
        e = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((s, e, m.group().strip().strip("\n")))
    return out


def _map_tokens_to_stages(
    t: Any, spans: List[tuple[int, int, str]]
) -> List[Dict[str, Any]]:
    """
    Aggregate entropic metrics for each identified reasoning stage.
    """
    toks = t.get("tokens", [])
    if not toks:
        return []
    # reconstruct character spans from tokens
    char_spans: List[tuple[int, int]] = []
    idx = 0
    for tok in toks:
        ct = (
            tok.replace("Ġ", " ")
            .replace(" ", " ")
            .replace("Ċ", "\n")
            .replace("<0x0A>", "\n")
        )
        s0 = idx
        e0 = idx + len(ct)
        char_spans.append((s0, e0))
        idx = e0
    e = t.get("entropies", [])
    g = compute_logit_gap_vector(t)
    out: List[Dict[str, Any]] = []
    for s0, s1, name in spans:
        idxs = [i for i, (a, b) in enumerate(char_spans) if b > s0 and a < s1]
        if not idxs:
            continue
        se = [e[i] for i in idxs if i < len(e)]
        sg = [g[i] for i in idxs if i < len(g)]
        if not se:
            continue
        out.append(
            {
                "stage_name": name,
                "token_count": len(idxs),
                "avg_entropy": float(np.mean(se)),
                "min_logit_gap": float(np.min(sg)) if sg else 0.0,
            }
        )
    return out


def parse_stages(t: Any) -> List[Dict[str, Any]]:
    """
    Public entry point for stage‑wise analysis.
    """
    text = t.get("text", "")
    if not text:
        return []
    spans = _identify_stage_boundaries(text)
    return _map_tokens_to_stages(t, spans)


# ---------------------------------------------------------------------------
# Heuristic confidence scoring
# ---------------------------------------------------------------------------

def analyze_text_confidence(text: str) -> float:
    """
    Detect lexical hedging in the supplied text.
    """
    if not text:
        return 0.0
    hedges = [
        "might", "perhaps", "possibly", "unclear", "maybe", "assume",
        "unlikely", "probably", "guess", "unsure", "estimate",
        "approximate", "seems", "appears", "could", "suggests",
    ]
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w in hedges)
    return min(1.0, count / (len(words) / 5 + 1))


# ---------------------------------------------------------------------------
# Semantic uncertainty metrics
# ---------------------------------------------------------------------------

def extract_final_answer(text: str) -> str:
    """
    Attempt to extract the final numeric answer.
    """
    if not isinstance(text, str):
        return ""
    if "####" in text:
        return text.split("####")[-1].strip()
    m = re.search(r"\\boxed\\{([^}]+)\\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(
        r"(?:Final Answer|The answer|Result)\s*(?:is|:|)\s*([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.I,
    )
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return ""


def compute_semantic_entropy(traces: List[Any]) -> float:
    """
    Compute the Shannon entropy of the distribution of answers.
    """
    answers = [extract_final_answer(t.get("text", "")) for t in traces]
    answers = [a for a in answers if a]
    if not answers:
        return 0.0
    C = Counter(answers)
    p = np.array([v / len(answers) for v in C.values()])
    plogp = np.where(p > 0, p * np.log(p), 0)
    return -float(np.sum(plogp))


def llm_judge_correctness(
    question: str, gold: str, pred: str, agent: Optional[Any] = None
) -> bool:
    """
    Judge whether a predicted answer matches the gold answer.
    """
    if gold.strip() == pred.strip():
        return True
    try:
        if abs(float(gold.replace(",", "")) - float(pred.replace(",", ""))) < 1e-5:
            return True
    except (ValueError, TypeError):
        pass
    # If an agent is provided, attempt to use it as a judge
    if agent:
        try:
            prompt = (
                "You are a strict math grader.\n"
                f"Question: {question}\n"
                f"Gold Answer: {gold}\n"
                f"Student Answer: {pred}\n"
                "Did the student get the correct answer? Reply 'YES' or 'NO'."
            )
            resp = agent.solve(prompt, n_samples=1)
            if resp and isinstance(resp, list):
                ans = resp[0].get("text", "")
                return "YES" in ans.upper()
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Mechanistic uncertainty
# ---------------------------------------------------------------------------

_ENTROPY_INDICES: Optional[np.ndarray] = None
_CURRENT_CONFIG_PATH: Optional[str] = None


def load_indices(custom_path: Optional[str] = None) -> None:
    """
    Load the indices of entropy neurons from a JSON file.
    """
    global _ENTROPY_INDICES, _CURRENT_CONFIG_PATH
    default_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config/entropy_neurons_default.json")
    )
    p = custom_path or os.getenv("UQ_NEURON_CONFIG", default_path)
    if _ENTROPY_INDICES is not None and _CURRENT_CONFIG_PATH == p:
        return
    if p and os.path.exists(p):
        try:
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                _ENTROPY_INDICES = np.array(data, dtype=int)
                _CURRENT_CONFIG_PATH = p
            else:
                _ENTROPY_INDICES = np.array([], dtype=int)
        except Exception:
            _ENTROPY_INDICES = np.array([], dtype=int)
    else:
        _ENTROPY_INDICES = np.array([], dtype=int)


def get_token_mechanistic_scores(a: Optional[np.ndarray]) -> np.ndarray:
    """
    Compute mechanistic uncertainty score *per token*.
    Returns a 1D array of shape (T,) where T is the number of tokens.
    """
    global _ENTROPY_INDICES
    if _ENTROPY_INDICES is None:
        load_indices()
    
    # Validation
    if a is None or not isinstance(a, np.ndarray) or a.size == 0 or _ENTROPY_INDICES.size == 0:
        return np.array([])
    
    try:
        # Ensure 'a' is (T, H)
        if a.ndim == 1:
            a = a.reshape(1, -1)
            
        max_dim = a.shape[-1]
        valid_indices = _ENTROPY_INDICES[_ENTROPY_INDICES < max_dim]
        
        if valid_indices.size == 0:
            return np.zeros(a.shape[0])
            
        # Select only entropy neurons -> Shape (T, k)
        r = a[:, valid_indices]
        
        # Average across the neurons (axis 1) to get a score per token -> Shape (T,)
        s = np.mean(r, axis=1)
        
        # Squashing function (tanh * 0.5)
        return np.tanh(s * 0.5)
        
    except Exception:
        return np.array([])


def get_mechanistic_score_summary(a: Optional[np.ndarray]) -> float:
    """
    Compute scalar mechanistic uncertainty score (average of token scores).
    """
    token_scores = get_token_mechanistic_scores(a)
    if token_scores.size == 0:
        return 0.0
    return float(np.mean(token_scores))


# ---------------------------------------------------------------------------
# Primary data classes
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyVector:
    """
    Container for the components of a holistic uncertainty vector.
    Cleaned of duplicates and unused fields.
    """
    avg_entropy: float
    min_logit_gap: float
    semantic_entropy: float
    heuristic_score: float
    mechanistic_score: float


class UncertaintyVectorComputer:
    """
    Compute holistic uncertainty vectors from generation traces.
    """

    def compute_vector_from_traces(
        self, traces: List[Any], external_feedback: Optional[Any] = None
    ) -> Optional[UncertaintyVector]:
        if not traces:
            return None
        t0 = traces[0]
        # entropic metrics
        ent = compute_entropy_metrics(t0)
        
        # semantic metrics (multi‑sample)
        se = 0.0
        if len(traces) > 1:
            se = compute_semantic_entropy(traces)
            
        # heuristic (hedging)
        h = analyze_text_confidence(t0.get("text", ""))
        
        # mechanistic
        acts = t0.get("activations")
        if acts is not None and not isinstance(acts, np.ndarray):
            try:
                acts = np.array(acts)
            except Exception:
                acts = None
        m = get_mechanistic_score_summary(acts)

        return UncertaintyVector(
            avg_entropy=ent.get("avg_entropy", 0.0),
            min_logit_gap=ent.get("min_logit_gap", 0.0),
            semantic_entropy=se,
            heuristic_score=h,
            mechanistic_score=m,
        )

    def compute_stage_vectors(self, t: Any) -> List[Dict[str, Any]]:
        """
        Compute per‑stage uncertainty summaries.
        """
        return parse_stages(t)

    def extract_token_metrics(self, t: Any) -> List[Dict[str, Any]]:
        """
        Extract granular token-by-token metrics for SQL logging.
        Includes: Entropy, Logit Gap, Mechanistic Score.
        """
        tokens = t.get("tokens", [])
        entropies = t.get("entropies", [])
        top1 = t.get("top1_logprobs", [])
        top2 = t.get("top2_logprobs", [])
        
        # Calculate derived vectors
        gap_arr = compute_logit_gap_vector(t)
        
        acts = t.get("activations")
        if acts is not None and not isinstance(acts, np.ndarray):
            try:
                acts = np.array(acts)
            except Exception:
                acts = None
        mech_arr = get_token_mechanistic_scores(acts)

        out = []
        # Use length of tokens/entropies as baseline
        length = min(len(tokens), len(entropies))
        
        for i in range(length):
            clean_tok = (
                tokens[i]
                .replace("Ġ", " ")
                .replace("Ċ", "\\n")
                .replace("<0x0A>", "\\n")
            )
            
            # Safe access handling
            t1 = float(top1[i]) if i < len(top1) else 0.0
            t2 = float(top2[i]) if i < len(top2) else -9999.0
            gap = float(gap_arr[i]) if i < len(gap_arr) else 0.0
            mech = float(mech_arr[i]) if (mech_arr.size > i) else 0.0

            out.append({
                "position": i,
                "token": clean_tok,
                "entropy": float(entropies[i]),
                "top1_logprob": t1,
                "top2_logprob": t2,
                "logit_gap": gap,
                "mechanistic_score": mech
            })
        return out