"""
Utility functions and classes for computing holistic uncertainty vectors.

This module consolidates the pieces of the uncertainty quantification
framework required by the main pipeline.  It defines a dataclass to
represent the Holistic Uncertainty Vector (HUV) and provides a
computer to construct HUVs from generation traces.  The traces are
expected to be dictionaries or objects with a ``get`` method returning
fields such as ``text``, ``entropies``, ``top1_logprobs``,
``top2_logprobs`` and optionally ``activations``.

Only the functions used by the calibration and experiment runner are
included here.  Expensive optional dependencies such as
``sentence‑transformers`` and external API calls have been stripped
out to keep the pipeline self contained.
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

def compute_saup_score(e: List[float]) -> float:
    """
    Compute the square‑average uncertainty (SAUP) score for a list of
    entropies.  This is the root mean square of the entropies which
    emphasises variability over the sequence.
    """
    e_arr = np.array(e, dtype=float)
    if e_arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(e_arr ** 2)))


def compute_logit_gap_vector(t: Any) -> np.ndarray:
    """
    Compute the per‑token logit gap (difference between the top‑1 and
    top‑2 log probabilities) from a trace.  A larger gap indicates the
    model is more confident in its chosen token.
    """
    t1 = np.array(t.get("top1_logprobs", []), dtype=float)
    t2 = np.array(t.get("top2_logprobs", []), dtype=float)
    # handle missing or mismatched lengths gracefully
    if t1.size and t2.size and t1.size == t2.size:
        # Replace ‑inf in t2 with a large negative value so that the gap
        # remains finite.
        t2_safe = np.where(np.isinf(t2), -1e6, t2)
        return t1 - t2_safe
    return np.array([], dtype=float)


def compute_entropy_metrics(t: Any) -> Dict[str, float]:
    """
    Aggregate entropic metrics from a trace.  Returns a dictionary
    containing:

      * saup_score       – square‑average uncertainty across the sequence
      * avg_entropy      – mean token entropy
      * min_logit_gap    – minimum logit gap across the sequence
      * avg_logit_gap    – mean logit gap across the sequence

    If the trace contains no entropy information, zeros are returned for all
    metrics.
    """
    e = t.get("entropies", [])
    g = compute_logit_gap_vector(t)
    if not e:
        return {
            "saup_score": 0.0,
            "avg_entropy": 0.0,
            "min_logit_gap": 0.0,
            "avg_logit_gap": 0.0,
        }
    e_arr = np.array(e, dtype=float)
    metrics = {
        "saup_score": compute_saup_score(e_arr.tolist()),
        "avg_entropy": float(np.mean(e_arr)),
        "min_logit_gap": float(np.min(g)) if g.size else 0.0,
        "avg_logit_gap": float(np.mean(g)) if g.size else 0.0,
    }
    return metrics


# ---------------------------------------------------------------------------
# Stage‑wise analysis utilities
# ---------------------------------------------------------------------------

def _identify_stage_boundaries(text: str) -> List[tuple[int, int, str]]:
    """
    Identify boundaries in a chain‑of‑thought style answer.  The
    function looks for common step markers (e.g. “Step 1:”) and falls
    back to splitting on newlines if none are found.
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
    Aligns token boundaries to character spans heuristically and then
    computes average entropy, SAUP and minimum logit gap for each stage.
    """
    toks = t.get("tokens", [])
    if not toks:
        return []
    # reconstruct character spans from tokens; handle common BPE artefacts
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
        # filter indices for which entropy and logit gap values exist
        se = [e[i] for i in idxs if i < len(e)]
        sg = [g[i] for i in idxs if i < len(g)]
        if not se:
            continue
        out.append(
            {
                "stage_name": name,
                "token_count": len(idxs),
                "avg_entropy": float(np.mean(se)),
                "saup_score": compute_saup_score(se),
                "min_logit_gap": float(np.min(sg)) if sg else 0.0,
            }
        )
    return out


def parse_stages(t: Any) -> List[Dict[str, Any]]:
    """
    Public entry point for stage‑wise analysis.  Given a trace with a
    ``text`` field and token‑level metrics it returns a list of stage
    summaries.
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
    Detect lexical hedging in the supplied text.  The output is a
    number between 0 and 1 representing how uncertain the wording
    appears.  A larger score indicates more uncertainty.

    This implementation uses a simple keyword based heuristic and does
    not depend on large embedding models.
    """
    if not text:
        return 0.0
    hedges = [
        "might",
        "perhaps",
        "possibly",
        "unclear",
        "maybe",
        "assume",
        "unlikely",
        "probably",
        "guess",
        "unsure",
        "estimate",
        "approximate",
        "seems",
        "appears",
        "could",
        "suggests",
    ]
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(1 for w in words if w in hedges)
    # normalise by length with a scaling factor; cap at 1.0
    return min(1.0, count / (len(words) / 5 + 1))


# ---------------------------------------------------------------------------
# Semantic uncertainty metrics
# ---------------------------------------------------------------------------

def extract_final_answer(text: str) -> str:
    """
    Attempt to extract the final numeric answer from a long chain of
    thought.  Handles GSM8K style `####` markers, LaTeX \boxed
    answers and simple “The answer is ...” phrases.  Falls back to the
    last number found in the text.
    """
    if not isinstance(text, str):
        return ""
    # GSM8K style: answer is after a `####` delimiter
    if "####" in text:
        return text.split("####")[-1].strip()
    # LaTeX boxed answers: \boxed{42}
    m = re.search(r"\\boxed\\{([^}]+)\\}", text)
    if m:
        return m.group(1).strip()
    # Natural language phrases like “The answer is 42.”
    m = re.search(
        r"(?:Final Answer|The answer|Result)\s*(?:is|:|)\s*([0-9,]+(?:\.[0-9]+)?)",
        text,
        re.I,
    )
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: return the last number found
    nums = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return ""


def compute_semantic_entropy(traces: List[Any]) -> float:
    """
    Compute the Shannon entropy of the distribution of answers across
    multiple samples.  A value of zero means all samples agree; higher
    values indicate disagreement.
    """
    answers = [extract_final_answer(t.get("text", "")) for t in traces]
    answers = [a for a in answers if a]
    if not answers:
        return 0.0
    C = Counter(answers)
    p = np.array([v / len(answers) for v in C.values()])
    plogp = np.where(p > 0, p * np.log(p), 0)
    return -float(np.sum(plogp))


def compute_semantic_divergence(traces: List[Any]) -> float:
    """
    Compute the ratio of unique answers to total answers.  This ranges
    from zero (all answers identical) to one (all answers different).
    """
    answers = [extract_final_answer(t.get("text", "")) for t in traces]
    answers = [a for a in answers if a]
    if not answers:
        return 0.0
    return len(set(answers)) / float(len(answers))


def llm_judge_correctness(
    question: str, gold: str, pred: str, agent: Optional[Any] = None
) -> bool:
    """
    Judge whether a predicted answer matches the gold answer.  Exact
    string matches and numeric equivalence are supported.  If an
    optional ``agent`` is provided with a ``solve`` method, this
    function can delegate the decision to a model.  External API calls
    have been removed to keep the pipeline self contained.
    """
    # Basic numeric or string equivalence
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
    Load the indices of entropy neurons from a JSON file.  The file
    should contain a JSON array of integers.  A path can be supplied
    explicitly via ``custom_path``.  The environment variable
    ``UQ_NEURON_CONFIG`` will be consulted if no path is provided.

    Loaded indices are cached globally to avoid reloading the same
    configuration repeatedly.
    """
    global _ENTROPY_INDICES, _CURRENT_CONFIG_PATH
    # Determine a default config path relative to this file
    default_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "config/entropy_neurons_default.json")
    )
    p = custom_path or os.getenv("UQ_NEURON_CONFIG", default_path)
    # If the path hasn't changed, reuse the cached indices
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
                raise ValueError("Config file must contain a JSON list of integers.")
        except Exception as e:
            print(f"Warning: failed to load neuron indices from {p}: {e}")
            _ENTROPY_INDICES = np.array([], dtype=int)
            _CURRENT_CONFIG_PATH = None
    else:
        # no config found; fallback to empty indices
        _ENTROPY_INDICES = np.array([], dtype=int)
        _CURRENT_CONFIG_PATH = None


def get_mechanistic_score(a: Optional[np.ndarray]) -> float:
    """
    Compute a scalar mechanistic uncertainty score from a matrix of
    activations.  The input ``a`` should be an array of shape
    (T, H) where T is the number of generated tokens and H is the
    hidden dimension.  The score is the mean activation of the
    configured entropy neurons passed through a tanh squashing
    function.
    """
    global _ENTROPY_INDICES
    if _ENTROPY_INDICES is None:
        load_indices()
    # validate inputs
    if not isinstance(a, np.ndarray) or a.size == 0 or _ENTROPY_INDICES.size == 0:
        return 0.0
    try:
        # ensure 2D
        if a.ndim == 1:
            a = a.reshape(1, -1)
        max_dim = a.shape[-1]
        valid_indices = _ENTROPY_INDICES[_ENTROPY_INDICES < max_dim]
        if valid_indices.size == 0:
            return 0.0
        r = a[:, valid_indices]
        s = float(np.mean(r))
        return float(np.tanh(s * 0.5))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Primary data classes
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyVector:
    """
    Container for the components of a holistic uncertainty vector.
    """

    avg_entropy: float
    min_logit_gap: float
    saup_score: float
    semantic_divergence: float
    semantic_entropy: float
    heuristic_score: float
    mechanistic_score: float
    external_score: float = 0.0
    textual_summary: str = ""


class UncertaintyVectorComputer:
    """
    Compute holistic uncertainty vectors from a list of generation
    traces.  The first trace is considered the primary response; if
    multiple traces are provided semantic uncertainty metrics will be
    computed across them.
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
        if len(traces) > 1:
            sd = compute_semantic_divergence(traces)
            se = compute_semantic_entropy(traces)
        else:
            sd = 0.0
            se = 0.0
        # heuristic (hedging)
        h = analyze_text_confidence(t0.get("text", ""))
        # mechanistic
        acts = t0.get("activations")
        # convert lists to numpy if necessary
        if acts is not None and not isinstance(acts, np.ndarray):
            try:
                acts = np.array(acts)
            except Exception:
                acts = None
        m = get_mechanistic_score(acts) if acts is not None else 0.0
        return UncertaintyVector(
            avg_entropy=ent.get("avg_entropy", 0.0),
            min_logit_gap=ent.get("min_logit_gap", 0.0),
            saup_score=ent.get("saup_score", 0.0),
            semantic_divergence=sd,
            semantic_entropy=se,
            heuristic_score=h,
            mechanistic_score=m,
            textual_summary="Computed",
        )

    def compute_stage_vectors(self, t: Any) -> List[Dict[str, Any]]:
        """
        Compute per‑stage uncertainty summaries for a single trace.
        """
        return parse_stages(t)