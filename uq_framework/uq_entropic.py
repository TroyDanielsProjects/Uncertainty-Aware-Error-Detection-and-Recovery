
import re
import numpy as np
from typing import Dict, List, Tuple
# Ensure correct import path
try:
    from agents.base_agent import Trace
except ImportError:
    Trace = dict # Mock if import fails

def compute_saup_score(e: List[float]) -> float:
    # Root Mean Square of entropies (captures variability)
    e = np.array(e)
    if len(e) == 0:
        return 0.0
    return float(np.sqrt(np.mean(e**2)))

def compute_logit_gap_vector(t: Trace) -> np.ndarray:
    # Difference between top1 and top2 log probabilities
    t1 = np.array(t.get("top1_logprobs", []))
    t2 = np.array(t.get("top2_logprobs", []))
    if len(t1) and len(t2) and len(t1) == len(t2):
        # Since logprobs are negative, t1 >= t2. The gap is t1 - t2 (positive value).
        # Handle potential -inf in t2 if only one token was possible.
        t2_safe = np.where(np.isinf(t2), -1e6, t2)
        return t1 - t2_safe
    return np.array([])

def compute_entropy_metrics(t: Trace) -> Dict[str, float]:
    e = t.get("entropies", [])
    g = compute_logit_gap_vector(t)
    
    if not e:
        return {"saup_score": 0.0, "avg_entropy": 0.0, "min_logit_gap": 0.0, "avg_logit_gap": 0.0}

    return {
        "saup_score": compute_saup_score(e),
        "avg_entropy": float(np.mean(e)),
        # Min logit gap indicates the point of highest local uncertainty (smallest gap)
        "min_logit_gap": float(np.min(g)) if len(g) else 0.0,
        "avg_logit_gap": float(np.mean(g)) if len(g) else 0.0,
    }

# --- Stage-Wise Analysis ---

def _identify_stage_boundaries(text: str) -> List[Tuple[int, int, str]]:
    # Regex to identify common step markers in CoT
    pat = re.compile(r"(?:^|\n)(?:Step\s+\d+[:.]|\d+\.|First,|Second,|Third,|Finally,)", re.I)
    matches = list(pat.finditer(text))
    
    if not matches:
        # Fallback: split by paragraphs if enough newlines exist
        if text.count("\n") > 2:
            ps = list(re.finditer(r"(?:^|\n)(.+?)(?=\n|$)", text, re.DOTALL))
            return [(m.start(), m.end(), f"Paragraph {i+1}") for i, m in enumerate(ps)]
        return [(0, len(text), "Full Trace")]

    out = []
    for i, m in enumerate(matches):
        s = m.start()
        # Clean up the start index if it starts with a newline
        if text[s] == '\n': s += 1
        
        e = matches[i+1].start() if i+1 < len(matches) else len(text)
        out.append((s, e, m.group().strip().strip('\n')))
    return out

def _map_tokens_to_stages(t: Trace, spans: List[Tuple[int, int, str]]) -> List[Dict]:
    toks = t.get("tokens", [])
    if not toks:
        return []

    # Reconstruct character spans from tokens (handles common tokenizer artifacts)
    ts = []
    idx = 0
    for tok in toks:
        # Handle BPE/SentencePiece artifacts (e.g., GPT/Llama tokenizers)
        ct = tok.replace("Ġ", " ").replace(" ", " ").replace("Ċ", "\n").replace("<0x0A>", "\n")
        s, e = idx, idx + len(ct)
        ts.append((s, e))
        idx = e

    e = t.get("entropies", [])
    g = compute_logit_gap_vector(t)

    out = []
    for s0, s1, name in spans:
        # Find tokens overlapping the stage span
        idxs = [i for i, (a, b) in enumerate(ts) if b > s0 and a < s1]
        if not idxs:
            continue

        # Ensure indices are valid for entropy/gap arrays (generation might stop abruptly)
        valid_e_idxs = [i for i in idxs if i < len(e)]
        valid_g_idxs = [i for i in idxs if i < len(g)]

        se = [e[i] for i in valid_e_idxs]
        sg = [g[i] for i in valid_g_idxs] if len(g) else []

        if not se: continue

        out.append({
            "stage_name": name,
            "token_count": len(idxs),
            "avg_entropy": float(np.mean(se)),
            "saup_score": compute_saup_score(se),
            "min_logit_gap": float(np.min(sg)) if sg else 0.0,
        })

    return out

def parse_stages(t: Trace) -> List[Dict]:
    # Main function for stage-wise analysis
    text = t.get("text", "")
    if not text:
        return []
    spans = _identify_stage_boundaries(text)
    return _map_tokens_to_stages(t, spans)
