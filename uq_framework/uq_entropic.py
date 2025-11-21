import re
import numpy as np
from typing import Dict, List, Tuple
from agents.base_agent import Trace

def compute_saup_score(e: List[float]) -> float:
    e = np.array(e)
    if len(e) == 0:
        return 0.0
    return float(np.sqrt(np.mean(e**2)))

def compute_logit_gap_vector(t: Trace) -> np.ndarray:
    t1 = np.array(t.get("top1_logprobs", []))
    t2 = np.array(t.get("top2_logprobs", []))
    if len(t1) and len(t2) and len(t1) == len(t2):
        return t1 - t2
    return np.array([])

def compute_entropy_metrics(t: Trace) -> Dict[str, float]:
    e = t.get("entropies", [])
    g = compute_logit_gap_vector(t)
    if not e:
        return {"saup_score": 0.0, "avg_entropy": 0.0, "min_logit_gap": 0.0}

    return {
        "saup_score": compute_saup_score(e),
        "avg_entropy": float(np.mean(e)),
        "min_logit_gap": float(np.min(g)) if len(g) else 0.0,
        "avg_logit_gap": float(np.mean(g)) if len(g) else 0.0,
    }

def _identify_stage_boundaries(text: str) -> List[Tuple[int, int, str]]:
    pat = re.compile(r"(?:^|\n)(?:Step\s+\d+[:.]|\d+\.|First,|Second,|Third,|Finally,)", re.I)
    matches = list(pat.finditer(text))
    if not matches:
        if text.count("\n") > 2:
            ps = list(re.finditer(r"(?:^|\n)(.+?)(?=\n|$)", text, re.DOTALL))
            return [(m.start(), m.end(), f"Paragraph {i+1}") for i, m in enumerate(ps)]
        return [(0, len(text), "Full Trace")]

    out = []
    for i, m in enumerate(matches):
        s = m.start()
        e = matches[i+1].start() if i+1 < len(matches) else len(text)
        out.append((s, e, m.group().strip()))
    return out

def _map_tokens_to_stages(t: Trace, spans: List[Tuple[int, int, str]]) -> List[Dict]:
    toks = t.get("tokens", [])
    if not toks:
        return []

    # reconstruct character spans
    ts = []
    idx = 0
    for tok in toks:
        ct = tok.replace("Ġ", " ").replace("Ċ", "\n")
        s, e = idx, idx + len(ct)
        ts.append((s, e))
        idx = e

    e = t.get("entropies", [])
    g = compute_logit_gap_vector(t)

    out = []
    for s0, s1, name in spans:
        idxs = [i for i, (a, b) in enumerate(ts) if b > s0 and a < s1]
        if not idxs:
            continue

        se = [e[i] for i in idxs]
        sg = [g[i] for i in idxs] if len(g) else []

        out.append({
            "stage_name": name,
            "token_count": len(idxs),
            "avg_entropy": float(np.mean(se)),
            "saup_score": compute_saup_score(se),
            "min_logit_gap": float(np.min(sg)) if sg else 0.0,
        })

    return out

def parse_stages(t: Trace) -> List[Dict]:
    text = t.get("text", "")
    if not text:
        return []
    spans = _identify_stage_boundaries(text)
    return _map_tokens_to_stages(t, spans)
