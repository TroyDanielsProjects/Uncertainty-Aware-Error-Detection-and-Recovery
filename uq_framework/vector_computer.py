
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

# Ensure correct imports
try:
    from agents.base_agent import Trace
except ImportError:
    Trace = dict # Mock if import fails

from uq_framework.uq_entropic import compute_entropy_metrics, parse_stages
from uq_framework.uq_semantic import compute_semantic_divergence, compute_semantic_entropy
from uq_framework.uq_heuristic import analyze_text_confidence
from uq_framework.uq_mechanistic_interface import get_mechanistic_score


@dataclass
class UncertaintyVector:
    """The Holistic Uncertainty Vector (HUV)"""
    # 1. Entropic
    avg_entropy: float
    min_logit_gap: float
    saup_score: float
    # 2. Semantic
    semantic_divergence: float
    semantic_entropy: float
    # 3. Heuristic
    heuristic_score: float
    # 4. Mechanistic
    mechanistic_score: float
    
    external_score: float = 0.0 # Placeholder for future use
    textual_summary: str = ""


class UncertaintyVectorComputer:
    def __init__(self, config=None):
        self.config = config or {}

    def compute_vector_from_traces(
        self, traces: List[Trace], external_feedback=None
    ) -> Optional[UncertaintyVector]:
        
        if not traces:
            return None

        t0 = traces[0] # Primary trace for single-sample metrics

        # 1. Entropic (token-level fluency/confusion)
        ent = compute_entropy_metrics(t0)

        # 2. Semantic (multi-sample consistency)
        if len(traces) > 1:
            sd = compute_semantic_divergence(traces)
            se = compute_semantic_entropy(traces)
        else:
            sd = se = 0.0

        # 3. Heuristic (linguistic hedging)
        h = analyze_text_confidence(t0.get("text", ""))

        # 4. Mechanistic (internal state/entropy neurons)
        acts = t0.get("activations")
        
        # Ensure acts is a numpy array if it exists (handles potential type mismatches)
        if acts is not None and not isinstance(acts, np.ndarray):
             try:
                 acts = np.array(acts)
             except:
                 acts = None # Handle conversion failure

        m = get_mechanistic_score(acts) if acts is not None else 0.0


        return UncertaintyVector(
            avg_entropy = ent.get("avg_entropy", 0.0),
            min_logit_gap = ent.get("min_logit_gap", 0.0),
            saup_score = ent.get("saup_score", 0.0),
            semantic_divergence = sd,
            semantic_entropy = se,
            heuristic_score = h,
            mechanistic_score = m,
            textual_summary = "Computed"
        )

    def compute_stage_vectors(self, t: Trace) -> List[Dict]:
        # Analyze uncertainty per reasoning step (CoT) using entropic metrics
        return parse_stages(t)
