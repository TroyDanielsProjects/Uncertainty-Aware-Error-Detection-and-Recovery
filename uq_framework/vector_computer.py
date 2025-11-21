from typing import List, Dict, Optional
from dataclasses import dataclass

from agents.base_agent import Trace
from uq_framework.uq_entropic import compute_entropy_metrics, parse_stages
from uq_framework.uq_semantic import compute_semantic_divergence, compute_semantic_entropy
from uq_framework.uq_heuristic import analyze_text_confidence
from uq_framework.uq_mechanistic_interface import get_mechanistic_score

@dataclass
class UncertaintyVector:
    avg_entropy: float
    min_logit_gap: float
    saup_score: float
    semantic_divergence: float
    semantic_entropy: float
    heuristic_score: float
    mechanistic_score: float
    external_score: float
    textual_summary: str

class UncertaintyVectorComputer:
    def __init__(self, config=None):
        self.config = config or {}

    def compute_vector_from_traces(self, traces: List[Trace], external_feedback=None) -> Optional[UncertaintyVector]:
        if not traces:
            return None

        t0 = traces[0]

        ent = compute_entropy_metrics(t0)

        if len(traces) > 1:
            sd = compute_semantic_divergence(traces)
            se = compute_semantic_entropy(traces)
        else:
            sd, se = 0.0, 0.0

        h = analyze_text_confidence(t0.get("text",""))

        m = 0.0
        acts = t0.get("activations")
        if acts is not None:
            m = get_mechanistic_score(acts)

        U = UncertaintyVector(
            avg_entropy=ent.get("avg_entropy",0.0),
            min_logit_gap=ent.get("min_logit_gap",0.0),
            saup_score=ent.get("saup_score",0.0),
            semantic_divergence=sd,
            semantic_entropy=se,
            heuristic_score=h,
            mechanistic_score=m,
            external_score=0.0,
            textual_summary="Computed"
        )
        return U

    def compute_stage_vectors(self, t: Trace) -> List[Dict]:
        return parse_stages(t)
