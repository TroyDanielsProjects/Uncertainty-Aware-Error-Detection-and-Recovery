
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Trace:
    """
    A structure holding the output and internal metrics of a generation.
    """
    text: str
    tokens: List[str] = field(default_factory=list)
    # Entropic UQ data
    entropies: List[float] = field(default_factory=list)
    top1_logprobs: List[float] = field(default_factory=list)
    top2_logprobs: List[float] = field(default_factory=list)
    # Mechanistic UQ data (e.g., activations from the last MLP layer's activation function)
    # Shape: (Time, MLP_Dimension)
    activations: Optional[np.ndarray] = None
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key, default=None):
        # Provide dictionary-like access for compatibility
        return getattr(self, key, default)


class BaseAgent:
    """
    Abstract base class for language model agents.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def solve(self, task: str, n_samples: int = 1) -> List[Trace]:
        """
        Generates solutions for a given task.
        """
        raise NotImplementedError("The 'solve' method must be implemented by subclasses.")

    def __repr__(self):
        return f"<{self.__class__.__name__}(model='{self.model_name}')>"
