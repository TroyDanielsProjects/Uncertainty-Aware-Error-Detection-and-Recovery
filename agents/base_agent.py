from abc import ABC, abstractmethod
from typing import List, Any, Optional

class Trace(dict):
    """Container for model outputs: text, tokens, entropies, logprobs, activations."""
    pass

class BaseAgent(ABC):
    def __init__(self, model_name):
        self.model = model_name

    @abstractmethod
    def solve(self, task_description: str, n_samples: int = 1) -> List[Trace]:
        pass
