import numpy as np
import json
import os

_ENTROPY_INDICES = None

def load_indices():
    global _ENTROPY_INDICES
    p = "config/entropy_neurons_llama3.json"
    if os.path.exists(p):
        with open(p, "r") as f:
            _ENTROPY_INDICES = np.array(json.load(f))
    else:
        _ENTROPY_INDICES = np.array([])

def get_mechanistic_score(a: np.ndarray) -> float:
    global _ENTROPY_INDICES
    if _ENTROPY_INDICES is None:
        load_indices()

    if a.size == 0 or len(_ENTROPY_INDICES) == 0:
        return 0.0

    try:
        r = a[..., _ENTROPY_INDICES]
        s = float(np.mean(r))
        return float(np.tanh(s * 0.5))
    except:
        return 0.0
