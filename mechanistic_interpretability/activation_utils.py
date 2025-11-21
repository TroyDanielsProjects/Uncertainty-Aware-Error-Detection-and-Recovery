import torch
import numpy as np
from typing import List

class ActivationMonitor:
    def __init__(self, model, layer_idx=-1):
        self.model = model
        self.layer_idx = layer_idx
        self.activations: List[torch.Tensor] = []
        self.hook_handle = None

    def _hook_fn(self, module, inp, out):
        x = inp[0]
        if x.shape[1] > 0:
            self.activations.append(x[:, -1:, :].detach().clone())

    def _get_layer(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer_idx].mlp.down_proj
        return None

    def __enter__(self):
        layer = self._get_layer()
        if layer:
            self.hook_handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, a, b, c):
        if self.hook_handle:
            self.hook_handle.remove()

    def get_batch_activations(self) -> np.ndarray:
        if not self.activations:
            return np.array([])
        try:
            t = torch.cat(self.activations, dim=1)
            return t.float().cpu().numpy()
        except:
            return np.array([])

def compute_state_entropy(a: np.ndarray) -> float:
    if a.size == 0:
        return 0.0
    return float(np.var(a, axis=0).mean())
