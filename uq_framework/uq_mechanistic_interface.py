
import numpy as np
import json
import os

_ENTROPY_INDICES = None
_CURRENT_CONFIG_PATH = None

def load_indices(custom_path=None):
    global _ENTROPY_INDICES, _CURRENT_CONFIG_PATH
    
    # Calculate default path relative to the project structure
    # This assumes a default config might exist, but typically calibration is model-specific.
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/entropy_neurons_default.json'))
    
    # Determine the path to the config file
    p = custom_path or os.getenv("UQ_NEURON_CONFIG", default_path)
    
    # Optimization: Avoid reloading if the path hasn't changed
    if _ENTROPY_INDICES is not None and _CURRENT_CONFIG_PATH == p:
        return

    if p and os.path.exists(p):
        try:
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                _ENTROPY_INDICES = np.array(data)
                _CURRENT_CONFIG_PATH = p
                print(f"Loaded {len(_ENTROPY_INDICES)} entropy indices from {p}")
            else:
                raise ValueError("Config file must contain a JSON list of integers.")
        except Exception as e:
            print(f"Warning: Failed to load config {p}: {e}. Mechanistic score will be 0.")
            _ENTROPY_INDICES = np.array([])
            _CURRENT_CONFIG_PATH = None
    else:
        # print(f"Warning: Config file {p} not found. Mechanistic score will be 0.")
        _ENTROPY_INDICES = np.array([])
        _CURRENT_CONFIG_PATH = None

def get_mechanistic_score(a: np.ndarray) -> float:
    """
    Computes the mechanistic uncertainty score based on activations of entropy neurons.
    Input 'a' should be the activations from the MLP layer (Shape: [Time, MLP_Dim]).
    """
    global _ENTROPY_INDICES
    if _ENTROPY_INDICES is None:
        load_indices()

    # Input validation
    if not isinstance(a, np.ndarray) or a.size == 0 or len(_ENTROPY_INDICES) == 0:
        return 0.0

    try:
        # Ensure activations array is at least 2D (Tokens, MLP_Dim)
        if a.ndim == 1:
            a = a.reshape(1, -1)

        # Select the activations corresponding to the entropy neuron indices
        # Ensure indices are within the bounds of the activation dimensions (MLP dimension)
        max_dim = a.shape[-1]
        valid_indices = _ENTROPY_INDICES[_ENTROPY_INDICES < max_dim]
        
        if len(valid_indices) == 0:
            # print(f"Warning: No valid entropy indices found for activation dimension size {max_dim}.")
            return 0.0

        r = a[..., valid_indices]
        
        # Calculate the mean activation across all tokens and selected neurons
        s = float(np.mean(r))
        
        # Apply a squashing function (tanh) to normalize the score. 
        # The scaling factor (0.5) is adjustable based on empirical results.
        # This follows the methodology definition provided.
        return float(np.tanh(s * 0.5))
    except Exception as e:
        # print(f"Error during mechanistic score calculation: {e}")
        return 0.0
