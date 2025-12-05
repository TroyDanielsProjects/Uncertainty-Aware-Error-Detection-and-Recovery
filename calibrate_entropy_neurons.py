"""
Utility for calibrating entropy neurons.

This script identifies a subset of hidden units in a causal language
model that correlate strongly with the model's output distribution.
These "entropy neurons" can be used later to compute mechanistic
uncertainty scores.  The implementation is based on the original
calibration notebook but has been modified to fix minor issues (such
as undefined variables) and to avoid printing usage instructions when
imported as a module.
"""

from __future__ import annotations

import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse
import sys

# Resolve the project root relative to this file
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from transformers import AutoModelForCausalLM  # type: ignore
except ImportError:
    print("Error: 'transformers' library not found. Calibration requires it.")
    AutoModelForCausalLM = None


def calibrate(
    model_name: str,
    save_path: str,
    identify_by_cosine_sim: bool = False,
    identify_by_variance: bool = True,
    identify_by_max_std_dev: bool = True,
    k: int = 5,
) -> None:
    """
    Identify entropy neurons for the specified model and save their
    indices to disk.  The selection strategy can be controlled via
    flags.  By default neurons with the smallest variance in the
    normalised logit attribution matrix (cosine similarity metric) are
    selected.

    Parameters
    ----------
    model_name: str
        HuggingFace model identifier.
    save_path: str
        Relative path (from the project root) where the JSON file of
        neuron indices will be written.
    identify_by_cosine_sim: bool, optional
        Whether to select neurons using the cosine similarity metric.
    identify_by_variance: bool, optional
        Whether to select neurons based on low variance.  Ignored if
        ``identify_by_cosine_sim`` is True.
    identify_by_max_std_dev: bool, optional
        Whether to select neurons based on the maximum standard deviation.
    k: int, optional
        Number of neurons to select.
    """
    if AutoModelForCausalLM is None:
        return
    print(f"Calibrating entropy neurons for {model_name}…")
    # load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return
    # locate MLP down projection and LM head
    try:
        layer_idx = len(model.model.layers) - 1
        down_proj = model.model.layers[layer_idx].mlp.down_proj
        lm_head = model.lm_head
        Wout = down_proj.weight.float().detach()  # (d_model, d_mlp)
        WU = lm_head.weight.float().detach()  # (V, d_model)
    except AttributeError:
        print(f"Error: could not identify necessary layers in {model_name}.")
        return
    neuron_candidates: List[torch.Tensor] = []
    if identify_by_cosine_sim:
        neuron_candidates.append(cosine_sim_identification(WU, Wout, k=k))
    elif identify_by_variance or identify_by_max_std_dev:
        L = WU @ Wout  # (V, d_mlp)
        if identify_by_variance:
            vars = torch.var(L, dim=0)
            neuron_candidates.append(torch.topk(vars, k=k, largest=False).indices)
        elif identify_by_max_std_dev:
            neuron_candidates.append(max_std_dev_identification(L, k=k))
    if neuron_candidates:
        entropy_neurons = torch.cat(neuron_candidates)
    else:
        entropy_neurons = torch.tensor([], dtype=torch.long)
    full_save_path = os.path.join(project_root, save_path)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    with open(full_save_path, "w") as f:
        json.dump(entropy_neurons.tolist(), f)
    print(f"Calibration complete. Saved {len(entropy_neurons)} indices to {full_save_path}")


def cosine_sim_identification(WU: torch.Tensor, Wout: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Identify entropy neurons using the normalised logit variance
    (cosine similarity) metric.  Returns a tensor of shape (k,) with
    the indices of the selected neurons.
    """
    # precompute L2 norms
    WU_norm = torch.norm(WU, dim=1, keepdim=True)
    batch_size = 256
    variances: List[torch.Tensor] = []
    print("Calculating normalised logit variances per neuron…")
    d_mlp = Wout.shape[1]
    for i in tqdm(range(0, d_mlp, batch_size)):
        Wout_batch = Wout[:, i : i + batch_size]  # (d_model, batch_size)
        Wout_norm_batch = torch.norm(Wout_batch, dim=0, keepdim=True)
        L_batch = WU @ Wout_batch  # (V, batch_size)
        Norm_Factor = WU_norm @ Wout_norm_batch  # (V, batch_size)
        L_norm = L_batch / (Norm_Factor + 1e-9)
        vars_batch = torch.var(L_norm, dim=0)
        variances.append(vars_batch)
    all_variances = torch.cat(variances)
    topk_indices = torch.topk(all_variances, k=k, largest=False).indices
    return topk_indices


def max_std_dev_identification(L: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Identify entropy neurons based on the maximum standard deviation of
    logit attributions.  Computes the infinity norm of the deviation
    from the mean for each neuron and selects the smallest k.
    """
    expectation = torch.mean(L, dim=0)
    l_inf_stddev = torch.max(torch.abs((L - expectation)), dim=0).values
    topk_indices = torch.topk(l_inf_stddev, k=k, largest=False).indices
    return topk_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Entropy Neurons (Normalised).")
    parser.add_argument("--model", type=str, help="HuggingFace model ID")
    parser.add_argument("--out", type=str, help="Output path for the JSON file")
    args = parser.parse_args()
    if args.model and args.out:
        calibrate(args.model, args.out)
    else:
        print("Usage: python calibrate_entropy_neurons.py --model <ID> --out <PATH>")