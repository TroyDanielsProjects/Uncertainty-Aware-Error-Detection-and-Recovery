import torch
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import sys

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    print("Error: 'transformers' library not found. Calibration requires it.")
    AutoModelForCausalLM = None

def calibrate(model_name, save_path, identify_by_cosine_sim=False, identify_by_variance=True, identify_by_max_std_dev=True, k=5):
    """
    Identifies Entropy Neurons based on Normalized Logit Attribution Variance (Cosine Similarity).
    
    consistent with 'entropy_neurons.ipynb':
    1. L = WU @ Wout (Logit attribution)
    2. NormFactor = ||WU||_dim1 * ||Wout||_dim0
    3. L_norm = L / NormFactor
    4. Select neurons with LOW variance in L_norm.
    """
    if AutoModelForCausalLM is None:
        return

    print(f"Calibrating entropy neurons for {model_name} (Normalized/Cosine Approach)")

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu", 
            low_cpu_mem_usage=True,
            trust_remote_code=True 
        )
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return

    # Identify Wout and WU
    try:
        layer_idx = len(model.model.layers) - 1
        down_proj = model.model.layers[layer_idx].mlp.down_proj
        lm_head = model.lm_head

        Wout = down_proj.weight.float().detach() # [d_model, d_mlp]
        WU = lm_head.weight.float().detach()     # [V, d_model]

    except AttributeError:
        print(f"Error: Could not identify necessary layers in {model_name}.")
        return

    # list to keep track of all entropy neurons
    neuron_candidates = []

    # identify the entropy neurons through different tecniques
    if identify_by_cosine_sim:
        neuron_candidates.append(cosine_sim_identification(WU, Wout, k=k))
    elif identify_by_variance or identify_by_max_std_dev:
        L = WU @ Wout
        if identify_by_variance:
            vars = torch.var(L, dim=0)
            neuron_candidates.append(torch.topk(vars, k=k, largest=False).indices)
        elif identify_by_max_std_dev:
            neuron_candidates.append(max_std_dev_identification(L))

    # 2. Concatenate all collected tensors
    if len(neuron_candidates) > 0:
        entropy_neurons = torch.cat(neuron_candidates)
    else:
        entropy_neurons = torch.tensor([], dtype=torch.long)

    full_save_path = os.path.join(project_root, save_path)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    with open(full_save_path, "w") as f:
        json.dump(entropy_neurons.tolist(), f)

    print(f"Calibration complete. Saved {len(entropy_neurons)} indices to {full_save_path}")

def cosine_sim_identification(WU, Wout, k=5):
    # 1. Pre-calculate ||WU|| (Row norms) for the subset
    # Shape: [V_sub, 1]
    WU_norm = torch.norm(WU, dim=1, keepdim=True)

    batch_size = 256
    variances = []

    print("Calculating normalized logit variances per neuron...")
    
    for i in tqdm(range(0, d_mlp, batch_size)):
        d_mlp = Wout.shape[1] 
        Wout_batch = Wout[:, i:i+batch_size] # [d_model, batch_size]
        
        # 2. Calculate ||Wout|| (Column norms) for this batch
        # Shape: [1, batch_size]
        Wout_norm_batch = torch.norm(Wout_batch, dim=0, keepdim=True)
        
        # 3. Compute Raw Logits: ψ = WU @ Wout
        # Shape: [V_sub, batch_size]
        L_batch = WU @ Wout_batch
        
        # 4. Compute Normalization Factor (Outer Product)
        # Shape: [V_sub, batch_size] = [V_sub, 1] * [1, batch_size]
        Norm_Factor = WU_norm @ Wout_norm_batch
        
        # 5. Compute Normalized Attribution (Cosine Similarity)
        # Add epsilon to avoid division by zero
        L_norm = L_batch / (Norm_Factor + 1e-9)
        
        # 6. Compute variance across vocabulary (dim=0)
        vars_batch = torch.var(L_norm, dim=0)
        variances.append(vars_batch)

    all_variances = torch.cat(variances)
    
    topk_indices = torch.topk(all_variances, k=k, largest=False).indices
    return topk_indices

def max_std_dev_identification(L, k=5):
    # calculate the mean over the vocabulary dimension
    expectation = torch.mean(L, dim=0)
    # Want to take the max distance from the mean over the vocab dimension for each neuron
    l_inf_sttdev = torch.max(torch.abs((L - expectation)), dim=0).values
    topk_indices = torch.topk(l_inf_sttdev, k=k, largest=False).indices
    return topk_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Entropy Neurons (Normalized).")
    parser.add_argument("--model", type=str, help="HuggingFace model ID")
    parser.add_argument("--out", type=str, help="Output path")
    
    if len(sys.argv) > 1:
        args = parser.parse_args()
        if args.model and args.out:
            calibrate(args.model, args.out)
        else:
             print("Error: Arguments missing.")
    else:
        print("Usage: python scripts/calibrate_entropy_neurons.py --model <ID> --out <PATH>")