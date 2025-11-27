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

def calibrate(model_name, save_path):
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
            torch_dtype=torch.float16,
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

    vocab_size = WU.shape[0]
    d_mlp = Wout.shape[1] 

    # Optimization: Sample vocabulary to save memory
    # The notebook uses full vocab, but 128k vocab (Llama3) is too large for consumer RAM without batching
    V_subset_size = min(vocab_size, 8192) 
    print(f"Vocab size: {vocab_size}. MLP Dim: {d_mlp}. Sampling {V_subset_size} tokens for calibration.")
    
    idx = torch.randperm(vocab_size)[:V_subset_size]
    WU_sub = WU[idx] # [V_sub, d_model]

    # 1. Pre-calculate ||WU|| (Row norms) for the subset
    # Shape: [V_sub, 1]
    WU_norm = torch.norm(WU_sub, dim=1, keepdim=True)

    batch_size = 256
    variances = []

    print("Calculating normalized logit variances per neuron...")
    
    for i in tqdm(range(0, d_mlp, batch_size)):
        Wout_batch = Wout[:, i:i+batch_size] # [d_model, batch_size]
        
        # 2. Calculate ||Wout|| (Column norms) for this batch
        # Shape: [1, batch_size]
        Wout_norm_batch = torch.norm(Wout_batch, dim=0, keepdim=True)
        
        # 3. Compute Raw Logits: ψ = WU @ Wout
        # Shape: [V_sub, batch_size]
        L_batch = WU_sub @ Wout_batch
        
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

    # Identify Entropy Neurons: characterized by LOW variance
    K_percent = 0.05 
    k = int(d_mlp * K_percent)
    
    topk_indices = torch.topk(all_variances, k=k, largest=False).indices

    full_save_path = os.path.join(project_root, save_path)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    with open(full_save_path, "w") as f:
        json.dump(topk_indices.tolist(), f)

    print(f"Calibration complete. Saved {len(topk_indices)} indices to {full_save_path}")

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