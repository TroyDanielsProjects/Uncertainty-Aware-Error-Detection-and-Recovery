import torch
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import sys
import matplotlib.pyplot as plt

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    print("Error: 'transformers' library not found. Verification requires it.")
    AutoModelForCausalLM = None

def verify(model_name, config_path):
    """
    Verifies if the indices in 'config_path' actually correspond to low-variance entropy neurons.
    It re-runs the variance calculation on the model and compares the stored indices against the fresh distribution.
    """
    if AutoModelForCausalLM is None:
        return

    # 1. Load Stored Indices
    full_config_path = os.path.join(project_root, config_path)
    if not os.path.exists(full_config_path):
        print(f"Error: Config file not found at {full_config_path}")
        return

    try:
        with open(full_config_path, 'r') as f:
            stored_indices = set(json.load(f))
        print(f"Loaded {len(stored_indices)} indices from {config_path}")
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"Verifying against model: {model_name}...")

    # 2. Load Model & Weights
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cpu", 
            low_cpu_mem_usage=True,
            trust_remote_code=True 
        )
        
        layer_idx = len(model.model.layers) - 1
        down_proj = model.model.layers[layer_idx].mlp.down_proj
        lm_head = model.lm_head

        Wout = down_proj.weight.float().detach() # [d_model, d_mlp]
        WU = lm_head.weight.float().detach()     # [V, d_model]
        
    except Exception as e:
        print(f"Failed to load model or identify layers: {e}")
        return

    # 3. Re-Calculate Variances (Identical Math to Calibration Script)
    vocab_size = WU.shape[0]
    d_mlp = Wout.shape[1] 

    # Sampling 4096 tokens is enough for statistical verification
    V_subset_size = min(vocab_size, 4096)
    idx = torch.randperm(vocab_size)[:V_subset_size]
    WU_sub = WU[idx] 
    
    # Pre-calc Row Norms
    WU_norm = torch.norm(WU_sub, dim=1, keepdim=True)

    batch_size = 256
    variances = []

    print("Re-calculating variances (this may take a minute)...")
    for i in tqdm(range(0, d_mlp, batch_size)):
        Wout_batch = Wout[:, i:i+batch_size]
        Wout_norm_batch = torch.norm(Wout_batch, dim=0, keepdim=True)
        
        L_batch = WU_sub @ Wout_batch
        Norm_Factor = WU_norm @ Wout_norm_batch
        
        # Normalized Logit (Cosine Similarity)
        L_norm = L_batch / (Norm_Factor + 1e-9)
        
        vars_batch = torch.var(L_norm, dim=0)
        variances.append(vars_batch)

    all_variances = torch.cat(variances) # Tensor of shape [d_mlp]

    # 4. Comparative Analysis
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)

    # Get variances for the stored indices
    stored_indices_list = list(stored_indices)
    # Filter out any indices that might be out of bounds (sanity check)
    valid_stored = [i for i in stored_indices_list if i < d_mlp]
    
    if len(valid_stored) == 0:
        print("FAIL: No valid indices found in config file relative to model size.")
        return

    stored_vars = all_variances[valid_stored]
    
    # Get variances for random indices (same count)
    random_indices = torch.randperm(d_mlp)[:len(valid_stored)]
    random_vars = all_variances[random_indices]

    mean_stored = stored_vars.mean().item()
    mean_all = all_variances.mean().item()
    
    print(f"Global Mean Variance:       {mean_all:.6f}")
    print(f"Stored Neurons Mean Var:    {mean_stored:.6f}")
    print(f"Ratio (Stored / Global):    {mean_stored / mean_all:.4f} (Should be << 1.0)")
    
    # Percentile Check
    # What % of stored neurons are in the actual bottom 5% of the new calculation?
    k = int(d_mlp * 0.05)
    current_topk = set(torch.topk(all_variances, k=k, largest=False).indices.tolist())
    
    overlap = len(stored_indices.intersection(current_topk))
    overlap_pct = overlap / len(stored_indices) * 100
    
    print(f"Overlap with fresh calibration: {overlap_pct:.2f}%")
    print("(Note: Overlap < 100% is normal due to random vocabulary sampling, but should be > 50%)")

    if mean_stored < (mean_all * 0.5):
        print("\n✅ PASSED: Stored neurons have significantly lower variance than average.")
    else:
        print("\n❌ FAILED: Stored neurons do not look distinct. You should re-calibrate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Entropy Neuron Calibration.")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--config", type=str, required=True, help="Path to existing JSON config")
    
    args = parser.parse_args()
    verify(args.model, args.config)