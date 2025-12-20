import torch
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import sys
from transformers import AutoModelForCausalLM

class Entropy_Neurons_Identification:

    def __init__(self, 
             model_name="unsloth/Meta-Llama-3.1-8B", 
             output_path="entropy_neurons.json",
             dtype = torch.float32,
             device = 'cpu',
             k=10):
        
        self.model_name = model_name
        self.output_path = output_path
        self.k = k

            # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=device, 
                low_cpu_mem_usage=True,
                trust_remote_code=True 
            )
            print("Loaded in model successfully")
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")

        # Identify Wout and WU
        try:
            layer_idx = len(self.model.model.layers) - 1
            down_proj = self.model.model.layers[layer_idx].mlp.down_proj
            lm_head = self.model.lm_head
            print(f"Layer name is: model.model.layers[{layer_idx}].mlp.down_proj")
            self.Wout = down_proj.weight.float().detach() # [d_model, d_mlp]
            self.WU = lm_head.weight.float().detach()     # [V, d_model]
            self.L = self.WU @ self.Wout
            print("Successfully detached Wout and WU matrices")
        except AttributeError:
            print(f"Error: Could not identify necessary layers in {model_name}.")


    def identify(self, identify_by_cosine_sim=False, identify_by_variance=True, identify_by_max_std_dev=True, file_name='entropy_neurons.json'):
        """
        Identifies Entropy Neurons based on Normalized Logit Attribution Variance (Cosine Similarity).
        
        consistent with 'entropy_neurons.ipynb':
        1. L = WU @ Wout (Logit attribution)
        2. Select entropy by cosine_sim
            1a. NormFactor = ||WU||_dim1 * ||Wout||_dim0
            2a. L_norm = L / NormFactor
            3a. Select neurons with LOW variance in L_norm.
        5. Select by variance
            1b. Compute variance over L
            2b. Select neurons with LOW variance in L
        5. Select by max std dev
            1c. Compute the max standard deviation (l-infty-std-dev)
            2c. Select neurons with LOW max Std Dev
        """

        # list to keep track of all entropy neurons
        neuron_candidates = []

        # identify the entropy neurons through different tecniques
        if identify_by_cosine_sim:
            neuron_candidates.append(self.cosine_sim_identification())
            print("Successfully Identified Neurons bases on Cosine Similarity")
        if identify_by_variance:
            vars = torch.var(self.L, dim=0)
            print(vars.shape)
            neuron_candidates.append(torch.topk(vars, k=self.k, largest=False).indices)
            print("Successfully Identified Neurons based on variance")
        if identify_by_max_std_dev:
            neuron_candidates.append(self.max_std_dev_identification())
            print("Successfully Identified Neurons based on max std dev")

        # 2. Concatenate all collected tensors
        if len(neuron_candidates) > 0:
            entropy_neurons = torch.cat(neuron_candidates)
        else:
            entropy_neurons = torch.tensor([], dtype=torch.long)

        with open(file_name, 'w') as file:
            # 3. Dump the list into the file
            # indent=4 makes it readable (pretty-printed)
            json.dump(entropy_neurons.tolist(), file, indent=4)

        print(f"Calibration complete. Saved {len(entropy_neurons)} indices to {self.output_path}")

    def cosine_sim_identification(self):
        # 1. Pre-calculate ||WU|| (Row norms) for the subset
        # Shape: [V_sub, 1]
        WU_norm = torch.norm(self.WU, dim=1, keepdim=True)

        batch_size = 256
        variances = []

        print("Calculating normalized logit variances per neuron...")

        d_mlp = self.Wout.shape[1]
        
        for i in tqdm(range(0, d_mlp, batch_size)):
            d_mlp = self.Wout.shape[1] 
            Wout_batch = self.Wout[:, i:i+batch_size] # [d_model, batch_size]
            
            # 2. Calculate ||Wout|| (Column norms) for this batch
            # Shape: [1, batch_size]
            Wout_norm_batch = torch.norm(Wout_batch, dim=0, keepdim=True)
            
            # 3. Compute Raw Logits: ψ = WU @ Wout
            # Shape: [V_sub, batch_size]
            L_batch = self.WU @ Wout_batch
            
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
        
        topk_indices = torch.topk(all_variances, k=self.k, largest=False).indices
        return topk_indices

    def max_std_dev_identification(self):
        # calculate the mean over the vocabulary dimension
        expectation = torch.mean(self.L, dim=0)
        # Want to take the max distance from the mean over the vocab dimension for each neuron
        l_inf_sttdev = torch.max(torch.abs((self.L - expectation)), dim=0).values
        topk_indices = torch.topk(l_inf_sttdev, k=self.k, largest=False).indices
        return topk_indices

if __name__ == "__main__":
    identify_neurons = Entropy_Neurons_Identification()
    identify_neurons.identify(identify_by_cosine_sim=True, identify_by_max_std_dev=False, file_name='cosine_var_entropy_neurons.json')