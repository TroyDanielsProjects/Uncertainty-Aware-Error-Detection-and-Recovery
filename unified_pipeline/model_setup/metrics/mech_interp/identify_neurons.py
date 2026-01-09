import torch
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
import logging
import sys
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

class Entropy_Neurons_Identification:

    def __init__(
                self, 
                model_name,
                model, 
                output_path,
                k
    ):
        
        logger.info(f"Initializing Entropy Neuron Identification for {model_name}")
        
        self.model = model
        self.output_path = output_path
        self.k = k

        # Identify Wout and WU
        try:
            # 1. Resolve Base Model (Llama/Gemma/Qwen usually use .model, older Qwen might use .transformer)
            if hasattr(self.model, "model"):
                base_model = self.model.model
            elif hasattr(self.model, "transformer"):
                base_model = self.model.transformer
            else:
                base_model = self.model
                logger.warning(f"Could not find .model or .transformer attribute. Attempting to use model instance directly.")

            # 2. Resolve Layers
            if hasattr(base_model, "layers"):
                layers = base_model.layers
            elif hasattr(base_model, "h"): # Fallback for some architectures
                layers = base_model.h
            else:
                raise AttributeError(f"Could not find 'layers' or 'h' in base model for {model_name}")

            layer_idx = len(layers) - 1
            logger.info(f"Targeting last layer index: {layer_idx}")
            
            last_layer = layers[layer_idx]

            # 3. Resolve MLP Down Projection
            # Standard Llama/Gemma/Qwen2+ use 'mlp.down_proj'
            # gemma - model.layers.17.mlp.down_proj.weight
            if hasattr(last_layer, "mlp"):
                mlp_module = last_layer.mlp
                if hasattr(mlp_module, "down_proj"):
                    down_proj = mlp_module.down_proj
                elif hasattr(mlp_module, "c_proj"): # Sometimes seen in older Qwen/GPT-like
                    down_proj = mlp_module.c_proj
                    logger.info("Detected 'c_proj' instead of 'down_proj' (Qwen/GPT style).")
                else:
                    raise AttributeError(f"Could not find 'down_proj' or 'c_proj' in MLP layer for {model_name}")
            else:
                raise AttributeError(f"Could not find 'mlp' module in last layer for {model_name}")

            # 4. Resolve LM Head
            # Usually self.model.lm_head
            if hasattr(self.model, "lm_head"):
                lm_head = self.model.lm_head
            elif hasattr(self.model, "embed_out"): # Some architectures
                lm_head = self.model.embed_out
            else:
                raise AttributeError(f"Could not find 'lm_head' for {model_name}")

            logger.info(f"Layer path identified for {model_name}. Extracting weights...")

            self.Wout = down_proj.weight.float().detach() # [d_model, d_mlp]
            self.WU = lm_head.weight.float().detach()     # [V, d_model]

            logger.info("Computing Logit Attribution Matrix L (WU @ Wout)...")

            self.L = self.WU @ self.Wout

            logger.info(f"Successfully computed L matrix with shape {self.L.shape}")

        except AttributeError as e:
            logger.critical(f"Error: Could not identify necessary layers in {model_name}. Structure might differ.")
            raise e


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
            logger.info("Starting Cosine Similarity Identification...")
            neuron_candidates.append(self.cosine_sim_identification())
            logger.info("Successfully Identified Neurons based on Cosine Similarity")
        if identify_by_variance:
            logger.info("Starting Variance Identification...")
            vars = torch.var(self.L, dim=0)
            # print(vars.shape) 
            neuron_candidates.append(torch.topk(vars, k=self.k, largest=False).indices)
            logger.info("Successfully Identified Neurons based on Variance")
        if identify_by_max_std_dev:
            logger.info("Starting Max Deviation Identification...")
            neuron_candidates.append(self.max_std_dev_identification())
            logger.info("Successfully Identified Neurons based on Max Std Dev")

        # 2. Concatenate all collected tensors
        if len(neuron_candidates) > 0:
            entropy_neurons = torch.cat(neuron_candidates)
        else:
            logger.warning("No identification methods selected. Returning empty list.")
            entropy_neurons = torch.tensor([], dtype=torch.long)

        # Ensure output directory exists
        if os.path.dirname(self.output_path):
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
        full_path = os.path.join(self.output_path, file_name) if os.path.isdir(self.output_path) else self.output_path

        with open(full_path, 'w') as file:
            # 3. Dump the list into the file
            # indent=4 makes it readable (pretty-printed)
            json.dump(entropy_neurons.tolist(), file, indent=4)
        logger.info(f"Calibration complete. Saved {len(entropy_neurons)} indices to {full_path}")


    def cosine_sim_identification(self):
        # 1. Pre-calculate ||WU|| (Row norms) for the subset
        # Shape: [V_sub, 1]
        WU_norm = torch.norm(self.WU, dim=1, keepdim=True)

        batch_size = 256
        variances = []

        d_mlp = self.Wout.shape[1]
        
        for i in tqdm(range(0, d_mlp, batch_size)):
            # d_mlp = self.Wout.shape[1] # Removed redundant assignment inside loop
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