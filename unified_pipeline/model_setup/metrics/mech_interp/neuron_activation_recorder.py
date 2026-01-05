import torch
import json
import logging
import csv
import os
from typing import List, Union, Optional

logger = logging.getLogger(__name__)

class NeuronActivationRecorder:
    def __init__(self, 
                 model: torch.nn.Module, 
                 neuron_indices: Union[List[int], str], 
                 layer_name: str, 
                 use_input: bool = True):
        """
        Args:
            model: The PyTorch model to hook into.
            neuron_indices: List of integers OR path to the JSON file created earlier.
            layer_name: The string name of the module to hook (e.g., 'model.layers.31.mlp.down_proj').
            use_input: If True, captures the layer INPUT. For Llama MLP down_proj, 
                       input is the output of up_proj/gate_proj (where neurons live).
        """
        self.model = model
        self.layer_name = layer_name
        self.use_input = use_input
        self.handle = None
        self.recorded_activations = []
        
        logger.info(f"Initializing Activation Recorder for layer: {layer_name}")

        # 1. Load Indices
        if isinstance(neuron_indices, str):
            if not os.path.exists(neuron_indices):
                raise FileNotFoundError(f"Neuron indices file not found: {neuron_indices}")
            
            logger.info(f"Loading neuron indices from file: {neuron_indices}")
            with open(neuron_indices, 'r') as f:
                self.indices = torch.tensor(json.load(f), dtype=torch.long)
        else:
            self.indices = torch.tensor(neuron_indices, dtype=torch.long)

        logger.info(f"Tracking {len(self.indices)} neurons.")

        # 2. Initial Device Placement (Best Guess)
        # We try to put indices on the model's device, but we will double-check in the hook.
        try:
            param = next(model.parameters())
            self.indices = self.indices.to(param.device)
        except StopIteration:
            logger.warning("Model has no parameters? Indices kept on CPU.")
            pass 

    def _hook_fn(self, module, input, output):
        """
        The hook function that runs on every forward pass.
        """
        # 1. Determine source (Input vs Output)
        if self.use_input:
            # Input is usually a tuple containing the tensor
            source_tensor = input[0]
        else:
            source_tensor = output

        # 2. Dynamic Device Handling (Critical for Multi-GPU/Model Parallelism)
        # Ensure indices are on the same device as the incoming tensor
        if self.indices.device != source_tensor.device:
            self.indices = self.indices.to(source_tensor.device)

        # 3. Select specific neurons
        # source_tensor shape: (Batch, Seq_Len, Hidden_Dim)
        try:
            selected = source_tensor[..., self.indices]
        except IndexError as e:
            logger.error(f"IndexError in hook. Tensor shape: {source_tensor.shape}, Max Index: {self.indices.max()}")
            raise e

        # 4. Flatten Batch and Sequence dimensions
        # We treat every token as an independent observation
        # Shape: (Batch * Seq_Len, Num_Selected_Neurons)
        flat_activations = selected.reshape(-1, selected.shape[-1])

        # 5. Detach and move to CPU immediately to save GPU memory
        self.recorded_activations.append(flat_activations.detach().cpu().clone())

    def __enter__(self):
        """Register the hook when entering the context manager."""
        self.recorded_activations = [] # Reset buffer
        
        # Locate the module by string name
        # We search specifically because dict lookup on modules isn't always direct 
        # if the name is nested (e.g. 'model.layers.0').
        target_module = None
        
        # Direct lookup is faster if possible
        try:
            target_module = self.model.get_submodule(self.layer_name)
        except AttributeError:
            # Fallback for older torch versions or complex wrappings
            for name, module in self.model.named_modules():
                if name == self.layer_name:
                    target_module = module
                    break
        
        if target_module is None:
            logger.error(f"Layer '{self.layer_name}' not found in model.")
            # List available layers to help debug
            available = list(name for name, _ in self.model.named_modules())
            logger.debug(f"Available layers (first 5): {available[:5]}")
            raise ValueError(f"Layer {self.layer_name} not found.")

        # Register the forward hook
        self.handle = target_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove the hook when exiting the context manager."""
        if self.handle:
            self.handle.remove()

    def get_activations(self) -> Optional[torch.Tensor]:
        """Returns the concatenated tensor of all recorded passes."""
        if not self.recorded_activations:
            logger.warning("No activations were recorded.")
            return None
            
        # Concatenate along the 0th dimension (Total_Tokens)
        # Result Shape: [Total_Tokens, Num_Neurons]
        cat_activations = torch.cat(self.recorded_activations, dim=0)
        return cat_activations

    def save_activations(self, filepath: str = 'activations.json'):
        """
        Saves the recorded tensor to disk.
        Supports .csv, .json, and .pt.
        """
        data = self.get_activations()
        if data is None:
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        logger.info(f"Saving activations of shape {data.shape} to {filepath}...")
        
        # Ensure indices are on CPU for listing
        indices_list = self.indices.cpu().tolist()

        try:
            if filepath.endswith('.pt') or filepath.endswith('.pth'):
                # Native PyTorch format (Most Efficient)
                torch.save(data, filepath)
                
            elif filepath.endswith('.csv'):
                # CSV Format
                data_list = data.tolist()
                headers = [f"Neuron_{idx}" for idx in indices_list]
                
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers) # Write Header
                    writer.writerows(data_list) # Write Data
                    
            elif filepath.endswith('.json'):
                # JSON Format
                data_list = data.tolist()
                output_dict = {
                    "neuron_indices": indices_list,
                    "activations": data_list
                }
                with open(filepath, 'w') as f:
                    json.dump(output_dict, f, indent=4)
            
            else:
                raise ValueError(f"Unsupported file format: {filepath}. Please use .csv, .json, or .pt")

            logger.info(f"Successfully saved to {filepath}")
            
        except IOError as e:
            logger.error(f"Failed to save activations: {e}")
            raise e