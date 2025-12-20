import torch
import json
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv

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
            layer_name: The string name of the module to hook (e.g., 'transformer.h.0.mlp').
            use_input: If True, captures the layer INPUT (good for down_proj to get Up/Gate neurons). 
                       If False, captures OUTPUT.
        """
        self.model = model
        self.layer_name = layer_name
        self.use_input = use_input
        self.handle = None
        self.recorded_activations = []
        
        # Load indices if a file path is provided, otherwise use the list
        if isinstance(neuron_indices, str):
            with open(neuron_indices, 'r') as f:
                self.indices = torch.tensor(json.load(f), dtype=torch.long)
        else:
            self.indices = torch.tensor(neuron_indices, dtype=torch.long)

        # Move indices to the same device as the model for slicing
        # Check first parameter device to determine where to put indices
        try:
            param = next(model.parameters())
            self.indices = self.indices.to(param.device)
        except StopIteration:
            pass # Empty model

    def _hook_fn(self, module, input, output):
        """
        The hook function that runs on every forward pass.
        """
        # 1. Determine source (Input vs Output)
        # For Llama MLP down_proj, the neurons are in the INPUT (from up_proj/gate_proj).
        # For standard Linear layers, the neurons are often the OUTPUT.
        if self.use_input:
            # input is a tuple, usually (tensor, )
            source_tensor = input[0]
        else:
            source_tensor = output

        # 2. Select specific neurons
        # source_tensor shape: (Batch, Seq_Len, Hidden_Dim/Intermediate_Dim)
        selected = source_tensor[..., self.indices]

        # 3. Flatten Batch and Sequence dimensions
        # This fixes the "shape mismatch" error. We treat all tokens as independent samples.
        # New shape: (Batch * Seq_Len, Num_Indices)
        flat_activations = selected.reshape(-1, selected.shape[-1])

        # 4. Detach and move to CPU
        self.recorded_activations.append(flat_activations.detach().cpu().clone())

    def __enter__(self):
        """Register the hook when entering the context manager."""
        self.recorded_activations = [] # Reset buffer
        
        # Find the specific layer by name
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                target_module = module
                break
        
        if target_module is None:
            raise ValueError(f"Layer {self.layer_name} not found in model.")

        # Register the forward hook
        self.handle = target_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove the hook when exiting the context manager."""
        if self.handle:
            self.handle.remove()

    def get_activations(self):
        """Returns the concatenated tensor of all recorded passes."""
        if not self.recorded_activations:
            return None
        # Concatenate along the 0th dimension (Total_Tokens)
        return torch.cat(self.recorded_activations, dim=0)

    def save_activations(self, filepath='activations.json'):
        """
        Saves the recorded tensor to disk.
        Supports .csv (readable), .json (readable), and .pt (efficient).
        """
        data = self.get_activations()
        if data is None:
            print("No activations recorded.")
            return

        print(f"Preparing to save activations of shape {data.shape}...")
        
        # Ensure indices are on CPU for listing
        indices_list = self.indices.cpu().tolist()

        if filepath.endswith('.pt') or filepath.endswith('.pth'):
            # Native PyTorch format
            torch.save(data, filepath)
            
        elif filepath.endswith('.csv'):
            # CSV Format: Readable, importable into Excel/Pandas
            data_list = data.tolist()
            # Create headers: "Neuron_10", "Neuron_45", etc.
            headers = [f"Neuron_{idx}" for idx in indices_list]
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers) # Write Header
                writer.writerows(data_list) # Write Data rows
                
        elif filepath.endswith('.json'):
            # JSON Format: Readable, but can be verbose
            data_list = data.tolist()
            output_dict = {
                "neuron_indices": indices_list,
                "activations": data_list
            }
            with open(filepath, 'w') as f:
                json.dump(output_dict, f, indent=4)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath}. Please use .csv, .json, or .pt")

        print(f"Saved to {filepath}")

# --- Usage Example ---
if __name__ == "__main__":
    # Mock model and data for demonstration
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Meta-Llama-3.1-8B",
            torch_dtype=torch.float32,
            device_map='cpu', 
            low_cpu_mem_usage=True,
            trust_remote_code=True 
        )
        # Set the tokenizer for the model
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")
        print("Loaded in model successfully")
    except Exception as e:
        print(f"Failed to load model unsloth/Meta-Llama-3.1-8B: {e}")
    indices_path = 'data.json' # Created by your previous step
    
    # Example usage
    # We use a context manager (with statement) to automatically handle 
    # registering and removing the hook.
    try:
        # Note: In a real transformer, layer_name might be "model.layers.10.mlp.down_proj"
        # keep in mind that we do not want Wout but actually the layer beforehand (activations)
        recorder = NeuronActivationRecorder(model, indices_path, layer_name='model.layers.31.mlp.down_proj') 
        
        with recorder:
            input = tokenizer("what is 2+2: Answer is ", return_tensors="pt").to("cpu")
            model.generate(**input, max_new_tokens=2)
            
        # Access results
        results = recorder.get_activations()
        print(f"Captured activations for {len(recorder.indices)} neurons.")
        print(f"Final shape: {results.shape}") # Should be [10, num_indices]
        recorder.save_activations()
        
    except FileNotFoundError:
        print("Please ensure data.json exists from the previous step.")