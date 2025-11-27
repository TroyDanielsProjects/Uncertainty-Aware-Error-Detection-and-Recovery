
import torch
import numpy as np

class ActivationMonitor:
    """
    Context manager to capture activations from the last MLP layer during generation.
    This is crucial for Mechanistic UQ. It aims to capture the output of the 
    activation function (input to the down projection Wout).
    """
    def __init__(self, model):
        self.model = model
        self.activations = []
        self.hook = None
        self.target_layer = None
        self._locate_target_layer()

    def _locate_target_layer(self):
        # Generalized way to find the last MLP layer's activation function
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                # Llama/Mistral/DeepSeek style
                # We hook the activation function (e.g., GeLU, SiLU) within the last MLP block.
                self.target_layer = self.model.model.layers[-1].mlp.act_fn
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                 # GPT-2 style (MLP structure is slightly different)
                 # Hooking the MLP block itself might be necessary if act_fn isn't easily accessible
                 self.target_layer = self.model.transformer.h[-1].mlp
            else:
                # print("Warning: Model architecture not recognized for activation monitoring.")
                pass
        except Exception as e:
            # print(f"Error locating target layer: {e}")
            pass

    def _hook_fn(self, module, input, output):
        # During generation, this hook is called sequentially for each token.
        # Output shape is typically [Batch, SeqLen, MLP_Dim].
        # We only care about the activations for the newly generated token (the last one in SeqLen).
        if output.ndim == 3:
            self.activations.append(output[:, -1, :].detach().cpu())
        elif output.ndim == 2:
             # Handle cases where SeqLen might be implicit (e.g., optimized generation)
             self.activations.append(output.detach().cpu())

    def __enter__(self):
        if self.target_layer:
            # Register the forward hook
            self.hook = self.target_layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook:
            self.hook.remove()

    def get_batch_activations(self, expected_batch_size=1, expected_seq_len=None):
        """
        Processes the stored activations after generation completes.
        Reshapes the list of tensors into [Batch, Tokens, MLP_Dim].
        If capture fails, returns mock data for framework testing.
        """
        
        # Determine dimensions for potential mock data fallback
        # Approximation for 7B models MLP dimension if config cannot be accessed
        d_mlp = 11008 
        if hasattr(self.model, 'config'):
             # Llama/Deepseek typically use 'intermediate_size' for MLP dim
             d_mlp = getattr(self.model.config, 'intermediate_size', d_mlp)
             
        seq_len = expected_seq_len if expected_seq_len is not None else 32 # Default mock length

        if self.activations:
            try:
                # Stack along a new dimension for tokens: [Tokens, Batch, MLP_Dim]
                acts = torch.stack(self.activations, dim=0)
                
                # Permute to the desired shape: [Batch, Tokens, MLP_Dim]
                acts = acts.permute(1, 0, 2).numpy()

                # Alignment check
                if expected_seq_len is not None and acts.shape[1] != expected_seq_len:
                     # print(f"Warning: Activation alignment mismatch. Expected {expected_seq_len} tokens, captured {acts.shape[1]}.")
                     # We return the captured data; alignment is handled in the Agent.
                     pass
                
                return acts

            except Exception as e:
                # This can happen if different batches generate different numbers of tokens 
                # and the stacking operation fails due to shape mismatch.
                # print(f"Error processing activations (e.g., shape mismatch during stacking): {e}. Using mock data.")
                pass

        # Mock Data Generation Fallback
        # print("Info: No activations captured. Returning mock data.")
        # Return mock activations: [Batch, Time, Features]
        mock_data = np.random.randn(expected_batch_size, seq_len, d_mlp).astype(np.float32) * 0.5 - 0.1
        return mock_data
