
import torch
import numpy as np
from typing import List

# Ensure correct imports
from agents.base_agent import BaseAgent, Trace
from mechanistic_interpretability.activation_utils import ActivationMonitor

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: 'transformers' library not found. HFAgent requires it.")
    AutoModelForCausalLM = None
    AutoTokenizer = None

class HFAgent(BaseAgent):
    """
    Agent running local HuggingFace models. Enables both Entropic and Mechanistic UQ.
    """
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", device=None):
        super().__init__(model_name)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if AutoTokenizer is None:
            return

        # Device configuration
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
             self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Initializing HFAgent ({model_name}) on device: {self.device}")

        # Load Model and Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Attempt loading with optimized settings (float16, auto device map)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device not in ["cpu", "mps"] else self.device,
                low_cpu_mem_usage=True,
                trust_remote_code=True # Necessary for some architectures (e.g., DeepSeek)
            )
            self.model.eval()
            print("HFAgent initialized successfully.")

        except Exception as e:
            print(f"Error initializing HFAgent for {model_name}: {e}")
            print("HFAgent will run in mock mode. Check model availability and resources.")
            self.model = None # Set to None to trigger mock mode in solve()


    def _compute_batch_entropy(self, scores):
        """
        Computes entropy and top-k logprobs from the scores tuple.
        scores shape: (T, B, V) where T=Tokens generated, B=Batch size, V=Vocab size.
        """
        logits = torch.stack(scores)               # [T, B, V]
        
        logp = torch.log_softmax(logits, dim=-1)   # [T, B, V]
        probs = torch.softmax(logits, dim=-1)      # [T, B, V]

        # Calculate Entropy: H(p) = -sum(p * log(p))
        # Use torch.where for stable calculation of 0*log(0)
        plogp = torch.where(probs > 0, probs * logp, 0.0)
        ent = -torch.sum(plogp, dim=-1)            # [T, B]

        # Get Top-2 LogProbs for Logit Gap calculation
        topk_logp = torch.topk(logp, 2, dim=-1).values # [T, B, 2]

        # Permute dimensions to [B, T, ...] and move to CPU
        return (
            ent.permute(1, 0).cpu().numpy(),          # [B, T]
            topk_logp.permute(1, 0, 2).cpu().numpy()  # [B, T, 2]
        )

    def _format_prompt(self, task):
        # Apply chat templates if available
        try:
            msgs = [
                {"role": "system", "content": "Solve the problem step-by-step."},
                {"role": "user", "content": task}
            ]
            prompt = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for base models or if template fails
            prompt = f"Question: {task}\nAnswer:"
        return prompt

    def _mock_solve(self, task: str, n_samples: int = 1) -> List[Trace]:
        # Mock implementation used when the model fails to load
        print(f"HFAgent running in MOCK mode for {self.model_name}.")
        
        # Use CoTAgent's mock capabilities as a robust fallback
        # We need to ensure CoTAgent is available for this import
        try:
            from agents.cot_agent import CoTAgent
            mock_agent = CoTAgent(model="mock")
            return mock_agent._mock(task, n_samples)
        except ImportError:
            print("Error: Cannot run mock solve, CoTAgent import failed.")
            return []


    def solve(self, task: str, n_samples: int = 1) -> List[Trace]:
        if self.model is None:
            return self._mock_solve(task, n_samples)

        prompt = self._format_prompt(task)
        prompts = [prompt] * n_samples # Create a batch

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        base_len = inputs.input_ids.shape[1]

        generation_config = {
            "max_new_tokens": 512,
            "do_sample": n_samples > 1,
            "temperature": 0.7 if n_samples > 1 else None,
            "top_p": 0.9 if n_samples > 1 else None,
            "return_dict_in_generate": True,
            "output_scores": True, # Crucial for Entropic UQ
            "pad_token_id": self.tokenizer.pad_token_id
        }

        # Use the ActivationMonitor context manager for Mechanistic UQ
        with ActivationMonitor(self.model) as mon:
            with torch.no_grad():
                try:
                    out = self.model.generate(**inputs, **generation_config)
                except Exception as e:
                    print(f"Error during HFAgent generation: {e}. Falling back to mock.")
                    return self._mock_solve(task, n_samples)

        # Decode the generated sequences
        seq = out.sequences[:, base_len:]
        texts = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        seq_len = seq.shape[1]

        # Compute Entropic UQ metrics
        ent, topk = self._compute_batch_entropy(out.scores)
        
        # Retrieve Mechanistic UQ data (activations)
        # ActivationMonitor handles the alignment/mocking
        acts_batch = mon.get_batch_activations(expected_batch_size=n_samples, expected_seq_len=seq_len)

        # Create Trace objects
        traces = []
        for i in range(n_samples):
            toks = self.tokenizer.convert_ids_to_tokens(seq[i])
            
            # Ensure lengths match (T_gen vs T_ent)
            T = min(len(toks), ent.shape[1])
            
            # Align activations (T_act)
            acts_i = None
            if i < len(acts_batch) and acts_batch[i] is not None:
                 T_act = acts_batch[i].shape[0]
                 if T_act >= T:
                      acts_i = acts_batch[i][:T]
                 # If T_act < T, it means activation capture missed some tokens.
                 # We omit the activations in this case for simplicity, or could pad them.
                 # else: print(f"Warning: Activations shorter than generation for sample {i}.")

            traces.append(Trace(
                text=texts[i],
                tokens=toks[:T],
                entropies=ent[i, :T].tolist(),
                top1_logprobs=topk[i, :T, 0].tolist(),
                top2_logprobs=topk[i, :T, 1].tolist(),
                activations=acts_i
            ))

        return traces
