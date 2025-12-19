"""
agent.py
HuggingFace inference agent. 
Wraps model generation and uses uq_core.ActivationMonitor for state capture.
"""
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from uq_core import Trace, ActivationMonitor

class HFAgent:
    def __init__(self, model_id: str):
        self.model_name = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # MPS support for Mac
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
            
        print(f"Loading {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not self.tokenizer.pad_token: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map=self.device, 
            trust_remote_code=True
        ).eval()

    def solve(self, prompt: str, n_samples: int = 1) -> List[Trace]:
        """
        Generates n_samples solutions.
        Captures activations automatically if mechanistic analysis is needed later.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]
        traces = []

        # Loop generation to ensure clean activation capture per sequence
        for _ in range(n_samples):
            monitor = ActivationMonitor(self.model)
            
            with monitor, torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=(n_samples > 1),
                    temperature=0.7 if n_samples > 1 else 0.0,
                    return_dict_in_generate=True, 
                    output_scores=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Process Output
            gen_seq = out.sequences[0, prompt_len:]
            text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
            tokens = self.tokenizer.convert_ids_to_tokens(gen_seq)
            
            # Extract Metrics
            entropies, t1s, t2s = [], [], []
            for scores in out.scores:
                logp = torch.log_softmax(scores[0], dim=-1)
                p = torch.exp(logp)
                entropies.append(-torch.sum(torch.nan_to_num(p * logp)).item())
                top2 = torch.topk(logp, 2)
                t1s.append(top2.values[0].item())
                t2s.append(top2.values[1].item())

            # Align Activations (Core Logic)
            acts = monitor.get_aligned_activations(prompt_len, len(entropies))
            if acts is not None: acts = acts[0]

            # Safety Truncate
            min_l = min(len(tokens), len(entropies))
            if acts is not None: min_l = min(min_l, len(acts))

            traces.append(Trace(
                text=text, 
                tokens=tokens[:min_l], 
                entropies=entropies[:min_l],
                top1_logprobs=t1s[:min_l], 
                top2_logprobs=t2s[:min_l],
                activations=acts[:min_l] if acts is not None else None
            ))
            
        return traces