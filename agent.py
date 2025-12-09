"""
Agent implementations for generating solutions and associated
uncertainty metrics.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# --- INSERTED ACTIVATION MONITOR CLASS (No external file needed) ---

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
                # Llama/Mistral/DeepSeek/Qwen style
                # We hook the activation function (e.g., GeLU, SiLU) within the last MLP block.
                self.target_layer = self.model.model.layers[-1].mlp.act_fn
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                 # GPT-2 style (MLP structure is slightly different)
                 self.target_layer = self.model.transformer.h[-1].mlp
            else:
                pass
        except Exception:
            pass

    def _hook_fn(self, module, input, output):
        # Output shape is typically [Batch, SeqLen, MLP_Dim].
        # We only care about the activations for the newly generated token (the last one in SeqLen).
        if output.ndim == 3:
            self.activations.append(output[:, -1, :].detach().cpu())
        elif output.ndim == 2:
             self.activations.append(output.detach().cpu())

    def __enter__(self):
        if self.target_layer:
            self.hook = self.target_layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook:
            self.hook.remove()

    def get_batch_activations(self, expected_batch_size=1, expected_seq_len=None):
        """
        Processes the stored activations after generation completes.
        Reshapes the list of tensors into [Batch, Tokens, MLP_Dim].
        """
        # Default dim for 7B models if config access fails
        d_mlp = 11008 
        if hasattr(self.model, 'config'):
             d_mlp = getattr(self.model.config, 'intermediate_size', d_mlp)
             
        seq_len = expected_seq_len if expected_seq_len is not None else 32

        if self.activations:
            try:
                # Stack along a new dimension: [Tokens, Batch, MLP_Dim]
                acts = torch.stack(self.activations, dim=0)
                # Permute to: [Batch, Tokens, MLP_Dim]
                acts = acts.permute(1, 0, 2).numpy()
                
                print(f"ActivationMonitor Success: Captured activations with shape {acts.shape}")
                return acts
            except Exception as e:
                print(f"ActivationMonitor Error processing stack: {e}")

        # Fallback: Return ZEROS (Safe, non-random) if capture failed
        print("ActivationMonitor: Returning zero-filled tensor due to capture failure.")
        return np.zeros((expected_batch_size, seq_len, d_mlp), dtype=np.float32)

# -------------------------------------------------------------------

# Optional external API client for CoTAgent.
try:
    from openai import OpenAI  # type: ignore
    CLIENT: Optional[Any] = None
except Exception:
    CLIENT = None  # type: ignore


@dataclass
class Trace:
    """
    A structure holding the output and internal metrics of a generation.
    """
    text: str
    tokens: List[str] = field(default_factory=list)
    # Entropic UQ data
    entropies: List[float] = field(default_factory=list)
    top1_logprobs: List[float] = field(default_factory=list)
    top2_logprobs: List[float] = field(default_factory=list)
    # Mechanistic UQ data
    activations: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class BaseAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def solve(self, task: str, n_samples: int = 1, capture_activations: bool = True) -> List[Trace]:
        raise NotImplementedError("The 'solve' method must be implemented by subclasses.")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model='{self.model_name}')>"


class CoTAgent(BaseAgent):
    """
    Agent utilizing chain‑of‑thought prompting via an external API.
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model_name=model)
        self.client: Optional[Any] = CLIENT
        if self.client is None:
            print(f"Info: OpenAI client not initialized for {model}. CoTAgent will use mock responses.")

    def _entropy(self, logprobs: np.ndarray) -> float:
        p = np.exp(logprobs)
        p_sum = np.sum(p)
        if p_sum > 0: p = p / p_sum
        plogp = np.where(p > 0, p * np.log(p), 0)
        return -float(np.sum(plogp))

    def _mock(self, task: str, n: int) -> List[Trace]:
        text = f"Step 1: Analyze '{task[:50]}...'. Step 2: Calculation 10*5=50. Step 3: Review. The final answer is 42."
        toks = text.split()
        ent = (np.random.rand(len(toks)) * 1.0 + 0.1).tolist()
        t1 = (-np.random.rand(len(toks)) * 0.2 - 0.05).tolist()
        t2 = [(a - (np.random.rand() * 1.5 + 0.5)) for a in t1]
        D_MLP = 14336
        acts = np.random.randn(len(toks), D_MLP).astype(np.float32) * 0.5 - 0.1
        return [
            Trace(text=text, tokens=toks, entropies=ent, top1_logprobs=t1, top2_logprobs=t2, activations=acts)
            for _ in range(n)
        ]

    def solve(self, task: str, n_samples: int = 1, capture_activations: bool = False) -> List[Trace]:
        # CoTAgent (API) cannot capture mechanistic activations, so capture_activations arg is ignored but kept for signature compatibility
        if self.client is None:
            return self._mock(task, n_samples)
        prompt = (
            "Solve the following problem step‑by‑step.\n"
            f"Question: {task}\n"
            "Solution:"
        )
        try:
            r = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                n=n_samples,
                logprobs=True,
                top_logprobs=5,
                temperature=0.7 if n_samples > 1 else 0.0,
            )
        except Exception as e:
            print(f"API request failed: {e}. Using mock response.")
            return self._mock(task, n_samples)
        out: List[Trace] = []
        for c in r.choices:
            lp = getattr(c, "logprobs", None)
            if not lp or not lp.content: continue
            toks: List[str] = []
            ents: List[float] = []
            t1: List[float] = []
            t2: List[float] = []
            for t in lp.content:
                toks.append(t.token)
                vals = sorted([x.logprob for x in t.top_logprobs], reverse=True)
                ents.append(self._entropy(np.array(vals)))
                t1.append(vals[0])
                t2.append(vals[1] if len(vals) > 1 else -np.inf)
            out.append(
                Trace(
                    text=c.message.content.strip(),
                    tokens=toks,
                    entropies=ents,
                    top1_logprobs=t1,
                    top2_logprobs=t2,
                    activations=None,
                )
            )
        return out


class HFAgent(BaseAgent):
    """
    Optimized HFAgent with integrated ActivationMonitor.
    """
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", device: Optional[str] = None):
        super().__init__(model_name)
        self.model_name = model_name
        self.device: Optional[str] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        print(f"Initializing HFAgent ({model_name}) on device: {self.device}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            print("Warning: 'transformers' library not found. HFAgent will run in mock mode.")
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model.eval()
            print("HFAgent initialized successfully.")
        except Exception as e:
            print(f"Error initializing HFAgent for {model_name}: {e}")
            self.model = None

    def _compute_batch_entropy(self, scores: List[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
            with torch.no_grad():
                logits = torch.stack(scores) 
                logp = torch.log_softmax(logits, dim=-1)
                p = logp.exp()
                plogp = p * logp
                plogp = torch.nan_to_num(plogp, nan=0.0)
                
                ent_device = -torch.sum(plogp, dim=-1)
                ent_device = ent_device.permute(1, 0)
                
                topk_vals_device = torch.topk(logp, 2, dim=-1).values
                topk_vals_device = topk_vals_device.permute(1, 0, 2)

                ent = ent_device.cpu().float().numpy()
                topk = topk_vals_device.cpu().float().numpy()

            return ent, topk

    def _mock_solve(self, task: str, n_samples: int = 1) -> List[Trace]:
        mock_agent = CoTAgent(model="mock")
        return mock_agent._mock(task, n_samples)

    def solve(self, task: str, n_samples: int = 1, capture_activations: bool = True) -> List[Trace]:
        if self.model is None or self.tokenizer is None:
            return self._mock_solve(task, n_samples)
            
        try:
            msgs = [
                {"role": "system", "content": "Solve the problem step‑by‑step."},
                {"role": "user", "content": task},
            ]
            prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"Question: {task}\nAnswer:"
            
        prompts = [prompt] * n_samples
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        base_len = inputs.input_ids.shape[1]
        
        generation_config = {
            "max_new_tokens": 512,
            "do_sample": n_samples > 1,
            "temperature": 0.7 if n_samples > 1 else None,
            "top_p": 0.9 if n_samples > 1 else None,
            "return_dict_in_generate": True,
            "output_scores": True, 
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Use integrated Monitor ONLY if requested
        monitor = ActivationMonitor(self.model) if capture_activations else None
        
        try:
            if monitor: monitor.__enter__()
            with torch.no_grad():
                out = self.model.generate(**inputs, **generation_config)
        except Exception as e:
            print(f"Error during HFAgent generation: {e}. Falling back to mock.")
            return self._mock_solve(task, n_samples)
        finally:
            if monitor: monitor.__exit__(None, None, None)

        seq = out.sequences[:, base_len:]
        texts = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        seq_len = seq.shape[1]
        
        ent, topk = self._compute_batch_entropy(out.scores)
        
        # Retrieve activations ONLY if monitor was active
        acts_batch = None
        if monitor:
            acts_batch = monitor.get_batch_activations(expected_batch_size=n_samples, expected_seq_len=seq_len)
            
        traces: List[Trace] = []
        for i in range(n_samples):
            toks = self.tokenizer.convert_ids_to_tokens(seq[i])
            T = min(len(toks), ent.shape[1])
            
            acts_i = None
            if acts_batch is not None and i < len(acts_batch) and acts_batch[i] is not None:
                a = acts_batch[i]
                if hasattr(a, 'cpu'): a = a.cpu().numpy()
                if T <= a.shape[0]:
                    acts_i = a[:T]
                else:
                    acts_i = a

            traces.append(
                Trace(
                    text=texts[i],
                    tokens=toks[:T],
                    entropies=ent[i, :T].tolist(),
                    top1_logprobs=topk[i, :T, 0].tolist(),
                    top2_logprobs=topk[i, :T, 1].tolist(),
                    activations=acts_i,
                )
            )
            
        return traces