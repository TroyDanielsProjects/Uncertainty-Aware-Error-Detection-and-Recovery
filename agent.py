"""
Agent implementations for generating solutions and associated
uncertainty metrics.

This module consolidates the agent classes used by the main pipeline.
It includes:

  * Trace – a dataclass representing a single model generation with
    token level information and optional activations.
  * BaseAgent – an abstract base class defining the ``solve`` API.
  * CoTAgent – a mockable chain‑of‑thought agent that can call an
    external API or fall back to synthetic responses.
  * HFAgent – a HuggingFace based agent capable of returning
    entropic and mechanistic metrics via activation hooks.

The original agent implementations were scattered across multiple
modules; here they are rewritten in a single file and stripped of any
package‑relative imports so they can be used directly.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# ActivationMonitor is used by HFAgent to capture hidden activations.
try:
    from mechanistic_interpretability.activation_utils import ActivationMonitor  # type: ignore
except ImportError:
    ActivationMonitor = None  # type: ignore

# Optional external API client for CoTAgent.  If unavailable,
# CoTAgent will fall back to mock responses.
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
    # Mechanistic UQ data (activations from the last MLP layer).
    activations: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dictionary‑like accessor for compatibility with the UQ framework.
        """
        return getattr(self, key, default)


class BaseAgent:
    """
    Abstract base class for language model agents.  Subclasses must
    implement the ``solve`` method which generates one or more
    ``Trace`` objects for a given task.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def solve(self, task: str, n_samples: int = 1) -> List[Trace]:
        raise NotImplementedError("The 'solve' method must be implemented by subclasses.")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(model='{self.model_name}')>"


class CoTAgent(BaseAgent):
    """
    Agent utilizing chain‑of‑thought prompting via an external API.
    If the API client is unavailable (no OpenAI key) the agent will
    return synthetic traces for testing.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        super().__init__(model_name=model)
        self.client: Optional[Any] = CLIENT
        if self.client is None:
            # no API client – operate in mock mode
            print(f"Info: OpenAI client not initialized for {model}. CoTAgent will use mock responses.")

    def _entropy(self, logprobs: np.ndarray) -> float:
        """
        Compute the Shannon entropy of a set of log probabilities.
        """
        p = np.exp(logprobs)
        p_sum = np.sum(p)
        if p_sum > 0:
            p = p / p_sum
        plogp = np.where(p > 0, p * np.log(p), 0)
        return -float(np.sum(plogp))

    def _mock(self, task: str, n: int) -> List[Trace]:
        """
        Generate deterministic mock responses when the API is not available.
        """
        text = f"Step 1: Analyze '{task[:50]}...'. Step 2: Calculation 10*5=50. Step 3: Review. The final answer is 42."
        toks = text.split()
        # synthetic entropies and logprobs
        ent = (np.random.rand(len(toks)) * 1.0 + 0.1).tolist()
        t1 = (-np.random.rand(len(toks)) * 0.2 - 0.05).tolist()
        t2 = [(a - (np.random.rand() * 1.5 + 0.5)) for a in t1]
        # synthetic activations (T, H) with a standard hidden dim
        D_MLP = 14336
        acts = np.random.randn(len(toks), D_MLP).astype(np.float32) * 0.5 - 0.1
        return [
            Trace(text=text, tokens=toks, entropies=ent, top1_logprobs=t1, top2_logprobs=t2, activations=acts)
            for _ in range(n)
        ]

    def solve(self, task: str, n_samples: int = 1) -> List[Trace]:
        if self.client is None:
            return self._mock(task, n_samples)
        # build a prompt for the API
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
            if not lp or not lp.content:
                continue
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
    Optimized HFAgent that offloads measurement math to CPU to avoid MPS stalls.
    """

    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", device: Optional[str] = None):
        super().__init__(model_name)
        self.model_name = model_name
        self.device: Optional[str] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        
        # determine device
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
            
            # STRICT TYPES: Use float16 for MPS/CUDA. Do NOT use 4-bit/8-bit on Mac.
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
            """
            FIXED & OPTIMIZED: Computes entropy on GPU and permutes BEFORE cpu conversion.
            """
            with torch.no_grad():
                # 1. Stack on device (T, B, V)
                logits = torch.stack(scores) 
                
                # 2. Compute Probabilities & Entropy on GPU
                logp = torch.log_softmax(logits, dim=-1)
                p = logp.exp()
                plogp = p * logp
                plogp = torch.nan_to_num(plogp, nan=0.0)
                
                # Reduce to scalar metrics (T, B) -> (B, T)
                ent_device = -torch.sum(plogp, dim=-1)
                ent_device = ent_device.permute(1, 0) # <--- Permute here (PyTorch)
                
                # TopK on device (T, B, K) -> (B, T, K)
                topk_vals_device = torch.topk(logp, 2, dim=-1).values
                topk_vals_device = topk_vals_device.permute(1, 0, 2) # <--- Permute here (PyTorch)

                # 3. Move ONLY the results to CPU
                ent = ent_device.cpu().float().numpy()
                topk = topk_vals_device.cpu().float().numpy()

            return ent, topk

    def _mock_solve(self, task: str, n_samples: int = 1) -> List[Trace]:
        mock_agent = CoTAgent(model="mock")
        return mock_agent._mock(task, n_samples)

    def solve(self, task: str, n_samples: int = 1) -> List[Trace]:
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
            "output_scores": True, # Required for measurement
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Handle Mechanistic Capture
        # NOTE: If ActivationMonitor is slow, this block is the cause. 
        monitor = ActivationMonitor(self.model) if ActivationMonitor else None
        
        try:
            if monitor: monitor.__enter__()
            with torch.no_grad():
                out = self.model.generate(**inputs, **generation_config)
        except Exception as e:
            print(f"Error during HFAgent generation: {e}. Falling back to mock.")
            return self._mock_solve(task, n_samples)
        finally:
            if monitor: monitor.__exit__(None, None, None)

        # Process Results
        seq = out.sequences[:, base_len:]
        texts = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        seq_len = seq.shape[1]
        
        # MEASUREMENT STEP (Now optimized)
        ent, topk = self._compute_batch_entropy(out.scores)
        
        # Retrieve activations (Ensure this doesn't leak memory)
        if monitor and hasattr(monitor, "get_batch_activations"):
            # This might be moving things to CPU, which is good
            acts_batch = monitor.get_batch_activations(expected_batch_size=n_samples, expected_seq_len=seq_len)
        else:
            acts_batch = None
            
        traces: List[Trace] = []
        for i in range(n_samples):
            toks = self.tokenizer.convert_ids_to_tokens(seq[i])
            T = min(len(toks), ent.shape[1])
            
            acts_i = None
            if acts_batch is not None and i < len(acts_batch) and acts_batch[i] is not None:
                # Ensure acts are numpy
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