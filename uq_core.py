"""
uq_core.py
Core logic for Uncertainty Quantification.
Centralizes: Calibration, Activation Monitoring, Metric Computation (Semantic & Mechanic), and Grading.
"""
from __future__ import annotations
import os
import re
import json
import sqlite3
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from collections import Counter
from tqdm import tqdm

# --- 1. Heuristic Logic (Semantic Embeddings) ---
_HAS_ST = False
_MODEL = None
_U_VEC = None
_C_VEC = None

try:
    from sentence_transformers import SentenceTransformer, util
    # Load lightweight model (~80MB on first run)
    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    
    _U_WORDS = ["maybe", "perhaps", "unlikely", "doubtful", "unclear", "possible", "guess", "assume", "speculate"]
    _C_WORDS = ["definitely", "certainly", "proven", "obvious", "undeniable", "fact", "true", "guaranteed"]
    
    # Pre-compute anchor vectors
    _ANCHORS = _MODEL.encode([" ".join(_U_WORDS), " ".join(_C_WORDS)])
    _U_VEC, _C_VEC = _ANCHORS[0], _ANCHORS[1]
    _HAS_ST = True
except ImportError:
    print("Info: sentence-transformers not found. Using keyword fallback.")
except Exception as e:
    print(f"Warning: Embeddings failed ({e}). Using fallback.")

# --- 2. Mechanistic Configuration ---
_ENTROPY_INDICES: Optional[np.ndarray] = None

def set_entropy_indices(indices: Union[List[int], np.ndarray, str]) -> None:
    """Load neuron indices from list, array, or filepath."""
    global _ENTROPY_INDICES
    if isinstance(indices, str):
        if os.path.exists(indices):
            with open(indices, "r") as f: indices = json.load(f)
        else:
            indices = []
    _ENTROPY_INDICES = np.array(indices, dtype=int)

# --- 3. Data Structures ---
@dataclass
class Trace:
    text: str
    tokens: List[str] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    top1_logprobs: List[float] = field(default_factory=list)
    top2_logprobs: List[float] = field(default_factory=list)
    activations: Optional[np.ndarray] = None # Aligned [Seq, Dim]
    
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

@dataclass
class UncertaintyVector:
    avg_entropy: float
    min_logit_gap: float
    heuristic_score: float
    mechanistic_score: float
    semantic_entropy: float = 0.0

# --- 4. Inference Monitoring ---
class ActivationMonitor:
    """Context manager to capture and align hidden states from the last MLP layer."""
    def __init__(self, model):
        self.model = model
        self.activations = []
        self.hook = None
        self.target_layer = self._locate_layer()

    def _locate_layer(self):
        # Llama/Mistral/Qwen
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[-1].mlp.act_fn 
        # GPT-2/Neo/J
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[-1].mlp 
        return None

    def _hook(self, _, __, output):
        # OPTIMIZATION: Keep on device to avoid CPU sync bottleneck during generation
        t = output[0] if isinstance(output, tuple) else output
        self.activations.append(t.detach()) 

    def __enter__(self):
        if self.target_layer: self.hook = self.target_layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self.hook: self.hook.remove()

    def get_aligned_activations(self, prompt_len: int, gen_len: int) -> Optional[np.ndarray]:
        if not self.activations: return None
        try:
            # Normalize shapes: [B, S, D] (prefill) vs [B, D] (decode) -> [B, Total, D]
            # Processing happens here (post-generation) to minimize latency impact
            stack = torch.cat([t if t.ndim == 3 else t.unsqueeze(1) for t in self.activations], dim=1)
            
            # Align: Gen token i comes from Prompt token i-1
            start = max(0, prompt_len - 1)
            end = start + gen_len
            
            # Slice first, THEN move to CPU to save bandwidth
            if stack.shape[1] < end: 
                relevant = stack[:, -gen_len:, :]
            else:
                relevant = stack[:, start:end, :]
                
            return relevant.cpu().numpy()
        except Exception: return None

# --- 5. Metrics & Scoring ---
class MetricComputer:
    @staticmethod
    def extract_final_answer(text: str) -> str:
        if not isinstance(text, str): return ""
        if "####" in text: return text.split("####")[-1].strip()
        if m := re.search(r"\\boxed\\{([^}]+)\\}", text): return m.group(1).strip()
        if m := re.search(r"(?:answer|result) is\s*([-+]?\d+(?:[.,]\d+)?)", text, re.I): 
            return m.group(1).replace(",", "").strip()
        if nums := re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?", text): 
            return nums[-1].replace(",", "").strip()
        return ""

    @staticmethod
    def calculate_heuristic_score(text: str, use_embeddings: bool = False) -> float:
        """
        Computes confidence based on Semantic Embeddings (if enabled) or Keyword Fallback.
        use_embeddings: Set to False during fast generation loop.
        """
        if not text: return 0.0
        
        # 1. Semantic Embedding Logic (Preferred, but slow)
        if use_embeddings and _HAS_ST and _MODEL is not None:
            try:
                emb = _MODEL.encode(text)
                sim_u = float(util.cos_sim(emb, _U_VEC).item())
                sim_c = float(util.cos_sim(emb, _C_VEC).item())
                # Result is relative closeness to Certainty vs Uncertainty
                # Range approx -1 to 1. Normalize to 0-1.
                raw_diff = sim_c - sim_u
                return max(0.0, min(1.0, (raw_diff + 0.5))) 
            except Exception:
                pass # Fall through to keywords on error

        # 2. Keyword Fallback (Fast)
        hedges = {"might", "perhaps", "possibly", "unclear", "maybe", "assume", "unlikely"}
        words = text.lower().split()
        if not words: return 0.0
        ratio = sum(1 for w in words if w in hedges) / (len(words) + 1)
        return max(0.0, 1.0 - (ratio * 5)) # Penalize hedges

    @staticmethod
    def logit_gap(t1: List[float], t2: List[float]) -> np.ndarray:
        a1, a2 = np.array(t1), np.array(t2)
        if a1.size != a2.size or not a1.size: return np.array([])
        return a1 - np.where(np.isinf(a2), -1e6, a2)

    @staticmethod
    def mechanistic_score(acts: Optional[np.ndarray]) -> float:
        if acts is None or _ENTROPY_INDICES is None or not _ENTROPY_INDICES.size: return 0.0
        try:
            valid = _ENTROPY_INDICES[_ENTROPY_INDICES < acts.shape[-1]]
            if not valid.size: return 0.0
            return float(np.mean(np.tanh(np.mean(acts[..., valid], axis=-1) * 0.5)))
        except Exception: return 0.0

    @classmethod
    def compute_vector(cls, trace: Trace) -> UncertaintyVector:
        gap = cls.logit_gap(trace.top1_logprobs, trace.top2_logprobs)
        return UncertaintyVector(
            avg_entropy=float(np.mean(trace.entropies)) if trace.entropies else 0.0,
            min_logit_gap=float(np.min(gap)) if gap.size else 0.0,
            # Disable embeddings for fast generation
            heuristic_score=cls.calculate_heuristic_score(trace.text, use_embeddings=True), 
            mechanistic_score=cls.mechanistic_score(trace.activations)
        )

# --- 6. Offline Analysis (Batch & Grading) ---
class OfflineAnalyzer:
    def __init__(self, db_path: str, openai_client=None):
        self.db_path = db_path
        self.client = openai_client

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def compute_semantic_entropy(self, experiment_id: int):
        """Groups traces by question, clusters answers, computes Shannon entropy."""
        with self._get_conn() as conn:
            df = pd.read_sql("SELECT result_id, question_id_external, full_trace_text FROM Results WHERE experiment_id = ?", 
                             conn, params=(experiment_id,))
            if df.empty: return

            df['ans'] = df['full_trace_text'].apply(MetricComputer.extract_final_answer)
            updates = []
            
            for _, group in df.groupby('question_id_external'):
                n = len(group)
                if n <= 1: se = 0.0
                else:
                    probs = np.array(list(Counter(group['ans']).values())) / n
                    se = -np.sum(probs * np.log(probs + 1e-10))
                updates.extend([(float(se), rid) for rid in group['result_id']])
            
            conn.executemany("UPDATE Results SET uq_semantic_entropy = ? WHERE result_id = ?", updates)
            conn.commit()

    def run_local_judge(self, experiment_id: int, agent: Any):
        """Uses a local Agent to verify mathematical equivalence."""
        print(f"Local Judge ({getattr(agent, 'model_name', 'Agent')}): Grading...")
        with self._get_conn() as conn:
             rows = conn.execute(
                "SELECT result_id, question_text, gold_answer, predicted_answer, full_trace_text FROM Results WHERE experiment_id = ? AND eval_method != 'Exact Match'", 
                (experiment_id,)
            ).fetchall()
        
        updates = []
        for r in tqdm(rows, desc="Grading"):
            prompt = (
                f"You are a strict math grader.\nQuestion: {r['question_text']}\nGold: {r['gold_answer']}\n"
                f"Student: {r['predicted_answer']}\nTrace: \"\"\"{r['full_trace_text'][-800:]}\"\"\"\n"
                f"Does Student match Gold? Reply YES or NO."
            )
            try:
                resp_traces = agent.solve(prompt, n_samples=1)
                clean = resp_traces[0].text.strip().upper().replace(".", "") if resp_traces else ""
                is_correct = ("YES" in clean and "NO" not in clean) or clean.startswith("YES")
                updates.append((1 if is_correct else 0, clean, f"Local_Judge", r['result_id']))
            except Exception: pass

        if updates:
            with self._get_conn() as conn:
                conn.executemany("UPDATE Results SET is_correct = ?, gpt_eval_reason = ?, eval_method = ? WHERE result_id = ?", updates)
                conn.commit()
        print(f"Graded {len(updates)} items.")

    def run_deep_judge(self, experiment_id: int, model="gpt-4o"):
        """Uses OpenAI API to verify answers."""
        if not self.client: return
        with self._get_conn() as conn:
            df = pd.read_sql("SELECT result_id, question_text, gold_answer, full_trace_text, predicted_answer FROM Results WHERE experiment_id = ? AND eval_method != 'Exact Match'", 
                             conn, params=(experiment_id,))
            updates = []
            print(f"Deep Judge (OpenAI): Grading {len(df)} items...")
            for _, row in tqdm(df.iterrows(), total=len(df)):
                prompt = (f"Q: {row['question_text']}\nGold: {row['gold_answer']}\nStudent: {row['predicted_answer']}\n"
                          "Is Student equivalent to Gold? JSON: {is_correct: bool, reason: str}")
                try:
                    res = self.client.chat.completions.create(
                        model=model, messages=[{"role": "user", "content": prompt}], 
                        response_format={"type": "json_object"}, temperature=0
                    )
                    data = json.loads(res.choices[0].message.content)
                    updates.append((1 if data.get('is_correct') else 0, data.get('reason', ''), 'OpenAI', row['result_id']))
                except Exception: pass
            
            with self._get_conn() as conn:
                conn.executemany("UPDATE Results SET is_correct = ?, gpt_eval_reason = ?, eval_method = ? WHERE result_id = ?", updates)
                conn.commit()

# --- 7. Calibration Setup ---
def calibrate_model(model_name: str, save_path: str, k: int = 5):
    """Setup function to identify entropy neurons."""
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", trust_remote_code=True)
        # Extract last MLP layer weights
        if hasattr(model, 'model'): layer = model.model.layers[-1] # Llama
        else: layer = model.transformer.h[-1] # GPT
        
        Wout = layer.mlp.down_proj.weight.float().detach()
        WU = model.lm_head.weight.float().detach()
        
        # Compute Variance of Logits (L = WU @ Wout)
        print("Computing projection...")
        L = WU @ Wout
        vars_ = torch.var(L, dim=0)
        indices = torch.topk(vars_, k=k, largest=False).indices

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f: json.dump(indices.tolist(), f)
        print(f"Calibration complete. Saved to {save_path}")
    except Exception as e:
        print(f"Calibration failed: {e}")