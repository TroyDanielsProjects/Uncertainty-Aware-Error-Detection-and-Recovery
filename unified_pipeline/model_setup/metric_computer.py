import re
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from collections import Counter
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# --- 1. Heuristic Logic (Semantic Embeddings) ---
_HAS_ST = False
_MODEL = None
_U_VEC = None
_C_VEC = None

try:
    from sentence_transformers import SentenceTransformer, util
    # Load lightweight model
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

class MetricComputer:
    
    @staticmethod
    def extract_answer_gsm8k(text: str, prompt_to_remove: str = None) -> str:
        """
        Extracts the answer from GSM8K examples or model output.
        """
        if not text: return ""

        # 1. Strip Prompt (Safety Step)
        if prompt_to_remove and text.startswith(prompt_to_remove):
            text = text[len(prompt_to_remove):]

        # 1. Gold Standard: GSM8K dataset specific delimiter (usually for ground truth)
        if '####' in text:
            return text.split('####')[-1].strip().replace(',', '')
        
        # 2. Model Prompt Format: ***Answer: [answer]***
        match = re.search(r'\*\*\*Answer:\s*(.*?)\*\*\*', text)
        if match:
            extracted = match.group(1).strip() # FIX: group(1), not group(2)
            
            # OPTIONAL BUT RECOMMENDED:
            # Even though we captured the specific format, the model might have 
            # sneaked in "5 apples". Let's run the number extractor on this specific substring.
            # If that fails, return the raw string (it might be a text answer).
            clean_extracted = extracted.replace(',', '').replace('$', '')
            numbers = re.findall(r'-?\d+(?:\.\d+)?', clean_extracted)
            if numbers:
                return numbers[-1] # Return the number found inside the tags
            return extracted # Return the raw text inside tags if no number found
            
        # 3. Fallback: Find the last number in the entire text
        # Remove common currency/formatting chars to simplify regex
        clean_text = text.replace(',', '').replace('$', '')
        
        # Find all integers or floats
        numbers = re.findall(r'-?\d+(?:\.\d+)?', clean_text)
        
        if numbers:
            return numbers[-1]
            
        return ""
    
    @staticmethod
    def extract_answer_mmlu(text: str, prompt_to_remove: str = None) -> str:
        """
        Extracts the multiple-choice answer (A, B, C, D) from MMLU model output.
        """
        if not text: return ""

        # 1. Strip Prompt (Safety Step)
        if prompt_to_remove and text.startswith(prompt_to_remove):
            text = text[len(prompt_to_remove):]

        # 2. Strict Format: ***Answer:[Letter]***
        # We explicitly look for A, B, C, or D to avoid capturing garbage.
        match = re.search(r'\*\*\*Answer:\s*([A-D])\s*\*\*\*', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 3. Fallback: Look for common variations
        # Examples it catches: "Answer: A", "The answer is (B)", "Option C"
        # We search for the *last* occurrence, as the chain of thought might discuss wrong answers first.
        patterns = [
            r'Answer:\s*([A-D])',        # "Answer: A"
            r'Option\s*([A-D])',         # "Option A"
            r'\b([A-D])\)',              # "A)" or "B)"
            r'\(([A-D])\)'               # "(A)" or "(B)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].upper() # Return the last found match
        
        return ""
    
    @staticmethod
    def extract_answer_mmlu_pro(text: str, prompt_to_remove: str = None) -> str:
        if not text: return ""

        if prompt_to_remove and text.startswith(prompt_to_remove):
            text = text[len(prompt_to_remove):]

        # Allow A-J (case insensitive)
        match = re.search(r'\*\*\*Answer:\s*([A-J])\s*\*\*\*', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Fallback patterns for A-J
        patterns = [
            r'Answer:\s*([A-J])',
            r'Option\s*([A-J])',
            r'\b([A-J])\)',
            r'\(([A-J])\)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
        
        return ""

    @staticmethod
    def logit_gap(t1: List[float], t2: List[float]) -> np.ndarray:
        a1, a2 = np.array(t1), np.array(t2)
        if a1.size != a2.size or not a1.size: return np.array([])
        return a1 - np.where(np.isinf(a2), -1e6, a2)
    
    @staticmethod
    def check_correctness(generated_answer: str, ground_truth: str) -> bool:
        """
        Comparing numbers can be tricky (e.g., 42.0 vs 42).
        We attempt to convert to float for comparison.
        """
        try:
            # Simple string match first
            if generated_answer.strip() == ground_truth.strip():
                return True
                
            # Float comparison
            gen_float = float(generated_answer)
            truth_float = float(ground_truth)
            return abs(gen_float - truth_float) < 1e-6
        except ValueError:
            return False
        
    @staticmethod 
    def aggregate_activations(activation_dict: dict, stats: list = ['min', 'max', 'mean', 'std']) -> dict:
        """
        Aggregates raw token-level activations into summary statistics per neuron.
        Handles JSON serialization issues (numpy types) and NaN values.
        """
        # 1. Debug Empty Input
        if not activation_dict:
            logger.warning("aggregate_activations received None or empty dict. Check ModelSetup recorder.")
            return {}

        try:
            # 2. Convert dictionary to DataFrame
            # Note: This raises ValueError if lists in activation_dict have different lengths
            df = pd.DataFrame(activation_dict)
            
            if df.empty:
                logger.warning("Activation DataFrame is empty (no tokens recorded?).")
                return {}
            
            # 3. Calculate statistics
            # Result: Index=[min, max...], Columns=[neuron_id_1, neuron_id_2...]
            agg_df = df.agg(stats)
            
            flat_results = {}
            
            # 4. Flatten and Sanitize
            for neuron_id in agg_df.columns:
                for stat in stats:
                    try:
                        val = agg_df.at[stat, neuron_id]
                        
                        # JSON SAFETY: Convert numpy/pandas types to Python native
                        if pd.isna(val) or pd.isnull(val):
                            val = None # JSON null
                        elif hasattr(val, 'item'):
                            val = val.item() # Convert numpy.float32 -> float
                        elif isinstance(val, (np.float32, np.float64)):
                            val = float(val)

                        flat_results[f"{neuron_id}_{stat}"] = val
                    except Exception as inner_e:
                        logger.warning(f"Could not extract {stat} for {neuron_id}: {inner_e}")
                    
            return flat_results

        except ValueError as ve:
            # This specific error happens if your neuron lists are different lengths
            logger.error(f"Shape Mismatch in Activations (ValueError): {ve}")
            return {}
        except Exception as e:
            logger.error(f"Critical failure in aggregation: {e}")
            return {}
        
    @staticmethod
    def logit_gap(t1: List[float], t2: List[float]) -> np.ndarray:
        a1, a2 = np.array(t1), np.array(t2)
        if a1.size != a2.size or not a1.size: return np.array([])
        return a1 - np.where(np.isinf(a2), -1e6, a2)
    
    @staticmethod
    def calculate_heuristic_score(text: str, use_embeddings: bool = False) -> float:
        """
        Computes confidence based on Semantic Embeddings (if enabled) or Keyword Fallback.
        use_embeddings: Set to False during fast generation loop.
        """
        if not text: return 0.0
        
        # 1. Heuristic Score Via Semantic Embedding Logic
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

        # 2. Keyword Fallback (unused)
        hedges = {"might", "perhaps", "possibly", "unclear", "maybe", "assume", "unlikely"}
        words = text.lower().split()
        if not words: return 0.0
        ratio = sum(1 for w in words if w in hedges) / (len(words) + 1)
        return max(0.0, 1.0 - (ratio * 5)) # Penalize hedges
    
    @staticmethod
    def calculate_semantic_entropy(generated_texts: List[str]) -> float:
        """
        Computes semantic entropy over a list of sampled generated texts.
        1. Extracts answers from each text (e.g., '5', '5', '6').
        2. Clusters identical answers.
        3. Computes Shannon entropy over the answer distribution.
        """
        if not generated_texts:
            return None
        
        # 1. Extract answers from all samples
        # We reuse your existing extraction logic to normalize answers (e.g. "5.0" -> "5")
        answers = [MetricComputer.extract_answer_gsm8k(text) for text in generated_texts]
        
        n = len(answers)
        if n == 0: return 0.0
        
        # 2. Count frequencies of each unique answer
        counts = Counter(answers)
        probs = np.array(list(counts.values())) / n
        
        # 3. Compute Shannon Entropy: -sum(p * log(p))
        # Add epsilon 1e-10 to avoid log(0) error
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return float(entropy)
