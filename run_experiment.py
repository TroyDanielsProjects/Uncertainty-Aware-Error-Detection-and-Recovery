"""
Main experiment runner: Generation (Phase 1) -> Grading (Phase 2).
"""

from __future__ import annotations

import os
import json
import gc
import torch
import numpy as np
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from agent import HFAgent, CoTAgent, Trace, BaseAgent
from vector_computer import (
    UncertaintyVectorComputer,
    extract_final_answer,
    load_indices,
    compute_semantic_entropy 
)
from db_manager import DBManager

# --- UTILS ---

def load_gsm8k_simple(limit: int = 100) -> List[Dict[str, str]]:
    print(f"Loading GSM8K (limit={limit})...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    try:
        import requests
        response = requests.get(url, timeout=10)
        lines = response.text.strip().split("\n")
        records = []
        for i, line in enumerate(lines):
            if i >= limit: break
            item = json.loads(line)
            answer = item["answer"].split("####")[-1].strip().replace(",", "")
            records.append({
                "id": str(i),
                "question": item["question"],
                "gold_answer": answer,
                "solution": item["answer"]
            })
        return records
    except Exception:
        print("Warning: Failed to load GSM8K. Using dummy data.")
        return [{"id": "0", "question": "1+1?", "gold_answer": "2", "solution": "2"}] * limit

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return f"Array({obj.shape})"
        return super().default(obj)

def save_trace(trace: Trace, path_prefix: str) -> str:
    data = asdict(trace)
    if "activations" in data and data["activations"] is not None:
        np.save(path_prefix + ".npy", data["activations"])
        data["activations"] = f"Stored: {os.path.basename(path_prefix)}.npy"
    with open(path_prefix + ".json", "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    return path_prefix + ".json"

def get_agent(model_id: str) -> BaseAgent:
    if any(x in model_id.lower() for x in ["llama", "mistral", "deepseek", "qwen", "/"]):
        return HFAgent(model_name=model_id)
    return CoTAgent(model=model_id)

# --- PHASE 1: GENERATION ---

def run_generation_phase(cfg: Dict, db: DBManager):
    print("\n=== PHASE 1: GENERATION ===")
    dataset = load_gsm8k_simple(limit=cfg.get("data_limit", 10))
    uq_computer = UncertaintyVectorComputer()
    
    experiment_id = db.create_experiment(
        cfg["experiment_name"], cfg.get("dataset", "gsm8k"), cfg, cfg.get("version", "1.0")
    )
    
    trace_dir = os.path.join("output", cfg["experiment_name"], "traces")
    os.makedirs(trace_dir, exist_ok=True)

    for label, spec in cfg["models"].items():
        print(f"Loading Generator: {spec['id']}")
        
        # Calibration Check
        mech_config = spec.get("mechanistic_config")
        if mech_config:
            if not os.path.exists(mech_config):
                print(f"WARNING: Mechanistic config file not found at: {mech_config}")
                print("Mechanistic scores will be 0.0. Did you run calibrate_entropy_neurons.py?")
            load_indices(custom_path=mech_config)
            
        model_db_id = db.register_model(spec["id"], mech_config=mech_config)
        
        try:
            agent = get_agent(spec["id"])
        except Exception as e:
            print(f"Skipping {label}: {e}")
            continue

        for row in tqdm(dataset, desc=f"Generating {label}"):
            try:
                # 1. Generate Batch of Traces (Capture Activations = True)
                traces = agent.solve(row["question"], n_samples=cfg.get("n_samples", 1), capture_activations=True)
                if not traces: continue

                # 2. Compute Batch-Level Semantic Entropy
                batch_semantic_entropy = 0.0
                if len(traces) > 1:
                    batch_semantic_entropy = compute_semantic_entropy(traces)

                for i, trace in enumerate(traces):
                    # 3. Compute Per-Trace Metrics
                    U = uq_computer.compute_vector_from_traces([trace])
                    if not U: continue
                    
                    # 4. Inject the Batch-Level Semantic Entropy
                    U.semantic_entropy = batch_semantic_entropy

                    # 5. Heuristic Check (Fast Exact Match)
                    pred = extract_final_answer(trace.text)
                    gold = row["gold_answer"]
                    
                    is_exact = False
                    try:
                        is_exact = (pred.strip() == gold.strip()) or \
                                   (abs(float(pred.replace(",","")) - float(gold.replace(",",""))) < 1e-5)
                    except: pass

                    # 6. Log
                    p_data = {
                        "predicted_answer": pred,
                        "is_correct": True if is_exact else False,
                        "reason": "Exact Match" if is_exact else "Pending Grading",
                        "eval_method": "Exact Match" if is_exact else "Pending",
                        "full_text": trace.text
                    }
                    
                    fname = f"E{experiment_id}_M{model_db_id}_Q{row['id']}_{label}_S{i}"
                    path = save_trace(trace, os.path.join(trace_dir, fname))

                    rid = db.log_result(
                        experiment_id, model_db_id, 
                        {"id_external": row['id'], "question": row['question'], "gold_answer": gold},
                        p_data, U, path, sample_index=i
                    )

                    if rid:
                        if cfg.get("analyze_stages"):
                            db.log_stage_results(rid, uq_computer.compute_stage_vectors(trace))
                        db.log_token_metrics(rid, uq_computer.extract_token_metrics(trace))

            except Exception as e:
                print(f"Error Q{row['id']}: {e}")

        # Cleanup Generator
        print(f"Unloading {label}...")
        del agent
        gc.collect()
        
        # CORRECT CLEANUP FOR MAC (MPS) & CUDA
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    return experiment_id

# --- PHASE 2: GRADING ---

def run_grading_phase(cfg: Dict, db: DBManager, exp_id: int):
    print("\n=== PHASE 2: GRADING (Local LLM) ===")
    
    grader_id = cfg.get("grader_model", "meta-llama/Llama-3.2-3B-Instruct")
    print(f"Loading Grader: {grader_id} ...")
    
    try:
        grader = get_agent(grader_id)
    except Exception as e:
        print(f"Failed to load grader: {e}")
        return

    rows = db.get_ungraded_results(exp_id)
    to_grade = [r for r in rows if r[5] != "Exact Match"] 
    
    print(f"Found {len(rows)} results. {len(rows) - len(to_grade)} Exact Matches. Grading {len(to_grade)} items...")

    for r in tqdm(to_grade, desc="Grading"):
        rid, q_text, gold, pred, trace_text, _ = r
        
        # --- IMPROVED PROMPT: Strict YES/NO ---
        prompt = (
            f"You are a strict math grader.\n"
            f"Question: {q_text}\n"
            f"Gold Answer: {gold}\n"
            f"Student Prediction: {pred}\n\n"
            f"The student's full reasoning trace is below:\n"
            f"\"\"\"{trace_text[-2000:]}\"\"\"\n\n"
            f"Does the student's prediction (or trace) MATCH the Gold Answer mathematically? "
            f"Ignore formatting differences (e.g. '$5' vs '5'). "
            f"If the reasoning is wrong or the answer is incorrect, say NO.\n"
            f"Reply with exactly one word: YES or NO."
        )
        
        try:
            # Generate minimal tokens (max_new_tokens=5) to force brevity
            # We assume the agent.solve() uses the generation config from HFAgent
            # Note: You might need to tweak max_new_tokens in agent.py if it's hardcoded to 512, 
            # but usually the model stops after the strict prompt.
            resp_traces = grader.solve(prompt, n_samples=1, capture_activations=False)
            raw_text = resp_traces[0].text if resp_traces else ""
            
            # --- ROBUST PARSING ---
            # 1. Clean up (remove punctuation, whitespace, newlines)
            clean_text = raw_text.strip().upper().replace(".", "")
            
            is_correct = False
            
            # 2. Strict Check
            if clean_text.startswith("YES"):
                is_correct = True
            elif clean_text.startswith("NO"):
                is_correct = False
            else:
                # Fallback: Check if YES appears without a preceding "NOT"
                # This handles verbose outputs like "YES, the answer is..."
                if "YES" in clean_text and "NO" not in clean_text:
                     is_correct = True
                else:
                     is_correct = False
            
            # Save the raw text as the reason so you can audit it later
            db.update_grading(rid, is_correct, raw_text, f"Model: {grader_id}")
            
        except Exception as e:
            print(f"Grading failed for ID {rid}: {e}")

    print("Grading Complete.")

# --- MAIN ---

if __name__ == "__main__":
    cfg = {
        "experiment_name": "HUV_Benchmark_MacOptimized",
        "version": "4.1",
        "dataset": "gsm8k",
        "data_limit": 50,     # Adjust to 500 when confident
        "n_samples": 3,        
        "db_path": "db/results.sqlite",
        "analyze_stages": True,
        
        # PHASE 1 CONFIG
        "models": {
            "Qwen2.5-Math-1.5B": {
                "id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
                # Ensure this matches what you used in calibration!
                "mechanistic_config": "config/entropy_neurons_qwen_math_1.5b.json",
            }
        },
        
        # PHASE 2 CONFIG
        # Using Llama-3.2-3B (Fits in Mac Memory)
        "grader_model": "meta-llama/Llama-3.2-3B-Instruct" 
    }
    
    db = DBManager(cfg["db_path"])
    exp_id = run_generation_phase(cfg, db)
    run_grading_phase(cfg, db, exp_id)