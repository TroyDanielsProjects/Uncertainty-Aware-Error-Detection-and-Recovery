"""
Main experiment runner: Phase 1 (Generation) ONLY.
"""
from __future__ import annotations
import os
import json
import gc
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from dataclasses import asdict
from typing import Dict, List

from agent import HFAgent, CoTAgent, Trace, BaseAgent
from vector_computer import UncertaintyVectorComputer, extract_final_answer, load_indices
from db_manager import DBManager

def load_gsm8k_simple(limit: int = 100):
    """Loads GSM8K data."""
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
                "gold_answer": answer
            })
        return records
    except Exception as e:
        print(f"Warning: GSM8K Load Failed ({e}). Using dummy.")
        return [{"id": "0", "question": "1+1?", "gold_answer": "2"}] * limit

def compute_semantic_entropy_local(traces: List[Trace]) -> float:
    """Computes SE on the in-memory batch."""
    answers = [extract_final_answer(t.text) for t in traces]
    answers = [a for a in answers if a]
    if not answers: return 0.0
    counts = Counter(answers)
    total = len(answers)
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log(probs + 1e-10)))

def save_trace(trace: Trace, path_prefix: str) -> str:
    data = asdict(trace)
    if data.get("activations") is not None:
        np.save(path_prefix + ".npy", data["activations"])
        data["activations"] = f"Stored: {os.path.basename(path_prefix)}.npy"
    with open(path_prefix + ".json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path_prefix + ".json"

def run_experiment(cfg: Dict):
    print("=== PHASE 1: GENERATION ===")
    db = DBManager(cfg["db_path"])
    dataset = load_gsm8k_simple(limit=cfg["data_limit"])
    uq_computer = UncertaintyVectorComputer()
    
    exp_id = db.create_experiment(cfg["experiment_name"], cfg.get("dataset", "gsm8k"), cfg)
    trace_dir = os.path.join("output", cfg["experiment_name"], "traces")
    os.makedirs(trace_dir, exist_ok=True)

    for label, spec in cfg["models"].items():
        print(f"--- Model: {label} ---")
        mech_config = spec.get("mechanistic_config")
        if mech_config: load_indices(custom_path=mech_config)
        
        model_db_id = db.register_model(spec["id"], mech_config=mech_config)
        
        try:
            # We assume HFAgent here. 
            # If using CoT API, ensure config reflects that.
            agent = HFAgent(spec["id"]) if "/" in spec["id"] else CoTAgent(spec["id"])
        except Exception as e:
            print(f"Skipping {label}: {e}")
            continue

        for row in tqdm(dataset, desc=f"Gen {label}"):
            try:
                # 1. Generate Batch
                traces = agent.solve(row["question"], n_samples=cfg.get("n_samples", 1))
                if not traces: continue

                # 2. Compute Batch Metrics
                se = 0.0
                if len(traces) > 1:
                    se = compute_semantic_entropy_local(traces)

                for i, trace in enumerate(traces):
                    # 3. Vector Calculation
                    U = uq_computer.compute_vector(trace)
                    U.semantic_entropy = se # Inject batch metric

                    # 4. Fast Check (Exact Match)
                    pred = extract_final_answer(trace.text)
                    gold = row["gold_answer"]
                    is_exact = False
                    if pred and gold:
                        try:
                            if pred == gold or abs(float(pred.replace(",","")) - float(gold.replace(",",""))) < 1e-5:
                                is_exact = True
                        except: pass

                    # 5. Log
                    p_data = {
                        "predicted_answer": pred,
                        "is_correct": is_exact, 
                        "full_text": trace.text,
                        "eval_method": "Exact Match" if is_exact else "Pending"
                    }
                    
                    path = save_trace(trace, os.path.join(trace_dir, f"E{exp_id}_Q{row['id']}_S{i}"))
                    rid = db.log_result(exp_id, model_db_id, {"id_external": row['id'], "question": row['question'], "gold_answer": gold}, p_data, U, path, sample_index=i)
                    
                    if rid and cfg.get("analyze_stages"):
                        db.log_stage_results(rid, uq_computer.compute_stage_vectors(trace))
                        db.log_token_metrics(rid, uq_computer.extract_token_metrics(trace))

            except Exception as e:
                print(f"Error Q{row['id']}: {e}")

        # Cleanup
        del agent
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    print(f"Generation Complete. Exp ID: {exp_id}")
    print("Run 'python offline_computer.py' to perform grading.")

if __name__ == "__main__":
    cfg = {
        "experiment_name": "Generation_Only_Run",
        "db_path": "db/results.sqlite",
        "dataset": "gsm8k",
        "data_limit": 20, 
        "n_samples": 5, 
        "models": {
            "Llama-3-8B": {
                "id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "mechanistic_config": "config/entropy_neurons_llama3.json"
            }
        },
        "analyze_stages": True
    }
    run_experiment(cfg)