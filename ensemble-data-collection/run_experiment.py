"""
run_experiment.py
Orchestrator: Phase 1 (Generation) -> Phase 2 (Grading).
"""
import os
import gc
import json
import torch
from tqdm import tqdm
from typing import Dict, List

# --- Component Imports ---
from uq_core import (
    MetricComputer,       
    OfflineAnalyzer,      
    calibrate_model,      
    set_entropy_indices,
)
from agent import HFAgent
from db_manager import DBManager

# --- UTILS ---
def load_gsm8k_sample(limit: int = 100) -> List[Dict[str, str]]:
    print(f"Loading GSM8K (limit={limit})...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    try:
        import requests
        lines = requests.get(url, timeout=10).text.strip().split("\n")
        return [
            {
                "id": str(i), 
                "q": json.loads(line)["question"], 
                "gold": json.loads(line)["answer"].split("####")[-1].strip().replace(",", "")
            } 
            for i, line in enumerate(lines) if i < limit
        ]
    except Exception as e:
        print(f"Dataset load failed ({e}). Using dummy.")
        return [{"id": "0", "q": "1+1?", "gold": "2"}] * limit

# --- PHASE 1: GENERATION ---
def run_generation_phase(cfg: Dict, db: DBManager):
    dataset = load_gsm8k_sample(limit=cfg.get("data_limit", 10))
    exp_id = db.new_experiment(cfg["experiment_name"], cfg)
    
    print("\n=== PHASE 1: GENERATION ===")
    
    for label, spec in cfg["models"].items():
        print(f"--- Model: {spec['id']} ---")
        
        # 1. Setup Logic (Calibration)
        mech_path = spec.get("mechanistic_config")
        if mech_path and not os.path.exists(mech_path):
             print(f"Calibrating entropy neurons -> {mech_path}")
             calibrate_model(spec["id"], mech_path)
        set_entropy_indices(mech_path)
            
        mod_id = db.register_model(spec["id"], mech_config=mech_path)
        agent = HFAgent(spec["id"]) 

        # 2. Generation Loop
        for row in tqdm(dataset, desc=f"Gen {label}"):
            try:
                # A. Generate (Heavy GPU Work)
                traces = agent.solve(row["q"], n_samples=cfg.get("n_samples", 1))
                if not traces: continue

                # B. Compute Metrics (CPU Work)
                # We collect all data for this question first
                batch_results = []
                for trace in traces:
                    vec = MetricComputer.compute_vector(trace)
                    pred = MetricComputer.extract_final_answer(trace.text)
                    is_exact = (pred == row["gold"])
                    
                    batch_results.append({
                        "pred": pred,
                        "trace_txt": trace.text,
                        "is_exact": is_exact,
                        "vec": vec
                    })

                # C. Batch Log
                # Use the helper method so it handles the new JSON trace columns automatically
                for res in batch_results:
                    db.log_result(
                        exp_id, 
                        mod_id, 
                        row,            # Passes dictionary with 'id', 'q', 'gold'
                        res['pred'], 
                        res['trace_txt'], 
                        res['is_exact'], 
                        res['vec']      # Passes the UncertaintyVector with traces
                    )
            
            except Exception as e:
                print(f"Error Q{row['id']}: {e}")

        # Cleanup Model
        del agent
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()

    return exp_id

# --- PHASE 2: GRADING ---
def run_grading_phase(cfg: Dict, db: DBManager, exp_id: int):
    print("\n=== PHASE 2: ANALYSIS & GRADING ===")
    analyzer = OfflineAnalyzer(cfg["db_path"])
    
    # 1. Semantic Entropy
    print("Computing Semantic Entropy...")
    analyzer.compute_semantic_entropy(exp_id)

    # 2. Grading (Local LLM)
    # The analyzer now handles the heavy lifting
    if "grader_model" in cfg:
        print(f"Loading Grader: {cfg['grader_model']}")
        try:
            grader_agent = HFAgent(cfg["grader_model"])
            analyzer.run_local_judge(exp_id, grader_agent)
            
            del grader_agent
            gc.collect()
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()
            
        except Exception as e:
            print(f"Grading failed: {e}")

    print(f"Experiment {exp_id} Complete.")

# --- MAIN ---
if __name__ == "__main__":
    cfg = {
        "experiment_name": "HUV_Benchmark_Optimized",
        "version": "5.0",
        "dataset": "gsm8k",
        "data_limit": 1000,     
        "n_samples": 4,        
        "db_path": "db/results.sqlite",
        "analyze_stages": True,
        
        # PHASE 1 CONFIG
        "models": {
            "Qwen-General": {
                "id": "Qwen/Qwen2.5-1.5B-Instruct",
                "mechanistic_config": "config/entropy_neurons_qwen_general.json"
            }
        },
                
        # PHASE 2 CONFIG
        "grader_model": "Qwen/Qwen2.5-3B-Instruct"
    }
    
    db = DBManager(cfg["db_path"])
    exp_id = run_generation_phase(cfg, db)
    run_grading_phase(cfg, db, exp_id)