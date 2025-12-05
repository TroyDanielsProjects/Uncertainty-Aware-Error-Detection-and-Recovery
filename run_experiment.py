"""
Main experiment runner for the consolidated uncertainty pipeline.
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
)
from db_manager import DBManager


def load_gsm8k_simple(limit: int = 100) -> List[Dict[str, str]]:
    """
    Load GSM8K data returning a simple list of dicts.
    """
    print(f"Attempting to load GSM8K data (limit={limit})...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    try:
        import requests
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().split("\n")
        records = []
        for i, line in enumerate(lines):
            if i >= limit:
                break
            item = json.loads(line)
            answer = item["answer"].split("####")[-1].strip().replace(",", "")
            records.append({
                "id": str(i),
                "question": item["question"],
                "gold_answer": answer,
                "solution": item["answer"]
            })
        print(f"Successfully loaded {len(records)} records.")
        return records
    except Exception as e:
        print(f"Failed to load GSM8K data: {e}. Returning mock data.")
        dummy_data = [
            {"id": "0", "question": "Janet has 16 eggs. She uses 4. How many left?", "gold_answer": "12"},
            {"id": "1", "question": "2 bolts blue, 1 bolt white. Total?", "gold_answer": "3"},
            {"id": "2", "question": "System A has 10. System B has 20. Total?", "gold_answer": "30"},
        ]
        return dummy_data[:limit]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return f"Array({obj.shape})"
        if hasattr(obj, "__dict__"): return asdict(obj)
        return super().default(obj)


def get_agent(model_id: str) -> BaseAgent:
    m = model_id.lower()
    if any(x in m for x in ["llama", "mistral", "gpt2", "deepseek", "/"]):
        return HFAgent(model_name=model_id)
    return CoTAgent(model=model_id)


def save_trace(trace: Trace, path_prefix: str) -> Optional[str]:
    """
    Persist full trace to disk (JSON + NPY). 
    Used as an artifact reference in the DB.
    """
    try:
        data = asdict(trace)
        if "activations" in data and data["activations"] is not None:
            act_path = path_prefix + ".npy"
            np.save(act_path, data["activations"])
            data["activations"] = f"Stored externally at: {os.path.basename(act_path)}"
        json_path = path_prefix + ".json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        return json_path
    except Exception as e:
        print(f"Failed to save trace file: {e}")
        return None


def judge_answer_with_agent(agent: BaseAgent, question: str, gold: str, pred: str) -> bool:
    """
    Determines correctness using a hybrid approach:
    1. Fast heuristic (exact string/float match).
    2. Slow agent-based judging (if heuristic fails).
    """
    # 1. Heuristic Check
    if gold.strip() == pred.strip():
        return True
    try:
        if abs(float(gold.replace(",", "")) - float(pred.replace(",", ""))) < 1e-5:
            return True
    except (ValueError, TypeError):
        pass

    # 2. Agent Check (The model judges itself)
    # We construct a prompt asking the model to verify equivalence.
    prompt = (
        f"You are a strict math grader.\n"
        f"Question: {question}\n"
        f"Gold Answer: {gold}\n"
        f"Student Answer: {pred}\n"
        f"Is the student answer mathematically equivalent to the gold answer? "
        f"Ignore formatting differences. Reply exactly 'YES' or 'NO'."
    )
    
    try:
        # Generate a single short response
        resp_traces = agent.solve(prompt, n_samples=1)
        if resp_traces:
            text = resp_traces[0].text.strip().upper()
            # Check if YES appears in the last few words (handling verbose chains)
            if "YES" in text or "CORRECT" in text:
                # Basic guard against "NOT CORRECT"
                if "NOT " not in text[-20:]: 
                    return True
    except Exception as e:
        print(f"Judging error: {e}")

    return False


def run_experiment(cfg: Dict[str, Any]) -> None:
    print(f"Starting Experiment: {cfg.get('experiment_name', 'run')}")
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Setup DB
    db_path = os.path.join(project_root, cfg.get("db_path", "db/results.sqlite"))
    db = DBManager(db_path=db_path)
    
    # 2. Setup Artifact Directory
    out_dir = os.path.join(project_root, cfg.get("output_dir", "output"))
    exp_dir = os.path.join(out_dir, cfg["experiment_name"])
    trace_dir = os.path.join(exp_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    # 3. Initialize Experiment
    experiment_id = db.create_experiment(
        cfg["experiment_name"],
        cfg.get("dataset", "gsm8k"),
        cfg,
        repo_version=cfg.get("version", "1.0"),
    )
    if not experiment_id:
        print("DB Error: Could not create experiment.")
        return

    # 4. Load Data
    dataset = load_gsm8k_simple(limit=cfg.get("data_limit", 100))
    uq_computer = UncertaintyVectorComputer()
    
    # 5. Main Loop
    for label, model_spec in cfg.get("models", {}).items():
        print(f"\n=== Processing Model: {label} ===")
        model_id_str = model_spec.get("id")
        
        # Register Model
        model_db_id = db.register_model(
            model_id_str,
            architecture=model_spec.get("architecture"),
            training_method=model_spec.get("training_method"),
            mech_config=model_spec.get("mechanistic_config"),
        )

        # Load Mechanistic Config
        mech_config = model_spec.get("mechanistic_config")
        if mech_config:
            load_indices(custom_path=os.path.join(project_root, mech_config))

        try:
            agent = get_agent(model_id_str) 
        except Exception as e:
            print(f"Failed to load agent {label}: {e}")
            continue

        for row in tqdm(dataset, desc=f"Running {label}"):
            try:
                # Generate Solution
                traces = agent.solve(row["question"], n_samples=cfg.get("n_samples", 1))
                if not traces: continue
                
                t0 = traces[0] 
                U = uq_computer.compute_vector_from_traces(traces)
                if U is None: continue

                pred = extract_final_answer(t0.text)
                
                # --- JUDGING STEP ---
                # Use the active agent to judge if simple matching fails
                is_correct = judge_answer_with_agent(
                    agent, row["question"], str(row.get("gold_answer", "")), pred
                )

                # Log Result
                prediction_data = {
                    "predicted_answer": pred,
                    "is_correct": is_correct,
                    "full_text": t0.text,
                }
                
                trace_path = save_trace(
                   t0, os.path.join(trace_dir, f"E{experiment_id}_M{model_db_id}_Q{row['id']}_{label}")
                )

                result_id = db.log_result(
                    experiment_id, model_db_id, 
                    {"id_external": row['id'], "question": row['question'], "gold_answer": row['gold_answer']}, 
                    prediction_data, U, trace_path
                )
                
                if result_id:
                    # Log Stages (Optional)
                    if cfg.get("analyze_stages", True):
                        stages = uq_computer.compute_stage_vectors(t0)
                        db.log_stage_results(result_id, stages)
                    
                    # Log Token Metrics
                    token_metrics = uq_computer.extract_token_metrics(t0)
                    db.log_token_metrics(result_id, token_metrics)

            except Exception as e:
                print(f"Error processing Q{row['id']} for {label}: {e}")

        # Cleanup
        print(f"Unloading {label}...")
        del agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"\nExperiment finished. DB: {db_path}")

if __name__ == "__main__":
    cfg = {
        "experiment_name": "HUV_DeepSeek_AgentJudge",
        "version": "1.3.1",
        "dataset": "gsm8k",
        "data_limit": 3,
        "n_samples": 3,
        "db_path": "db/results.sqlite",
        "analyze_stages": True,
        "models": {
            "DeepSeek_SFT": {
                "id": "deepseek-ai/deepseek-math-7b-instruct",
                "mechanistic_config": "config/entropy_neurons_deepseek_sft.json",
            }
        }
    }
    run_experiment(cfg)