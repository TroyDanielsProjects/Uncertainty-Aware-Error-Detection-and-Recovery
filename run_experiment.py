"""
Main experiment runner for the consolidated uncertainty pipeline.

This script orchestrates the end‑to‑end process of loading a dataset,
invoking one or more agents to generate chain‑of‑thought responses,
computing holistic uncertainty vectors for each sample, and logging
results to a SQLite database.  It relies on the accompanying
``agent``, ``vector_computer`` and ``db_manager`` modules and can be
executed directly as a script.
"""

from __future__ import annotations

import os
import json
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from agent import HFAgent, CoTAgent, Trace, BaseAgent
from vector_computer import (
    UncertaintyVectorComputer,
    extract_final_answer,
    llm_judge_correctness,
    load_indices,
)
from db_manager import DBManager


def load_gsm8k(limit: int = 100) -> pd.DataFrame:
    """
    Load a subset of the GSM8K dataset.  Attempts to fetch the
    official JSONL file from GitHub.  If the request fails, returns a
    small mock dataset for demonstration.
    """
    print(f"Attempting to load GSM8K data (limit={limit})...")
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    try:
        import requests  # local import to avoid mandatory dependency

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().split("\n")
        records: List[Dict[str, Any]] = []
        count = 0
        for line in lines:
            if count >= limit:
                break
            item = json.loads(line)
            answer = item["answer"].split("####")[-1].strip().replace(",", "")
            records.append(
                {
                    "question": item["question"],
                    "gold_answer": answer,
                    "solution": item["answer"],
                }
            )
            count += 1
        print(f"Successfully loaded {len(records)} records.")
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Failed to load GSM8K data from web: {e}. Returning mock data.")
        dummy_data = [
            {
                "question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remaining eggs for $2 each. How much money does she make daily?",
                "gold_answer": "18",
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "gold_answer": "3",
            },
            {
                "question": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Berlin. If the total number of sheep is 390. How many sheep does Toulouse have?",
                "gold_answer": "240",
            },
        ]
        return pd.DataFrame(dummy_data * (limit // len(dummy_data) + 1))[:limit]


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that gracefully handles NumPy types and dataclasses
    when serialising traces.  Large arrays (activations) are
    substituted with a placeholder string rather than inlined into the
    JSON.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return f"Numpy array (shape: {obj.shape})"
        if hasattr(obj, "__dict__"):
            return asdict(obj)
        return super().default(obj)


def get_agent(model_id: str) -> BaseAgent:
    """
    Factory function to instantiate the appropriate agent type based on
    the model identifier.  Identifiers containing a slash ('/') or
    referencing known HF models will produce an ``HFAgent``; otherwise
    a ``CoTAgent`` is returned.
    """
    m = model_id.lower()
    if any(x in m for x in ["llama", "mistral", "gpt2", "deepseek", "/"]):
        return HFAgent(model_name=model_id)
    return CoTAgent(model=model_id)


def save_trace(trace: Trace, path_prefix: str) -> Optional[str]:
    """
    Persist a trace to disk.  Large activation arrays are saved as a
    separate ``.npy`` file and the JSON references that file by name.
    Returns the path of the JSON file or None on failure.
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
        print(f"Failed to save trace: {e}")
        return None


def run_experiment(cfg: Dict[str, Any]) -> None:
    """
    Execute an experiment with memory-optimized looping (Model -> Data).
    Prevents OOM/Swapping by ensuring only one agent is loaded at a time.
    """
    import gc
    import torch

    print(f"Starting Experiment: {cfg.get('experiment_name', 'run')}")
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Setup DB and Output Directories
    db_path = os.path.join(project_root, cfg.get("db_path", "db/results.sqlite"))
    db = DBManager(db_path=db_path)
    
    out_dir = os.path.join(project_root, cfg.get("output_dir", "output"))
    exp_dir = os.path.join(out_dir, cfg["experiment_name"])
    trace_dir = os.path.join(exp_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    # 2. Initialize Experiment Record
    experiment_id = db.create_experiment(
        cfg["experiment_name"],
        cfg.get("dataset", "gsm8k"),
        cfg,
        repo_version=cfg.get("version", "1.0"),
    )
    if experiment_id is None:
        print("Failed to create experiment in DB. Exiting.")
        return
    print(f"Database initialized. Experiment ID: {experiment_id}")

    # 3. Load Data & Initialize Results Container
    # We use a dictionary keyed by row_index to aggregate results across different models
    df = load_gsm8k(limit=cfg.get("data_limit", 100))
    results_map: Dict[int, Dict[str, Any]] = {}
    
    # Pre-fill static question data
    for i, row in df.iterrows():
        results_map[i] = {
            "id_external": str(i),
            "question": row["question"],
            "gold_answer": str(row.get("gold_answer", "")).strip(),
        }

    uq = UncertaintyVectorComputer()
    models_config = cfg.get("models", {})
    ns = cfg.get("n_samples", 1)

    # 4. Main Loop: Iterate Models First (Memory Optimization)
    for label, model_spec in models_config.items():
        print(f"\n=== Processing Model: {label} ===")
        model_id_str = model_spec.get("id")
        
        # Register Model in DB
        model_db_id = db.register_model(
            model_id_str,
            architecture=model_spec.get("architecture"),
            training_method=model_spec.get("training_method"),
            mech_config=model_spec.get("mechanistic_config"),
        )

        # Load Mechanistic Config if present
        mech_config = model_spec.get("mechanistic_config")
        if mech_config:
            load_indices(custom_path=os.path.join(project_root, mech_config))

        try:
            # Instantiate Agent (Loads weights into RAM/VRAM)
            # Note: Ensure HFAgent uses device="mps" inside its class defaults or passed here
            agent = get_agent(model_id_str) 
        except Exception as e:
            print(f"Failed to load agent {label}: {e}")
            continue

        # Iterate Data for this Agent
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {label}"):
            try:
                # Solve
                traces = agent.solve(row["question"], n_samples=ns)
                if not traces:
                    continue
                
                t0 = traces[0]
                U = uq.compute_vector_from_traces(traces)
                if U is None:
                    continue

                # Grade & Process
                pred = extract_final_answer(t0.text)
                is_correct = llm_judge_correctness(
                    row["question"], str(row.get("gold_answer", "")), pred
                )

                # Log to DB
                prediction_data = {
                    "predicted_answer": pred,
                    "is_correct": is_correct,
                    "full_text": t0.text,
                }
                trace_path = None
                # trace_path = save_trace(
                #     t0, os.path.join(trace_dir, f"E{experiment_id}_M{model_db_id}_Q{i}_{label}")
                # )
                result_id = db.log_result(
                    experiment_id, model_db_id, results_map[i], prediction_data, U, trace_path
                )
                
                if result_id and cfg.get("analyze_stages", True):
                    stages = uq.compute_stage_vectors(t0)
                    db.log_stage_results(result_id, stages)

                # Update CSV Data in Memory
                prefix = f"{label}_"
                results_map[i].update({
                    f"{prefix}ans": pred,
                    f"{prefix}correct": int(is_correct),
                    f"{prefix}mech_score": U.mechanistic_score,
                    f"{prefix}avg_entropy": U.avg_entropy,
                    f"{prefix}semantic_div": U.semantic_divergence,
                    f"{prefix}heuristic_score": U.heuristic_score
                })

                del traces, t0, U, pred
                
            except Exception as e:
                print(f"Error processing Q{i} for {label}: {e}")
                # traceback.print_exc() # Optional: uncomment for verbose debug

        # 5. CRITICAL: Unload Model & Clear Memory
        print(f"Unloading {label} to free memory...")
        del agent
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Intermediate CSV save (after every model finishes)
        pd.DataFrame(results_map.values()).to_csv(
            os.path.join(exp_dir, "summary_results_partial.csv"), index=False
        )

    # 6. Final Save
    final_csv_path = os.path.join(exp_dir, "summary_results_final.csv")
    pd.DataFrame(results_map.values()).to_csv(final_csv_path, index=False)
    
    print("\nExperiment finished.")
    print(f"Results saved to DB: {db_path} (ID: {experiment_id})")
    print(f"Summary CSV saved to: {final_csv_path}")

if __name__ == "__main__":
    cfg = {
        "experiment_name": "HUV_DeepSeek_Pipeline",
        "version": "1.0.0",
        "dataset": "gsm8k",
        "data_limit": 3,
        "n_samples": 2,
        "output_dir": "output",
        "db_path": "db/results.sqlite",
        "analyze_stages": True,
        "use_llm_judge": False,
        "use_self_judge": False,

        "models": {
            "DeepSeek_SFT": {
                "id": "deepseek-ai/deepseek-math-7b-instruct",
                "architecture": "Llama",
                "training_method": "SFT",
                "mechanistic_config": "config/entropy_neurons_deepseek_sft.json",
            },
            "DeepSeek_RL": {
                "id": "deepseek-ai/deepseek-math-7b-rl",
                "architecture": "Llama",
                "training_method": "RL",
                "mechanistic_config": "config/entropy_neurons_deepseek_rl.json",
            }
        }
    }

    print("\n--- HUV Framework Runner ---")
    print(f"Configuration loaded for: {cfg['experiment_name']}")
    run_experiment(cfg)
