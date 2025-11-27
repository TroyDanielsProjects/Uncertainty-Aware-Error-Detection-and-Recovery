
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import asdict
import traceback
import sys

# Ensure the project root is in the Python path for imports
# This allows running the script from the project root or the scripts directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary components from the framework
try:
    # Optional imports
    try:
        import torch
    except ImportError:
        print("Warning: PyTorch not found. HFAgent will be limited.")

    from agents.cot_agent import CoTAgent
    from agents.hf_agent import HFAgent
    from agents.base_agent import Trace
    from data.data_utils import load_gsm8k
    from uq_framework.vector_computer import UncertaintyVectorComputer
    from uq_framework.uq_semantic import extract_final_answer, llm_judge_correctness
    from uq_framework.uq_mechanistic_interface import load_indices
    # Import the new DB Manager
    from db.db_manager import DBManager
except ImportError as e:
    print(f"Critical Import Error: {e}")
    print("Please ensure the repository structure is correct and dependencies are installed.")
    exit(1)

class NumpyEncoder(json.JSONEncoder):
    """Handles serialization of numpy types and dataclasses."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return f"Numpy array (shape: {obj.shape})" # Avoid serializing large arrays directly
        if hasattr(obj, '__dict__'): return asdict(obj) # Handle dataclasses
        return super().default(obj)

def get_agent(model_id):
    """Factory function to initialize the correct agent type."""
    m = model_id.lower()
    # Heuristic check for local HF models vs API models
    if any(x in m for x in ["llama", "mistral", "gpt2", "deepseek", "/"]):
        # Assumes local HuggingFace model
        return HFAgent(model_name=model_id)
    # Assumes OpenAI compatible API
    return CoTAgent(model=model_id)


def save_trace(trace: Trace, path_prefix: str):
    """
    Saves the raw trace data. Handles large arrays (activations) separately as .npy.
    """
    try:
        data = asdict(trace)
        # Handle activations (if present)
        if 'activations' in data and data['activations'] is not None:
            # Save activations as a separate .npy file for efficiency
            act_path = path_prefix + '.npy'
            np.save(act_path, data['activations'])
            # Store the relative path to the npy file in the JSON
            data['activations'] = f"Stored externally at: {os.path.basename(act_path)}"
        
        json_path = path_prefix + '.json'
        with open(json_path, 'w') as f:
            # Use NumpyEncoder for the rest of the data
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        return json_path
    except Exception as e:
        print(f"Failed to save trace: {e}")
        return None

def run_experiment(cfg):
    print(f"Starting Experiment: {cfg.get('experiment_name','run')}")
    
    # --- Setup ---
    # Paths are relative to the project root
    db_path = os.path.join(project_root, cfg.get("db_path", "db/results.sqlite"))
    db = DBManager(db_path=db_path)
    
    out_dir = os.path.join(project_root, cfg.get("output_dir", "output"))
    exp_dir = os.path.join(out_dir, cfg['experiment_name'])
    trace_dir = os.path.join(exp_dir, "traces")
    os.makedirs(trace_dir, exist_ok=True)

    # Initialize DB Experiment
    experiment_id = db.create_experiment(
        cfg['experiment_name'],
        cfg.get('dataset', 'gsm8k'),
        cfg,
        repo_version=cfg.get("version", "1.0")
    )
    if experiment_id is None:
        print("Failed to create experiment in DB. Exiting.")
        return
        
    print(f"Database initialized. Experiment ID: {experiment_id}")

    # Initialize Models and Register in DB
    models_config = cfg.get("models", {})
    agents = {}
    model_db_ids = {}
    
    for label, model_spec in models_config.items():
        model_id_str = model_spec.get("id")
        print(f"Loading Agent: {label} ({model_id_str})...")
        try:
            agents[label] = get_agent(model_id_str)
            # Register model in DB
            model_db_ids[label] = db.register_model(
                model_id_str, 
                architecture=model_spec.get("architecture"),
                training_method=model_spec.get("training_method"),
                mech_config=model_spec.get("mechanistic_config")
            )
        except Exception as e:
            print(f"Error initializing agent {model_id_str} (see details above). Skipping this model.")
            # traceback.print_exc()
            
    if not agents:
        print("No models loaded successfully. Exiting.")
        return

    uq = UncertaintyVectorComputer()
    df = load_gsm8k(limit=cfg.get("data_limit", 100))
    
    ns = cfg.get("n_samples", 1) # Number of samples for semantic UQ
    results_list_csv = [] # For CSV summary output

    # --- Main Loop ---
    for i, row in tqdm(df.iterrows(), total=len(df)):
        question_data = {
            "id_external": str(i), # Using index as ID
            "question": row["question"],
            "gold_answer": str(row.get("gold_answer","")).strip(),
        }
        
        # Container for CSV output (summary view)
        csv_row = question_data.copy()

        # Run each model on the SAME question
        for label, agent in agents.items():
            if label not in model_db_ids: continue
            model_db_id = model_db_ids[label]
            
            # Set the correct mechanistic config for this model if specified
            model_config = models_config[label]
            if model_config.get("mechanistic_config"):
                config_path = os.path.join(project_root, model_config["mechanistic_config"])
                # This ensures the global indices used by get_mechanistic_score are correct for the current model
                load_indices(custom_path=config_path)

            try:
                # 1. Generate Traces
                traces = agent.solve(row["question"], n_samples=ns)
            except Exception as e:
                print(f"Error during generation for {label} @ Q{i}: {e}")
                traceback.print_exc()
                continue
            
            if not traces: continue

            t0 = traces[0] # Primary trace

            # 2. Compute HUV (Holistic Uncertainty Vector)
            U = uq.compute_vector_from_traces(traces)
            if U is None: continue

            # 3. Extract Answer and Grade
            pred = extract_final_answer(t0.text)
            
            # Use configured grading strategy
            if cfg.get("use_llm_judge", False):
                 # Pass the agent itself as the judge if configured, otherwise relies on OpenAI fallback if available
                 judge_agent = agent if cfg.get("use_self_judge", False) else None
                 is_correct = llm_judge_correctness(question_data['question'], question_data['gold_answer'], pred, agent=judge_agent)
            else:
                 # Basic check (numeric equivalence)
                 is_correct = llm_judge_correctness(question_data['question'], question_data['gold_answer'], pred, agent=None) 

            prediction_data = {
                "predicted_answer": pred,
                "is_correct": is_correct,
                "full_text": t0.text
            }

            # 4. Save Raw Trace
            trace_filename_prefix = f"E{experiment_id}_M{model_db_id}_Q{i}_{label}"
            trace_path = save_trace(t0, os.path.join(trace_dir, trace_filename_prefix))

            # 5. Log Results to DB
            result_id = db.log_result(
                experiment_id, model_db_id, question_data, prediction_data, U, trace_path
            )

            # 6. (Optional) Compute and Log Stage-Wise UQ
            if result_id and cfg.get("analyze_stages", True):
                stages = uq.compute_stage_vectors(t0)
                db.log_stage_results(result_id, stages)

            # 7. Update CSV Row Data
            prefix = f"{label}_"
            csv_row[f"{prefix}ans"] = pred
            csv_row[f"{prefix}correct"] = int(is_correct)
            csv_row[f"{prefix}mech_score"] = U.mechanistic_score
            csv_row[f"{prefix}avg_entropy"] = U.avg_entropy
            csv_row[f"{prefix}semantic_div"] = U.semantic_divergence
            csv_row[f"{prefix}heuristic_score"] = U.heuristic_score

        results_list_csv.append(csv_row)

        # Intermediate saving (CSV)
        if (i + 1) % 20 == 0:
            pd.DataFrame(results_list_csv).to_csv(os.path.join(exp_dir, f"summary_results.csv"), index=False)

    # Final saving (CSV)
    final_csv_path = os.path.join(exp_dir, f"summary_results_final.csv")
    pd.DataFrame(results_list_csv).to_csv(final_csv_path, index=False)
    print(f"\nExperiment finished.")
    print(f"Results saved to DB: {db_path} (ID: {experiment_id})")
    print(f"Summary CSV saved to: {final_csv_path}")

if __name__ == "__main__":
    # Example Configuration: Analyzing the RL vs SFT effect (Experiment 1 from the proposal)
    
    # Note: This configuration requires access to the specified models and sufficient hardware.
    # If running locally, ensure dependencies are installed (see requirements.txt).
    
    # IMPORTANT: You must calibrate the neurons for these models first using the calibration script
    # and ensure the paths in 'mechanistic_config' are correct.
    
    cfg = {
        "experiment_name": "HUV_Framework_Demo_SFT_vs_RL",
        "version": "1.1.0",
        "dataset": "gsm8k",
        "data_limit": 10,      # Small limit for demonstration
        "n_samples": 5,        # Increased samples for Semantic UQ analysis
        "output_dir": "output",
        "db_path": "db/results.sqlite",
        "analyze_stages": True,
        "use_llm_judge": False, # Set to True if OpenAI API key is available or self-judge is desired
        "use_self_judge": False,
        
        "models": {
            # Example using DeepSeek Math models (requires calibration)
            # HFAgent will automatically mock responses if models fail to load (e.g., insufficient GPU memory).
            "SFT_Model": {
                "id": "deepseek-ai/deepseek-math-7b-instruct",
                "architecture": "Llama",
                "training_method": "SFT",
                "mechanistic_config": "config/entropy_neurons_deepseek_sft.json"
            },
            "RL_Model": {
                "id": "deepseek-ai/deepseek-math-7b-rl",
                "architecture": "Llama",
                "training_method": "RL",
                "mechanistic_config": "config/entropy_neurons_deepseek_rl.json"
            }
            # Example using OpenAI model (requires API key, enable CLIENT in cot_agent.py/uq_semantic.py):
            # "GPT4oMini": {
            #     "id": "gpt-4o-mini",
            #     "architecture": "GPT",
            #     "training_method": "RLHF"
            # }
        }
    }
    
    print("\n--- HUV Framework Runner ---")
    print(f"Configuration loaded for: {cfg['experiment_name']}")
    print("To execute the experiment, uncomment the 'run_experiment(cfg)' line at the end of this script.")
    print("Ensure dependencies are installed (pip install -r requirements.txt)")
    run_experiment(cfg)
