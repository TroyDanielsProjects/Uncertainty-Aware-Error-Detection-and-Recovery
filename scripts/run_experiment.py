import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import asdict

try:
    from agents.cot_agent import CoTAgent
    from agents.hf_agent import HFAgent
    from data.data_utils import load_gsm8k
    from uq_framework.vector_computer import UncertaintyVectorComputer
    from uq_framework.uq_semantic import extract_final_answer, llm_judge_correctness
    from uq_framework.uq_mechanistic_interface import get_mechanistic_score
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def get_agent(cfg):
    m = cfg.get("model", "gpt-4o-mini").lower()
    if any(x in m for x in ["llama", "mistral", "gpt2", "/"]):
        return HFAgent(model_name=cfg["model"])
    return CoTAgent(model=cfg["model"])

def run_experiment(cfg):
    print(f"Experiment: {cfg.get('experiment_name','run')}")

    agent = get_agent(cfg)
    uq = UncertaintyVectorComputer()

    df = load_gsm8k(limit=cfg.get("data_limit", 100))
    print(f"Loaded {len(df)} samples.")

    results = []
    ns = cfg.get("n_samples", 3)
    out_dir = cfg.get("output_dir", "./output")
    os.makedirs(out_dir, exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            traces = agent.solve(row["question"], n_samples=ns)
        except Exception as e:
            print(f"Inference failed @ {i}: {e}")
            continue
        if not traces:
            continue

        t0 = traces[0]
        if i == 0:
            acts = t0.get("activations")
            if acts is not None and len(acts):
                score = get_mechanistic_score(acts)
                print(f"ID 0 activations: {np.array(acts).shape}, score: {score:.4f}")
            else:
                print("ID 0 missing activations.")

        U = uq.compute_vector_from_traces(traces)
        S = uq.compute_stage_vectors(t0)

        pred = extract_final_answer(t0.get("text"))
        gold = str(row.get("gold_answer","")).strip()

        try: correct = float(pred) == float(gold)
        except: correct = (pred == gold)

        if not correct and cfg.get("use_llm_judge", False):
            correct = llm_judge_correctness(row["question"], gold, t0.get("text"))

        results.append({
            "question_id": i,
            "question": row["question"],
            "gold_answer": gold,
            "predicted_answer": pred,
            "is_correct": int(correct),
            "U_vector": asdict(U) if U else None,
            "stage_vectors": S,
            "trace_text": t0.get("text")
        })

        if i % 10 == 0 and i > 0:
            _save(results, out_dir)

    _save(results, out_dir, final=True)

def _save(results, out, final=False):
    with open(os.path.join(out, "full_results.json"), "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)

    flat = []
    for r in results:
        U = r["U_vector"]
        if not U: continue
        base = {k: v for k, v in r.items() if k not in ["U_vector","stage_vectors","trace_text"]}
        base.update(U)
        flat.append(base)

    pd.DataFrame(flat).to_csv(os.path.join(out, "experiment_summary.csv"), index=False)
    if final:
        print(f"Results saved to {out}")

if __name__ == "__main__":
    cfg = {
        "experiment_name": "Llama3_Large_Run",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "dataset": "gsm8k",
        "data_limit": 5,
        "n_samples": 3,
        "use_llm_judge": True,
        "output_dir": "./output/llama3_large"
    }
    run_experiment(cfg)
