import argparse
import os
import logging
import sys
from dataclasses import dataclass
from typing import List
import math
import torch
from pathlib import Path
import json
from torch.utils.data import TensorDataset, DataLoader

from classifier.metric_helper import MetricHelper
from classifier.basic_classifier import BasicMLP

# python3 ./unified_pipeline/run_classifier.py --experiment_path unified_pipeline/results/gsm8k/unsloth/Meta-Llama-3.1-8B/entropic_logit_gap_mech_interp_heuristic/max_dev_var_10/prefill_included_True/gsm8k_llama_base_results.jsonl


@dataclass(frozen=True)
class ClassifierConfig:
    experiment_path: str
    classifier: str
    epochs: List[int]
    ablation: List[str]
    split: float
    balance_dataset: bool
    confidence_interval: bool
    confidence_repeats: int

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'tolist'): 
            return obj.tolist()
        if hasattr(obj, 'item'):   
            return obj.item()
        return super().default(obj)

# -------------------------
# Classifer Runner
# -------------------------

class ClassifierRunner:
    def __init__(self, config: ClassifierConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.metric_helper = MetricHelper()
        self.feature_groups = None
        self.confidence_repeats = self.config.confidence_repeats if self.config.confidence_interval else 1

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = 'cpu'
            
        self.logger.info(f"Using device: {self.device}")

    def run(self):
        df = self.prepare_data()
        ablation_results = self.run_ablation(df)
        self.save_results(ablation_results)

    def prepare_data(self):
        self.logger.info("Preparing data...")
        df = self.metric_helper.load_and_prep_data(self.config.experiment_path, self.config.ablation)
        
        if self.config.balance_dataset:
            df = self.metric_helper.balance_binary_dataset(df)
        
        df = self.metric_helper.normalize_data(df)
        
        # Determine feature groups for ablation
        self.feature_groups = self.metric_helper.get_feature_groups(df)
        return df
    
    def run_ablation(self, df):
        ablation_results = dict()
        
        # Iterate over high-level groups (e.g. "entropic+mechanistic")
        for group_name, features in self.feature_groups.items():

            self.logger.info(f"--- Starting classifier for group: {group_name} ---")
            
            ablation_results[group_name] = dict()

            # Get Numpy arrays
            X, y = self.metric_helper.convert_df_to_tensors(df, features)
            
            # Convert to Tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # [N, 1] for BCE

            # Split
            split_index = math.floor(len(X_tensor) * self.config.split)
            X_train, X_test = X_tensor[:split_index], X_tensor[split_index:]
            y_train, y_test = y_tensor[:split_index], y_tensor[split_index:]

            self.logger.info(f"Train split: {len(X_train)} | Test split: {len(X_test)}")

            if len(X_train) == 0 or len(X_test) == 0:
                self.logger.error("Data split resulted in empty set. Skipping.")
                continue

            # Create DataLoaders (Batching logic here)
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            
            # Drop last to prevent crash on size 1 batch with BatchNorm (if added later)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            input_dim = X_train.shape[1]

            for i in range(self.confidence_repeats):
                self.logger.info(f"Run {i+1}/{self.confidence_repeats}")
                
                model = BasicMLP(input_dim, device=self.device)
                results = model.train_classifer(train_loader, test_loader, self.config.epochs)
                
                ablation_results[group_name][f"run_{i}"] = results
                
        return ablation_results


    def save_results(self, results, new_root="unified_pipeline/classifier_results"):
            p = Path(self.config.experiment_path)
            
            # 1. Define filenames (keeps original stem, e.g., 'test1_results' -> 'test1_results_classification.json')
            output_filename_results = f"{p.stem}_classification.json"
            output_filename_config = f"{p.stem}_config.json"

            # 2. Reconstruct Directory Structure
            # p.parts[0] is the old root (e.g. "unified_pipeline")
            # p.parts[-1] is the filename
            # We take everything in between [1:-1] to mirror the middle folder structure
            remaining_parts = p.parts[1:-1]
            
            # Combine new root + mirrored middle parts
            output_dir = Path(new_root, *remaining_parts)
            os.makedirs(output_dir, exist_ok=True)

            # 3. Define Full Output Paths
            output_path_results = output_dir / output_filename_results
            output_path_config = output_dir / output_filename_config

            self.logger.info(f"Saving results to {output_path_results}")

            # 4. Write Files
            with open(output_path_results, "w") as f:
                f.write(json.dumps(results, cls=NumpyEncoder, indent=4))
                
            with open(output_path_config, "w") as f:
                # FIX: Use vars() to convert Dataclass -> Dict before dumping
                f.write(json.dumps(vars(self.config), cls=NumpyEncoder, indent=4))

# -------------------------
# CLI
# -------------------------

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def parse_args():
    parser = argparse.ArgumentParser("Uncertainty Quantification Classifier")

    parser.add_argument("--experiment_path", required=True)
    parser.add_argument("--classifier", default="basic", choices=["basic"])
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--balance_dataset", action="store_false") 
    parser.add_argument("--confidence_interval", action="store_false")
    parser.add_argument("--confidence_repeats", type=int, default=10)
    parser.add_argument(
        '--epochs',
        type=positive_int,
        default=[5, 10, 15, 20],
        nargs='+',
        help='Epoch checkpoints to record accuracy'
    )
    parser.add_argument(
        "--ablation",
        nargs="+",
        default=["entropic", "min_logit_gap", "mechanistic", "heuristic_score"],
        choices=["entropic", "min_logit_gap", "mechanistic", "heuristic_score", "semantic_entropy"],
    )
    parser.add_argument("--run_name", default="classifier_run")

    return parser.parse_args()


def setup_logging(run_name: str):
    os.makedirs("logs/experiments", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(f"logs/experiments/{run_name}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("uq_experiment")


def main():
    args = parse_args()
    logger = setup_logging(args.run_name)

    config = ClassifierConfig(
        experiment_path = args.experiment_path,
        classifier = args.classifier,
        epochs = args.epochs,
        ablation = args.ablation,
        split = args.split,
        balance_dataset = args.balance_dataset,
        confidence_interval = args.confidence_interval,
        confidence_repeats = args.confidence_repeats
    )

    runner = ClassifierRunner(config=config, logger=logger)
    runner.run()


if __name__ == "__main__":
    main()