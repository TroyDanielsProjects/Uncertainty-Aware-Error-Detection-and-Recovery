import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import List
import torch
from huggingface_hub import login

from data.dataloader import DataLoader
from data.preprocessor import Preprocessor
from model_setup.model_setup import ModelSetup


# -------------------------
# Configuration
# -------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    task: str
    model_name: str
    dtype: str
    uq_methods: List[str]
    preprocess: bool
    save_trace: bool
    run_name: str
    output_dir: str
    mech_interp_ident: List[str]
    entropy_neurons: int
    data_size: int
    semantic_runs: int
    include_prefill: bool
    explanation: bool


# -------------------------
# Experiment Runner
# -------------------------

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_dataset(self):
        self.logger.info(f"Loading dataset: {self.config.task}")

        if self.config.task == "gsm8k":
            return DataLoader.get_or_read_gsm8k()
        elif self.config.task == "cais_mmlu":
            return DataLoader.get_or_read_cais_mmlu()
        elif self.config.task == "mmlu_pro":
            return DataLoader.get_or_read_mmlu_pro()
        else:
            raise ValueError(f"Unsupported task: {self.config.task}")

    def load_model(self, device, dtype):
        self.logger.info(
            f"Loading model={self.config.model_name}, "
            f"dtype={self.config.dtype}, "
            f"uq_methods={self.config.uq_methods},"
        )

        return ModelSetup(
            model_name=self.config.model_name,
            task=self.config.task,
            dtype=dtype,
            uq_methods=self.config.uq_methods,
            mech_interp_ident_methods=self.config.mech_interp_ident,
            entropy_neurons=self.config.entropy_neurons,
            data_size=self.config.data_size,
            semantic_runs=self.config.semantic_runs,
            include_prefill=self.config.include_prefill,
            device=device,
            explanation=self.config.explanation,
        )

    def run(self):
        # Setup Paths
        # We split config and data because config doesn't change, but data grows.
        output_dir = os.path.join(
            self.config.output_dir,
            self.config.task,
            self.config.model_name,
            f"{'_'.join(self.config.uq_methods)}",
            f"{'_'.join(self.config.mech_interp_ident)}_{self.config.entropy_neurons}",
            f"prefill_included_{self.config.include_prefill}"
        )
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, f"{self.config.run_name}_config.json")
        data_path_results = os.path.join(output_dir, f"{self.config.run_name}_results.jsonl")
        data_path_trace = os.path.join(output_dir, f"{self.config.run_name}_trace.jsonl")

        # Save Config (Overwrite is fine here, it's small)
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=4)
        self.logger.info(f"Config saved to {config_path}")

        dataset = self.load_dataset()

        if self.config.preprocess:
            self.logger.info("Preprocessing dataset")
            dataset = Preprocessor.preprocess_data(dataset)

        # 3. Resume Logic: If current experiement exists, pick up from last generation
        processed_ids = set()
        if os.path.exists(data_path_results):
            self.logger.info(f"Found existing data at {data_path_results}. Scanning for completed IDs...")
            try:
                with open(data_path_results, "r") as f:
                    for line_num, line in enumerate(f):
                        if line.strip(): # Skip empty lines
                            try:
                                entry = json.loads(line)
                                # We treat ID as string to be safe against int/str mismatches
                                processed_ids.add(str(entry["id"]))
                            except json.JSONDecodeError:
                                self.logger.warning(f"Skipping corrupt line {line_num} in results file.")
                if len(processed_ids) > 0:
                    self.logger.info(f"Resuming run. Found {len(processed_ids)} completed examples.")
                else:
                    self.logger.info(f"Results file found but length is 0: {len(processed_ids)}. Restarting run")
                    processed_ids = None
                if not os.path.exists(data_path_trace): # NOTE - we may want to make this more strict and is can cause a mismatch
                    self.logger.warning(f"Results output file exists but trace output file doesn't. This may cause a mismatch in results")
                    with open(data_path_trace, 'w') as f:
                        pass
            except Exception as e:
                processed_ids = None
                self.logger.error(f"Error occured when trying to find existing run: {e}")
                with open(data_path_results, 'w') as f:
                    pass
                with open(data_path_trace, 'w') as f:
                    pass
        else:
            processed_ids = None
            self.logger.info(f"Existing run not found. Starting fresh run")
            with open(data_path_results, 'w') as f:
                pass
            with open(data_path_trace, 'w') as f:
                pass
        
        # determine correct device
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = 'cpu'

        dtype = getattr(torch, self.config.dtype)
        model = self.load_model(device, dtype)


        if self.config.save_trace:
            model.run(
                dataset,
                data_path_results,
                processed_ids, 
                save_trace=True, 
                output_path_trace=data_path_trace
            )
        else:
            model.run(dataset, data_path_results, processed_ids)



# -------------------------
# CLI
# -------------------------

# will constrain arguments to be positiive integers
def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue

def parse_args():
    parser = argparse.ArgumentParser("Uncertainty Quantification Experiment")

    parser.add_argument("--task", default="gsm8k", choices=["gsm8k", "cais_mmlu", "mmlu_pro"])
    parser.add_argument("--model", default="llama", choices=["llama", "qwen", "gemma"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--include_prefill", action="store_false")
    parser.add_argument("--save_trace", action="store_false")
    parser.add_argument("--explanation", action="store_true")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_dir", default="unified_pipeline/results")
    parser.add_argument("--entropy_neurons", type=positive_int, default=10)
    parser.add_argument("--data_size", type=positive_int, default=0)
    parser.add_argument("--semantic_runs", type=positive_int, default=10)
    parser.add_argument(
        "--uq_methods",
        nargs="+",
        default=["entropic", "logit_gap", "mech_interp", "heuristic"],
        choices=["entropic", "logit_gap", "mech_interp", "heuristic", "semantic"],
    )
    parser.add_argument(
        "--mech_interp_ident",
        nargs="+",
        default=["max_dev", "var"],
        choices=["max_dev", "var", "cos_sim", "cos_sim_w_weight"],
    )

    return parser.parse_args()


# python3 unified_pipeline/run_experiments.py --run_name test1

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

    if args.model == "llama":
        model_name = "unsloth/Meta-Llama-3.1-8B"
    elif args.model == "qwen":
        model_name = "Qwen/Qwen3-8B"
    elif args.model == "gemma":
        model_name = "google/gemma-2b"

    config = ExperimentConfig(
        task=args.task,
        model_name=model_name,
        dtype=args.dtype,
        uq_methods=args.uq_methods,
        preprocess=args.preprocess,
        save_trace=args.save_trace,
        run_name=args.run_name,
        output_dir=args.output_dir,
        mech_interp_ident=args.mech_interp_ident,
        entropy_neurons=args.entropy_neurons,
        data_size=args.data_size,
        semantic_runs=args.semantic_runs,
        include_prefill=args.include_prefill,
        explanation=args.explanation,
    )
    logger.info(f"Logging into Huggingface: {config.run_name}")
    login(token="") # I don't mind this for private repo, if made public needs to change

    logger.info(f"Starting experiment: {config.run_name}")
    runner = ExperimentRunner(config, logger)
    runner.run()


if __name__ == "__main__":
    main()
