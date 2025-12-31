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
from models.model_setup import ModelSetup


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
            device=device
        )

    def run(self):
        dataset = self.load_dataset()

        if self.config.preprocess:
            self.logger.info("Preprocessing dataset")
            dataset = Preprocessor.preprocess_data(dataset)
        
        # determine correct device
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = 'cpu'

        dtype = getattr(torch, self.config.dtype)
        model = self.load_model(device, dtype)

        output_dir = os.path.join(
            self.config.output_dir,
            self.config.task,
            self.config.model_name,
            f"{"_".join(self.config.uq_methods)}",
            f"{"_".join(self.config.mech_interp_ident)}_{self.config.entropy_neurons}",
        )

        os.makedirs(output_dir, exist_ok=True)

        self.logger.info("Running experiment")

        if self.config.save_trace:
            results = model.run(
                dataset, 
                save_trace=True, 
                output_path=os.path.join(output_dir, f"{self.config.run_name}_trace.json")
            )
        else:
            results = model.run(dataset)

        self.save_results(output_dir, results)

    def save_results(self, output_dir, results):

        output_path = os.path.join(output_dir, f"{self.config.run_name}_results.json")

        payload = {
            "config": vars(self.config),
            "results": results,
        }

        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

        self.logger.info(f"Saved results to {output_path}")


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

    parser.add_argument("--task", default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B", choices=["unsloth/Meta-Llama-3.1-8B"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--save_trace", action="store_false")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_dir", default="unified_pipeline/results")
    parser.add_argument("--entropy_neurons", type=positive_int, default=10)
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
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(f"logs/{run_name}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger("uq_experiment")


def main():
    args = parse_args()
    logger = setup_logging(args.run_name)

    config = ExperimentConfig(
        task=args.task,
        model_name=args.model,
        dtype=args.dtype,
        uq_methods=args.uq_methods,
        preprocess=args.preprocess,
        save_trace=args.save_trace,
        run_name=args.run_name,
        output_dir=args.output_dir,
        mech_interp_ident=args.mech_interp_ident,
        entropy_neurons=args.entropy_neurons
    )
    logger.info(f"Logging into Huggingface: {config.run_name}")
    login(token="") # I don't mind this for private repo, if made public needs to change

    logger.info(f"Starting experiment: {config.run_name}")
    runner = ExperimentRunner(config, logger)
    runner.run()


if __name__ == "__main__":
    main()
