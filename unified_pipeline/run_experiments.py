import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import List

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
    run_name: str
    output_dir: str


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

    def load_model(self):
        self.logger.info(
            f"Loading model={self.config.model_name}, "
            f"dtype={self.config.dtype}, "
            f"uq_methods={self.config.uq_methods}"
        )

        return ModelSetup(
            model_name=self.config.model_name,
            dtype=self.config.dtype,
            uq_methods=self.config.uq_methods,
        )

    def run(self):
        dataset = self.load_dataset()

        if self.config.preprocess:
            self.logger.info("Preprocessing dataset")
            dataset = Preprocessor.preprocess_data(dataset)

        model = self.load_model()

        self.logger.info("Running experiment")
        results = model.run(dataset)

        self.save_results(results)

    def save_results(self, results):
        output_dir = os.path.join(
                self.config.output_dir,
                self.config.task,
                self.config.model_name,
            )
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{self.config.run_name}.json")

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

def parse_args():
    parser = argparse.ArgumentParser("Uncertainty Quantification Experiment")

    parser.add_argument("--task", default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--model", default="llama_3.1_8B", choices=["llama_3.1_8B"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument(
        "--uq_methods",
        nargs="+",
        default=["entropic", "logit_gap", "mech_interp", "heuristic"],
        choices=["entropic", "logit_gap", "mech_interp", "heuristic", "semantic"],
    )
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_dir", default="results")

    return parser.parse_args()


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
        model_name=args.model_name,
        dtype=args.dtype,
        uq_methods=args.uq_methods,
        preprocess=args.preprocess,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )

    logger.info(f"Starting experiment: {config.run_name}")
    runner = ExperimentRunner(config, logger)
    runner.run()


if __name__ == "__main__":
    main()
