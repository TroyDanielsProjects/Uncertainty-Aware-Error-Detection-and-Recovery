from typing import List, Dict
from datasets import load_dataset
import logging
import json
import os

logger = logging.getLogger(__name__)

class DataLoader:

    def get_gsm8k(self,
        split: str = "train",
        limit: int | None = None,
    ) -> List[Dict[str, str]]:
        """
        Load GSM8K from Hugging Face datasets.

        Args:
            split: "train" or "test"
            limit: Optional cap on number of examples.

        Returns:
            List of dicts with keys: id, q, gold
        """
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split: {split}")

        logger.info(f"Loading GSM8K split='{split}', limit={limit}")

        dataset = load_dataset("gsm8k", "main", split=split)

        if limit is not None:
            dataset = dataset.select(range(limit))

        samples = []
        for i, ex in enumerate(dataset):
            # GSM8K answer format: "... #### 42"
            try:
                gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            except Exception as e:
                raise ValueError(f"Malformed answer at index {i}") from e

            samples.append({
                "id": str(i),
                "q": ex["question"],
                "gold": gold,
            })

        return samples
    
    @classmethod
    def save_dict_dataset(self, data_dictionary, path_to_write):
        with open(path_to_write, "w") as json_file:
            # Use json.dump() to write the dictionary to the file
            json.dump(data_dictionary, json_file, indent=4) # indent=4 makes the file human-readable

        logger.info(f"Dictionary successfully saved to {path_to_write}")

    def get_or_read_gsm8k(self, path_to_data):

        if os.path.isfile(path_to_data):
            pass