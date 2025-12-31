from typing import List, Dict, Optional
from datasets import load_dataset
import logging
import json
import os
import torch 

logger = logging.getLogger(__name__)

class DataLoader:

    @classmethod
    def get_gsm8k(cls,
        split: str = "train",
        limit: Optional[int] = None,
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
            error_msg = f"Invalid split provided: {split}. Must be 'train' or 'test'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Downloading/Loading GSM8K from Hugging Face (split='{split}', limit={limit})...")

        try:
            # Load the dataset from HF Hub
            dataset = load_dataset("gsm8k", "main", split=split)
        except Exception as e:
            logger.critical(f"Failed to load dataset from Hugging Face: {e}")
            raise e

        # Apply limit if specified
        if limit is not None:
            logger.info(f"Limiting dataset to first {limit} examples.")
            dataset = dataset.select(range(limit))

        samples = []
        for i, ex in enumerate(dataset):
            # GSM8K answer format typically ends with: "... #### 42"
            try:
                # We split by the delimiter and take the last part
                gold = ex["answer"].split("####")[-1].strip().replace(",", "")
                
                # Sanity check: ensure the split actually resulted in a specific answer
                if len(gold) == 0:
                    logger.warning(f"Parsed empty gold answer at index {i}")

            except Exception as e:
                logger.error(f"Malformed answer format at index {i}: {e}")
                raise ValueError(f"Malformed answer at index {i}") from e

            samples.append({
                "id": str(i),
                "q": ex["question"],
                "gold": gold,
            })

        logger.info(f"Successfully processed {len(samples)} samples.")
        return samples
    
    @classmethod
    def save_dict_dataset(cls, data_dictionary: List[Dict], path_to_write: str):
        """
        Saves a list of dictionaries to a JSON file.
        Changed 'self' to 'cls' to adhere to @classmethod convention.
        """
        logger.info(f"Saving dataset to {path_to_write}...")
        try:
            with open(path_to_write, "w") as json_file:
                # Use json.dump() to write the dictionary to the file
                json.dump(data_dictionary, json_file, indent=4) 
            logger.info(f"Dataset successfully saved to {path_to_write}")
        except IOError as e:
            logger.error(f"Failed to write to file {path_to_write}: {e}")
            raise e

    @classmethod
    def get_or_read_gsm8k(cls, path_to_data: str = "./unified_pipeline/data/datasets/gsm8k.json", save_if_absent: bool = True):
        """
        Checks if a local JSON file exists; if so, loads it. 
        If not, downloads via get_gsm8k and saves it.
        
        Args:
            path_to_data: Local file path.
            save_if_absent: Boolean flag to save data locally after download.
            **kwargs: Arguments passed to get_gsm8k (e.g., split="test", limit=100).
        """
        if os.path.isfile(path_to_data):
            logger.info(f"Local file found at {path_to_data}. Loading...")
            try:
                # FIX: Must use json.load because the file was saved with json.dump
                with open(path_to_data, 'r') as f:
                    dataset = json.load(f)
                logger.info(f"Loaded {len(dataset)} samples from local file.")
                return dataset
            except json.JSONDecodeError as e:
                logger.error(f"File exists but is not valid JSON: {e}")
                raise e
        else:
            logger.info(f"No local file found at {path_to_data}. Fetching from source...")
            
            # Pass kwargs (split, limit) to the loader
            dataset = cls.get_gsm8k()
            
            if save_if_absent:
                cls.save_dict_dataset(dataset, path_to_data)
            
            return dataset