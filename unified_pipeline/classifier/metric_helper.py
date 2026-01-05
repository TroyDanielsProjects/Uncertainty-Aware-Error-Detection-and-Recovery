import pandas as pd
import numpy as np
import logging
import os
from typing import List, Tuple, Dict, Optional
import itertools

logger = logging.getLogger(__name__)

class MetricHelper:

    meta_cols = ["id", "pred", "gold", "trace_txt"]
    label_col = "is_exact"

    def load_and_prep_data(self, filepath: str, metrics: List[str], fill_na: bool = True):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Could not find {filepath}")
            
        df = pd.read_json(filepath, lines=True)
        logger.info(f"Loaded {len(df)} rows from {filepath}")

        # Filter columns strictly to what was requested + label
        cols_to_keep = [self.label_col]
        for col in df.columns:
            if col in metrics:
                cols_to_keep.append(col)
        
        if len(cols_to_keep) == 1:
            logger.warning(f"Warning: Only label column found. Metrics {metrics} missing.")

        df = df[cols_to_keep]

        # Flatten 'mechanistic' if present
        if "mechanistic" in df.columns:
            if df['mechanistic'].notna().any():
                logger.info("Flattening 'mechanistic' dictionary...")
                mech_df = pd.json_normalize(df['mechanistic'])
                mech_df.index = df.index
                df = pd.concat([df.drop(columns=['mechanistic']), mech_df], axis=1)
            else:
                df = df.drop(columns=['mechanistic'])
        
        # Fill NaNs
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        if self.label_col in df.columns:
            df[self.label_col] = df[self.label_col].astype(int)

        logger.info(f"Data prepared. Shape: {df.shape}")
        return df

    def get_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Groups features by high-level type (e.g. 'mechanistic', 'entropic')
        and returns all combinations of these GROUPS.
        """
        all_cols = list(set(df.columns) - {self.label_col})
        
        # 1. Identify Groups based on column names
        groups = {}
        
        # Mechanistic columns usually start with digit (e.g. "10_mean")
        mech_cols = [c for c in all_cols if c[0].isdigit() or "neuron" in c.lower()]
        if mech_cols:
            groups["mechanistic"] = mech_cols
            
        # Scalar metrics
        scalars = ["entropic", "min_logit_gap", "heuristic_score", "semantic_entropy"]
        for s in scalars:
            # We check if the scalar is in columns OR if a version of it exists
            matches = [c for c in all_cols if s in c]
            if matches:
                groups[s] = matches

        # 2. Create Combinations of Groups
        # e.g. ("entropic"), ("entropic", "mechanistic")
        group_names = list(groups.keys())
        feature_combinations = {}
        
        for r in range(1, len(group_names) + 1):
            for combo in itertools.combinations(group_names, r):
                # Create a key like "entropic_mechanistic"
                combo_name = "_".join(combo)
                
                # Aggregate all columns for this combination
                combo_cols = []
                for group in combo:
                    combo_cols.extend(groups[group])
                
                feature_combinations[combo_name] = combo_cols
        
        logger.info(f"Generated {len(feature_combinations)} ablation groups.")
        return feature_combinations

    def balance_binary_dataset(self, df: pd.DataFrame, label_col: str = "is_exact"):
        if label_col not in df.columns:
            return df

        true_df = df[df[label_col] == 1]
        false_df = df[df[label_col] == 0]
        
        min_count = min(len(true_df), len(false_df))
        if min_count == 0:
            logger.warning("Cannot balance dataset: one class is empty.")
            return df
        
        logger.info(f"Balancing dataset to {min_count} samples per class.")
        true_balanced = true_df.sample(n=min_count, random_state=42)
        false_balanced = false_df.sample(n=min_count, random_state=42)
        
        balanced_df = pd.concat([true_balanced, false_balanced])
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def normalize_data(self, df: pd.DataFrame):
        df = df.copy()
        feature_cols = [c for c in df.columns if c != self.label_col and pd.api.types.is_numeric_dtype(df[c])]
        
        if not feature_cols:
            return df

        features = df[feature_cols]
        std = features.std().replace(0, 1)
        mean = features.mean()
        df[feature_cols] = (features - mean) / std
        return df
    
    def convert_df_to_tensors(self, df: pd.DataFrame, feature_subset: List[str] = None):
        y_values = df[self.label_col].to_numpy().astype(np.float32)
        
        if feature_subset is not None:
            valid_feats = [f for f in feature_subset if f in df.columns]
            if len(valid_feats) < len(feature_subset):
                logger.warning(f"Some requested features missing. Found {len(valid_feats)}/{len(feature_subset)}")
            X_values = df[valid_feats].to_numpy().astype(np.float32)
        else:
            X_values = df.drop(columns=[self.label_col]).to_numpy().astype(np.float32)

        return X_values, y_values