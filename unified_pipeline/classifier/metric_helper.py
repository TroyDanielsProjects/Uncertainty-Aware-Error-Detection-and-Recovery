import pandas as pd
import numpy as np
import logging
import os
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

class MetricHelper:

    # Columns strictly for metadata tracking, not training
    meta_cols = ["id", "pred", "gold", "trace_txt", "semantic_text", "semantic"]
    label_col = "is_exact"

    def load_and_prep_data(self, filepath: str, fill_na: bool = True):
        """
        Loads JSONL, flattens the 'mechanistic' dictionary into columns, 
        and removes non-numeric metadata.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Could not find {filepath}")
            
        # 1. Load Data
        df = pd.read_json(filepath, lines=True)
        logger.info(f"Loaded {len(df)} rows from {filepath}")

        # 2. Flatten 'mechanistic' column (dict -> many columns)
        if "mechanistic" in df.columns:
            # Check if any row actually has data
            if df['mechanistic'].notna().any():
                logger.info("Flattening 'mechanistic' dictionary into features...")
                # Normalize flattens the dict keys into columns
                mech_df = pd.json_normalize(df['mechanistic'])
                
                # Make sure indices align before concat
                mech_df.index = df.index
                
                # Concatenate and drop original column
                df = pd.concat([df.drop(columns=['mechanistic']), mech_df], axis=1)
            else:
                logger.warning("'mechanistic' column exists but is empty/null.")
                df = df.drop(columns=['mechanistic'])
        
        # 3. Handle Missing Values
        if fill_na:
            # Fill numeric NaNs with 0 (assuming lack of activation = 0)
            # might want to use mean imputation depending on your theory
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
        # 4. Drop standard metadata columns if they exist
        existing_meta = [c for c in self.meta_cols if c in df.columns]
        if existing_meta:
            logger.info(f"Dropping metadata columns: {existing_meta}")
            df = df.drop(columns=existing_meta)

        # 5. Ensure Label is int (cast it to an int)
        if self.label_col in df.columns:
            df[self.label_col] = df[self.label_col].astype(int)

        logger.info(f"Final Data Shape: {df.shape}")
        return df

    def get_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Returns a dictionary separating feature names by type.
        Useful for ablation: 'run with just uncertainty', 'run with just neurons', etc.
        """
        all_cols = set(df.columns) - {self.label_col}
        
        # 1. Mechanistic: usually start with a digit (e.g. "10_mean") or "neuron"
        mech_feats = [c for c in all_cols if c[0].isdigit()]
        
        # 2. Uncertainty: specific keys we saved
        uncert_feats = [c for c in all_cols if c in ["entropic", "min_logit_gap", "heuristic_score"]]
        
        # 3. Anything else (catch-all)
        other_feats = [c for c in all_cols if c not in mech_feats and c not in uncert_feats]
        
        return {
            "all": list(all_cols),
            "mechanistic": mech_feats,
            "uncertainty": uncert_feats,
            "other": other_feats
        }

    @staticmethod
    def balance_binary_dataset(df: pd.DataFrame, label_col: str = "is_exact"):
        """
        Undersamples the majority class to create a 50/50 dataset.
        """
        if label_col not in df.columns:
            logger.warning(f"Label column {label_col} not found. Returning original DF.")
            return df

        true_df = df[df[label_col] == 1]
        false_df = df[df[label_col] == 0]
        
        n_true = len(true_df)
        n_false = len(false_df)
        
        if n_true == 0 or n_false == 0:
            logger.warning("One class is empty. Cannot balance.")
            return df

        logger.info(f"Balancing: True={n_true}, False={n_false}")
        
        min_count = min(n_true, n_false)
        
        true_balanced = true_df.sample(n=min_count, random_state=42)
        false_balanced = false_df.sample(n=min_count, random_state=42)
        
        balanced_df = pd.concat([true_balanced, false_balanced])
        # Shuffle
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return balanced_df

    def normalize_data(self, df: pd.DataFrame):
        """
        Z-score normalization. Returns a NEW dataframe.
        """
        df = df.copy()
        
        # Only normalize numeric columns, exclude label
        feature_cols = [c for c in df.columns if c != self.label_col and pd.api.types.is_numeric_dtype(df[c])]
        
        if not feature_cols:
            return df

        logger.info(f"Normalizing {len(feature_cols)} features...")
        
        features = df[feature_cols]
        # Replace 0 std with 1 to prevent NaN
        std = features.std().replace(0, 1)
        mean = features.mean()
        
        df[feature_cols] = (features - mean) / std
        
        return df
    
    def convert_df_to_numpy(self, df: pd.DataFrame, feature_subset: List[str] = None):
        """
        Converts DF to X, y numpy arrays.
        Args:
            feature_subset: If provided, only these columns are used for X.
                            If None, all columns except label are used.
        """
        y_values = df[self.label_col].to_numpy().astype(np.float32)
        
        if feature_subset is not None:
            # Ensure features exist
            valid_feats = [f for f in feature_subset if f in df.columns]
            if len(valid_feats) < len(feature_subset):
                missing = set(feature_subset) - set(valid_feats)
                logger.warning(f"Requested features missing from DF: {missing}")
            
            X_values = df[valid_feats].to_numpy().astype(np.float32)
        else:
            X_values = df.drop(columns=[self.label_col]).to_numpy().astype(np.float32)

        return X_values, y_values