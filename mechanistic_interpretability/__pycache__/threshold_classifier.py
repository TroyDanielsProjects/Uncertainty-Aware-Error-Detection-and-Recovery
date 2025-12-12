import torch
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import itertools

# --- 0. Logging Setup ---
class Logger:
    def __init__(self, filename="classifer_experiment_logger.txt"):
        self.filename = filename
        # Clear file if it exists, or create new
        with open(self.filename, 'w') as f:
            f.write(f"Experiment Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")

    def log(self, message):
        print(message)  # Print to console
        with open(self.filename, 'a') as f: # Append to file
            f.write(str(message) + "\n")

# Initialize Global Logger
logger = Logger()

# --- 1. Helper Functions (Data Loading & Engineering) ---

def load_and_prep_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")
    df = pd.read_csv(filepath)
    if df['is_correct'].dtype == object:
         df['is_correct'] = df['is_correct'].astype(str).str.lower() == 'true'
    neuron_cols = [c for c in df.columns if c.startswith('Neuron_')]
    return df, neuron_cols

def aggregate_df(df, neuron_cols, stats=['mean']):
    """
    Baseline Method: Calculates specific stats for every neuron.
    stats: list of strings, e.g., ['min', 'max']
    """
    # Group and calculate the requested statistics
    X = df.groupby(['problem_id', 'is_correct'])[neuron_cols].agg(stats)
    
    # Flatten MultiIndex columns (e.g., neuron_0_min, neuron_0_max)
    X.columns = [f'{col}_{stat}' for col, stat in X.columns]
    
    X = X.reset_index()
    return X

def threshold_activations(calibration_df, target_df):
    """New Method: Binary flags based on calibration thresholds."""
    neuron_cols = [c for c in calibration_df.columns if c.startswith('Neuron_')]
    
    # Calculate Thresholds (Q3 + 1.5*IQR) on Correct Calibration examples
    correct_calib = calibration_df[calibration_df['is_correct'] == True][neuron_cols]
    if correct_calib.empty:
         raise ValueError("No correct examples in calibration data to set thresholds!")

    Q1 = correct_calib.quantile(0.25)
    Q3 = correct_calib.quantile(0.75)
    IQR = Q3 - Q1
    thresholds = Q3 + (1.5 * IQR)
    
    # Apply to Target
    activations_mask = target_df[neuron_cols] > thresholds
    activations_binary = activations_mask.astype(int)
    activations_binary['problem_id'] = target_df['problem_id']
    
    # Aggregate (Max = did it spike at least once?)
    aggregated_df = activations_binary.groupby('problem_id').max()
    
    # Re-attach labels
    labels = target_df.groupby('problem_id')['is_correct'].first()
    aggregated_df['is_correct'] = labels
    return aggregated_df.reset_index()

def balance_binary_dataset(df):
    true_df = df[df['is_correct'] == True]
    false_df = df[df['is_correct'] == False]
    min_count = min(len(true_df), len(false_df))
    
    logger.log(f"Balancing: True={len(true_df)}, False={len(false_df)} -> Target={min_count}")
    
    true_balanced = true_df.sample(n=min_count, random_state=42)
    false_balanced = false_df.sample(n=min_count, random_state=42)
    
    balanced_df = pd.concat([true_balanced, false_balanced])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

def normalize_data(df):
    meta_cols = ['problem_id', 'is_correct']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    features = df[feature_cols]
    # Handle std=0 to avoid NaNs
    std = features.std().replace(0, 1)
    df[feature_cols] = (features - features.mean()) / std
    return df

def convert_df_to_dataloader(X):
    drop_cols = ['problem_id', 'is_correct']
    features_df = X.drop(columns=drop_cols, errors='ignore')
    X_tensor = torch.tensor(features_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(X['is_correct'].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset, X_tensor.shape[1]

# --- 2. The Model ---

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x)

# --- 3. The Experiment Runner ---

def run_experiment(full_dataset, name="Experiment", n_runs=10, epochs=100):
    logger.log(f"\n{'='*20} Running: {name} {'='*20}")
    
    # Fixed Data Split
    total_count = len(full_dataset)
    test_count = 100
    train_count = total_count - test_count
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_count, test_count], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    logger.log(f"Data Split -> Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    input_dim = full_dataset.tensors[0].shape[1]
    
    accuracies = []
    
    for run in range(n_runs):
        model = SimpleMLP(input_dim, hidden_dim=64).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                predicted_probs = torch.sigmoid(logits)
                predictions = (predicted_probs > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)
        
        acc = correct / total
        accuracies.append(acc)

    # Stats
    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)
    
    t_score = 2.262 # 95% Conf for N=10
    margin_of_error = t_score * (std_acc / np.sqrt(n_runs))
    logger.log(f"Mean: {mean_acc:.4f} | Std: {std_acc:.4f} | CI: [{mean_acc - margin_of_error:.4f}, {mean_acc + margin_of_error:.4f}]")
    
    return mean_acc, std_acc, margin_of_error, accuracies

# --- 4. Main Execution ---

if __name__ == "__main__":
    try:
        logger.log("Loading raw data...")
        calib_df, neuron_cols = load_and_prep_data("./gsm8k_analysis_results.csv")
        test_df, _ = load_and_prep_data("./gsm8k_analysis_test_results.csv")
        
        results_storage = {} # To store (mean, std, margin, raw_accs)
        
        # --- 1. Define Statistic Subsets ---
        # We want all non-empty subsets of [min, max, mean]
        base_stats = ['min', 'max', 'mean']
        stat_combinations = []
        for r in range(1, len(base_stats) + 1):
            stat_combinations.extend(itertools.combinations(base_stats, r))
        
        for epoch in range(20, 150, 10):
            # --- 2. Run Baseline Experiments (All Subsets) ---
            for stats_tuple in stat_combinations:
                stats_list = list(stats_tuple)
                name = f"Stats ({'+'.join(stats_list)}) Epoch ({epoch})"
                
                # Prepare Data
                agg_calib = aggregate_df(calib_df, neuron_cols, stats=stats_list)
                agg_test = aggregate_df(test_df, neuron_cols, stats=stats_list)
                
                combined = pd.concat([agg_calib, agg_test], ignore_index=True)
                combined = balance_binary_dataset(combined)
                combined = normalize_data(combined)
                
                dataset, _ = convert_df_to_dataloader(combined)
                
                # Run Experiment
                results_storage[name] = run_experiment(dataset, name=name, n_runs=50)

            # --- 3. Run Threshold Experiment ---
            logger.log("\nPreparing Threshold Dataset...")
            thresh_calib = threshold_activations(calib_df, calib_df)
            thresh_test = threshold_activations(calib_df, test_df)
            
            combined_thresh = pd.concat([thresh_calib, thresh_test], ignore_index=True)
            combined_thresh = balance_binary_dataset(combined_thresh)
            dataset_threshold, _ = convert_df_to_dataloader(combined_thresh)
            
            results_storage[f"Threshold Activations Epoch ({epoch})"] = run_experiment(dataset_threshold, name="Threshold Activations Epoch ({epoch})", n_runs=50)
        
        # --- 4. Final Summary & File Saving ---
        logger.log(f"\n{'='*20} Final Comparison Sorted by Accuracy {'='*20}")
        
        # Sort results by Mean Accuracy (Descending)
        sorted_results = sorted(results_storage.items(), key=lambda x: x[1][0], reverse=True)
        
        # Print Summary Table
        logger.log(f"{'Experiment Name':<30} | {'Mean Acc':<10} | {'95% CI':<20}")
        logger.log("-" * 65)
        for name, (mean, std, margin, _) in sorted_results:
            ci = f"± {margin:.4f}"
            logger.log(f"{name:<30} | {mean:.4f}     | {ci:<20}")

        # Save Raw Data
        logger.log("\nSaving raw accuracy data to 'experiment_raw_data.csv'...")
        
        # Construct DataFrame from results dictionary
        raw_data_dict = {'run_id': range(1, 11)}
        for name, (_, _, _, raw_accs) in results_storage.items():
            # Clean up column name for CSV (remove spaces/parentheses)
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_").lower()
            raw_data_dict[safe_name] = raw_accs
            
        raw_data_df = pd.DataFrame(raw_data_dict)
        raw_data_df.to_csv("classifer_experiment_raw_data.csv", index=False)
        logger.log("Done.")

    except Exception as e:
        logger.log(f"\nCRITICAL ERROR: {str(e)}")
        raise