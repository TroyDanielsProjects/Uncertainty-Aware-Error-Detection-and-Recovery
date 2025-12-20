import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

def load_and_prep_data(filepath):
    """
    Loads the CSV and identifies neuron columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Fix: Ensure is_correct is strictly boolean to avoid palette KeyErrors
    # The error "missing keys: {'False', 'True'}" indicates pandas read them as strings.
    if df['is_correct'].dtype == object:
         df['is_correct'] = df['is_correct'].astype(str).str.lower() == 'true'
    
    # Identify columns that start with 'Neuron_'
    neuron_cols = [c for c in df.columns if c.startswith('Neuron_')]
    
    print(f"Loaded data with {len(df)} rows.")
    print(f"Found {len(neuron_cols)} tracked neurons.")
    
    return df, neuron_cols

def aggregate_df(df, neuron_cols):
    # This calculates min, max, and mean for every column in neuron_cols
    X = df.groupby(['problem_id', 'is_correct'])[neuron_cols].agg(['min', 'max', 'mean'])

    # Right now, columns look like: ('neuron_0', 'min'), ('neuron_0', 'max')...
    # This step converts them to: 'neuron_0_min', 'neuron_0_max'...
    X.columns = [f'{col}_{stat}' for col, stat in X.columns]

    # 3. Reset index to turn problem_id/is_correct back into columns
    X = X.reset_index()
    return X

def convert_df_to_dataloader(X):
    # 1. Isolate Features (X)
    # Drop metadata and label columns to leave only neuron stats
    drop_cols = ['problem_id', 'is_correct']
    features_df = X.drop(columns=drop_cols, errors='ignore')

    # Convert to numpy first, then tensor. float32 is crucial for standard GPU training.
    X_tensor = torch.tensor(features_df.values, dtype=torch.float32)

    # 2. Isolate Labels (y)
    # Cast to float32 if using BCEWithLogitsLoss (standard for binary classification)
    # Unsqueeze(1) transforms shape from [N] to [N, 1]
    y_tensor = torch.tensor(X['is_correct'].values, dtype=torch.float32).unsqueeze(1)
    print(f'Accuracy of Llama 3.1 model on gsm8k dataset: {y_tensor.mean()}')

    # Check shapes
    print(f"Features shape: {X_tensor.shape}") # Should be (N_samples, N_neurons * 3)
    print(f"Labels shape:   {y_tensor.shape}") # Should be (N_samples, 1)

    # 3. (Optional) Wrap in a DataLoader for batching
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, X_tensor.shape[1]

# 1. Define the MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),  # Helps prevent overfitting on specific neurons
            
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            # Output Layer (No Sigmoid here!)
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, x):
        return self.network(x)
    
def balance_binary_dataset(df):
    """
    Helper to balance True/False examples by undersampling the majority class.
    """
    true_df = df[df['is_correct'] == True]
    false_df = df[df['is_correct'] == False]
    
    n_true = len(true_df)
    n_false = len(false_df)
    
    print(f"\n--- Balancing Dataset ---")
    print(f"Original counts -> True: {n_true}, False: {n_false}")
    
    # Find the count of the minority class
    min_count = min(n_true, n_false)
    
    # Sample both to the size of the minority
    true_balanced = true_df.sample(n=min_count, random_state=42)
    false_balanced = false_df.sample(n=min_count, random_state=42)
    
    # Concatenate and shuffle
    balanced_df = pd.concat([true_balanced, false_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced counts -> True: {len(true_balanced)}, False: {len(false_balanced)}")
    print(f"Total samples: {len(balanced_df)}")
    print("-------------------------\n")
    
    return balanced_df

def normalize_data(df):
    """
    Standardizes feature columns (mean=0, std=1).
    Ignores non-numeric columns like problem_id/is_correct.
    """
    print("\n--- Normalizing Data ---")
    
    # Identify feature columns (exclude metadata)
    meta_cols = ['problem_id', 'is_correct']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    # Calculate Mean and Std
    features = df[feature_cols]
    mean = features.mean()
    std = features.std()
    
    # Apply (X - Mean) / Std
    # Replace 0 std with 1 to avoid division by zero (for constant neurons)
    std = std.replace(0, 1)
    df[feature_cols] = (features - mean) / std
    
    print("Data normalized (Z-score scaling).")
    print("-------------------------\n")
    return df

if __name__ == "__main__":
    # --- Data Prep ---
    df, neuron_cols = load_and_prep_data("./gsm8k_analysis_results_qwen.csv")
    X_df = aggregate_df(df, neuron_cols)

    # --- NEW: Balance the Dataset ---
    X_df = balance_binary_dataset(X_df)

    # --- NEW: Normalize the Dataset ---
    X_df = normalize_data(X_df)
    
    # Get the full dataloader and input dimension
    full_loader, input_dim = convert_df_to_dataloader(X_df)

    # --- Splitting Logic ---
    # We extract the underlying dataset from the full_loader to split it
    full_dataset = full_loader.dataset 
    
    total_count = len(full_dataset)
    test_count = 30
    train_count = total_count - test_count

    # Ensure valid split size
    if train_count < 0:
        raise ValueError(f"Dataset too small ({total_count}) for test split of {test_count}")

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_count, test_count], 
        generator=generator
    )

    # Create separate DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    # --- Setup Model ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    model = SimpleMLP(input_dim, hidden_dim=64).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    epochs = 100
    model.train() 
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(train_loader):.4f}")

    # --- Evaluation Loop ---
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
            
    print(f"Test Accuracy: {correct/total:.4f}")