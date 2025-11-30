import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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

def plot_mean_activation_dist(df, neuron_cols, output_dir='plots'):
    """
    1. Histogram of Average Activation (Correct vs Incorrect).
    Aggregates all neuron values for a problem_id to get one 'mean' score.
    """
    # Group by problem_id to get one stat per generation
    # We take the mean across ALL tokens and ALL tracked neurons for that problem
    problem_stats = df.groupby(['problem_id', 'is_correct'])[neuron_cols].mean().mean(axis=1).reset_index()
    problem_stats.columns = ['problem_id', 'is_correct', 'mean_activation']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=problem_stats, 
        x='mean_activation', 
        hue='is_correct', 
        kde=True, 
        element="step",
        stat="density",
        common_norm=False,
        palette={True: "green", False: "red"}
    )
    plt.title("Distribution of Average Neuron Activation\n(Correct vs Incorrect)")
    plt.xlabel("Mean Activation Value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mean_activation_distribution.png")
    print(f"Saved mean_activation_distribution.png")
    plt.close()

def plot_max_activation_boxplot(df, neuron_cols, output_dir='plots'):
    """
    2. Max Activation Analysis.
    Tests the hypothesis: "Max activations typically show the answer is wrong."
    """
    # Get the single MAX value recorded across all tokens/neurons for each problem
    problem_max = df.groupby(['problem_id', 'is_correct'])[neuron_cols].max().max(axis=1).reset_index()
    problem_max.columns = ['problem_id', 'is_correct', 'max_activation']
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=problem_max,
        x='is_correct',
        y='max_activation',
        hue='is_correct', # Fix for FutureWarning
        legend=False,     # Fix for FutureWarning
        palette={True: "green", False: "red"}
    )
    
    plt.title("Maximum Activation Value Recorded\n(Do spikes predict errors?)")
    plt.xlabel("Is Answer Correct?")
    plt.ylabel("Max Activation Value")
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.savefig(f"{output_dir}/max_activation_boxplot.png")
    print(f"Saved max_activation_boxplot.png")
    plt.close()

def plot_last_token_activations(df, neuron_cols, output_dir='plots'):
    """
    3. Activations of the 'True Answer' Token.
    We approximate this by taking the LAST token of the generation.
    """
    # Get the last token index for each problem
    last_tokens = df.loc[df.groupby('problem_id')['token_index'].idxmax()]
    
    # Calculate mean activation of neurons ONLY on this last token
    last_tokens['last_token_mean_act'] = last_tokens[neuron_cols].mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    
    # Filter for True Answers only as requested, but comparing to False is often useful
    # We will show both for context
    sns.kdeplot(
        data=last_tokens,
        x='last_token_mean_act',
        hue='is_correct',
        fill=True,
        palette={True: "green", False: "red"},
        warn_singular=False
    )
    
    plt.title("Neuron Activation Levels at the Final Token")
    plt.xlabel("Mean Activation (Last Token)")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{output_dir}/last_token_activations.png")
    print(f"Saved last_token_activations.png")
    plt.close()

def plot_activation_trajectory(df, neuron_cols, output_dir='plots'):
    """
    4. Activation Trajectory.
    Shows how activations evolve over the generation steps.
    """
    # Calculate mean activation across all neurons for each row
    df['step_mean'] = df[neuron_cols].mean(axis=1)
    
    plt.figure(figsize=(12, 6))
    
    # We use lineplot which automatically computes mean and confidence intervals
    sns.lineplot(
        data=df,
        x='token_index',
        y='step_mean',
        hue='is_correct',
        palette={True: "green", False: "red"}
    )
    
    plt.title("Average Activation Trajectory Over Generation")
    plt.xlabel("Token Position (Time Step)")
    plt.ylabel("Average Activation")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{output_dir}/activation_trajectory.png")
    print(f"Saved activation_trajectory.png")
    plt.close()

if __name__ == "__main__":
    # Setup
    FILE_PATH = "gsm8k_analysis_results.csv" # Or "aime_results.csv"
    OUTPUT_DIR = "plots"
    
    # Check if file exists, if not create dummy for demo
    if not os.path.exists(FILE_PATH):
        print(f"Warning: {FILE_PATH} not found. Creating dummy data for demonstration.")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Create dummy data
        data = []
        for i in range(20): # 20 problems
            is_corr = i % 2 == 0
            # Correct answers generally lower/stable, Incorrect spiky
            base = 0.5 if is_corr else 1.0 
            for t in range(10): # 10 tokens
                row = {
                    "problem_id": i,
                    "token_index": t,
                    "is_correct": is_corr,
                    "Neuron_1": np.random.normal(base, 0.5),
                    "Neuron_2": np.random.normal(base, 0.5) + (5.0 if not is_corr and t==8 else 0) # Spike
                }
                data.append(row)
        pd.DataFrame(data).to_csv(FILE_PATH, index=False)
        
    try:
        # Load
        df, neuron_cols = load_and_prep_data(FILE_PATH)
        
        # Plot
        plot_mean_activation_dist(df, neuron_cols, OUTPUT_DIR)
        plot_max_activation_boxplot(df, neuron_cols, OUTPUT_DIR)
        plot_last_token_activations(df, neuron_cols, OUTPUT_DIR)
        plot_activation_trajectory(df, neuron_cols, OUTPUT_DIR)
        
        print(f"\nAll plots saved to ./{OUTPUT_DIR}/")
        
    except Exception as e:
        print(f"An error occurred: {e}")