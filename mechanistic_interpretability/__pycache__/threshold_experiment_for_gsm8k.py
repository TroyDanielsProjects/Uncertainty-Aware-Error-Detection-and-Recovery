import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_threshold_experiment(df, target_neurons=None, train_frac=0.8, output_folder='threshold_experiment_results'):
    """
    Runs an 'Error Detection' experiment using the 'Green Whisker' method.
    
    Logic:
    1. Split data into Train (80%) and Test (20%).
    2. On Train data, filter for CORRECT examples.
    3. Calculate the Upper Whisker (Q3 + 1.5*IQR) of the activation distribution for Correct examples.
       This value becomes the THRESHOLD.
    4. On Test data:
       - If activation > Threshold -> FLAG as Predicted Incorrect (Low Confidence).
       - If activation <= Threshold -> NO PREDICTION (Pass).
    
    We only evaluate the precision of the 'Flagged' items and the recall of actual failures.
    
    Args:
        df (pd.DataFrame): The raw dataframe.
        target_neurons (list, optional): List of neuron column names to analyze. 
                                         If None, automatically selects all columns starting with 'Neuron_'.
        train_frac (float): Fraction of data to use for training.
        output_folder (str): Folder to save results.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Auto-detect neurons if not specified
    if target_neurons is None:
        target_neurons = [c for c in df.columns if c.startswith('Neuron_')]
        
    results_summary = []

    print(f"Starting Threshold Experiment on {len(target_neurons)} neurons...")

    # 1. Aggregation (Token -> Problem level)
    # We need to perform aggregation FIRST to ensure we don't leak problem_id data between train/test
    # (i.e., we don't want half a problem's tokens in train and half in test)
    
    # We assume 'problem_id' exists. We need to create a problem-level dataset.
    # Note: We do this inside the loop if we want to support different neurons having different aggregations,
    # but since the aggregation is 'max', we can do it once if we are careful, 
    # but doing it per neuron is safer if columns vary.
    
    # Get unique problem IDs
    unique_problems = df['problem_id'].unique()
    np.random.seed(42) # For reproducibility
    np.random.shuffle(unique_problems)
    
    split_idx = int(len(unique_problems) * train_frac)
    train_problems = set(unique_problems[:split_idx])
    test_problems = set(unique_problems[split_idx:])
    
    log_file_path = os.path.join(output_folder, 'threshold_experiment_log.txt')
    
    with open(log_file_path, 'w') as f:
        for neuron in target_neurons:
            # Aggregate stats for this neuron
            problem_stats = df.groupby('problem_id').agg({
                neuron: 'max',
                'is_correct': 'first'
            }).reset_index()
            
            # Split into Train/Test
            train_df = problem_stats[problem_stats['problem_id'].isin(train_problems)].copy()
            test_df = problem_stats[problem_stats['problem_id'].isin(test_problems)].copy()
            
            # --- TRAIN: Calculate Threshold ---
            # Filter for Correct examples only
            train_correct = train_df[train_df['is_correct'] == True]
            
            if len(train_correct) == 0:
                msg = f"Skipping {neuron}: No correct examples in training set.\n"
                print(msg.strip())
                f.write(msg)
                continue
                
            activations = train_correct[neuron].astype(float)
            Q1 = activations.quantile(0.25)
            Q3 = activations.quantile(0.75)
            IQR = Q3 - Q1
            upper_whisker = Q3 + 1.5 * IQR
            
            # This is our cutoff. Any activation ABOVE this is "Predicted Incorrect" (Flagged).
            threshold = upper_whisker
            
            # --- TEST: Apply Threshold & Evaluate Flags ---
            test_df['activation'] = test_df[neuron].astype(float)
            
            # Identify Flagged Samples (High Activation)
            flagged_mask = test_df['activation'] > threshold
            flagged_df = test_df[flagged_mask]
            not_flagged_df = test_df[~flagged_mask]
            
            total_test = len(test_df)
            total_flagged = len(flagged_df)
            
            # Evaluate Precision (Of the ones we flagged, how many were actually incorrect?)
            # is_correct == False means it was an actual failure.
            flagged_incorrect_count = len(flagged_df[flagged_df['is_correct'] == False]) # Correctly Flagged
            flagged_correct_count = len(flagged_df[flagged_df['is_correct'] == True])    # False Alarm
            
            precision = flagged_incorrect_count / total_flagged if total_flagged > 0 else 0.0
            
            # Evaluate Recall (Of all actual failures, how many did we flag?)
            total_actual_failures = len(test_df[test_df['is_correct'] == False])
            recall = flagged_incorrect_count / total_actual_failures if total_actual_failures > 0 else 0.0
            
            # Calculate Missed Detections (Low activation but was incorrect)
            missed_failures = len(not_flagged_df[not_flagged_df['is_correct'] == False])
            
            # Log results
            log = f"""
{'='*40}
Neuron: {neuron}
{'='*40}
Threshold (Upper Whisker of Correct Train Data): {threshold:.4f}
(Train Stats: Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f})

Test Set Flagging Performance:
------------------------------
Total Test Samples:      {total_test}
Total Actual Failures:   {total_actual_failures}

[FLAGGED GROUP] (Activation > Threshold):
Total Flagged:           {total_flagged} ({total_flagged/total_test:.1%} of test set)
  - Successfully Caught: {flagged_incorrect_count} (Actually Incorrect)
  - False Alarms:        {flagged_correct_count} (Actually Correct)
  
Precision (Success / Total Flagged): {precision:.2%}
"When this neuron yells, it is right {precision:.1%} of the time."

[RECALL / COVERAGE]:
Total Failures Caught:   {flagged_incorrect_count} / {total_actual_failures}
Recall:                  {recall:.2%}
"This neuron catches {recall:.1%} of all failures."

(Missed {missed_failures} failures in the un-flagged group)
"""
            print(log)
            f.write(log)
            
            # Visualize the split
            plt.figure(figsize=(10, 6))
            
            # Plot 1: Histogram of Test Correct vs Incorrect with Threshold Line
            plt.hist(test_df[test_df['is_correct']==True]['activation'], bins=30, alpha=0.5, label='Actual Correct', color='green')
            plt.hist(test_df[test_df['is_correct']==False]['activation'], bins=30, alpha=0.5, label='Actual Incorrect', color='red')
            plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Flag Threshold ({threshold:.2f})')
            
            plt.title(f'{neuron}: Test Set Distribution & Flag Cutoff')
            plt.xlabel('Max Activation')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(output_folder, f'{neuron}_threshold_test.png'))
            plt.close()

# --- Example Usage (Mock Data) ---
if __name__ == "__main__":
    # creating dummy data to demonstrate
    # Including multiple neurons to show the loop functionalit
    
    df = pd.read_csv("./gsm8k_analysis_results.csv")
    
    # Run the NEW Threshold Experiment
    target_neurons = ['Neuron_1674']
    run_threshold_experiment(df, train_frac=0.8)