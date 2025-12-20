import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_neuron_calibration(df, neuron_col, clip_quantile=None):
    """
    Evaluates the calibration of a specific neuron's activation against the correctness of the output.
    
    The assumption is that High Activation indicates a High Probability of Failure (Incorrectness).
    
    Args:
        df (pd.DataFrame): The input dataframe containing columns:
                           ['problem_id', 'is_correct', neuron_col, ...]
        neuron_col (str): The name of the column representing the neuron to evaluate.
        clip_quantile (float, optional): A value between 0 and 0.5. If provided, the activations
                                         will be clipped at these quantiles (lower and upper) 
                                         before normalization to handle outliers. 
                                         e.g., 0.01 clips the top and bottom 1%.

    Returns:
        pd.DataFrame: A calibration table with bins, average predicted failure probability,
                      actual failure rate, and sample counts.
    """
    
    # 1. Group by problem_id to get the trajectory-level stats
    # We take the max activation for the neuron and the first is_correct label 
    # (assuming is_correct is constant for the problem_id)
    problem_stats = df.groupby('problem_id').agg({
        neuron_col: 'max',
        'is_correct': 'first'
    }).reset_index()

    activations = problem_stats[neuron_col].astype(float)

    # 2. Handle Outliers (Optional)
    if clip_quantile is not None:
        lower_bound = activations.quantile(clip_quantile)
        upper_bound = activations.quantile(1.0 - clip_quantile)
        # Clip values to the bounds
        activations = activations.clip(lower=lower_bound, upper=upper_bound)

    # 3. Normalize activations to [0, 1]
    # 0 = Lowest Max Activation (Thinks it is Correct)
    # 1 = Highest Max Activation (Thinks it is Incorrect)
    min_act = activations.min()
    max_act = activations.max()
    
    # Avoid division by zero if all values are the same
    if max_act == min_act:
        problem_stats['normalized_score'] = 0.0
    else:
        problem_stats['normalized_score'] = (activations - min_act) / (max_act - min_act)

    # 4. Discretize based on the first decimal (0.0, 0.1, ... 0.9, 1.0)
    # We floor to the nearest 0.1
    # We handle the edge case where normalized_score is exactly 1.0 by mapping it to 0.9 
    # or keeping it as 1.0 depending on preference. Here we simply floor.
    problem_stats['bin'] = np.floor(problem_stats['normalized_score'] * 10) / 10
    
    # Fix the edge case where 1.0 becomes 1.0, but usually we might want 1.0 included in the 0.9 bin 
    # or separate. Given the prompt asks to bucket based on first decimal, 1.0 is technically its own edge or 0.9+.
    # Let's cap the bin at 0.9 to group 1.0 into the highest bucket, typically 0.9-1.0 range.
    problem_stats['bin'] = problem_stats['bin'].clip(upper=0.9)

    # 5. Calculate Calibration Statistics
    # We want to compare:
    #   - "Inverse Probability" (Normalized Score) -> The model's belief it will fail.
    #   - "Actual Failure Rate" -> (1 - is_correct).
    
    # Convert is_correct to numeric (1 for True/Correct, 0 for False/Incorrect)
    # Then Failure = 1 - is_correct
    problem_stats['is_correct_num'] = problem_stats['is_correct'].astype(int)
    problem_stats['actual_failure'] = 1 - problem_stats['is_correct_num']

    calibration = problem_stats.groupby('bin').agg(
        count=('problem_id', 'count'),
        avg_predicted_failure_prob=('normalized_score', 'mean'),
        actual_failure_rate=('actual_failure', 'mean')
    ).reset_index()

    # Calculate error (calibration gap)
    calibration['calibration_error'] = abs(
        calibration['avg_predicted_failure_prob'] - calibration['actual_failure_rate']
    )

    return calibration

def analyze_all_neurons(df, output_folder='calibration_results', clip_quantile=None):
    """
    Loops through all columns starting with 'Neuron_', performs calibration analysis,
    and saves the text output and plots to a specified folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    neuron_cols = [c for c in df.columns if c.startswith('Neuron_')]
    log_file_path = os.path.join(output_folder, 'calibration_summary.txt')
    
    print(f"Starting analysis on {len(neuron_cols)} neurons. Output folder: {output_folder}")
    
    all_results_list = []

    with open(log_file_path, 'w') as f:
        for neuron in neuron_cols:
            print(f"Processing {neuron}...", end='\r')
            
            # 1. Run Analysis
            results = evaluate_neuron_calibration(df, neuron, clip_quantile)
            all_results_list.append(results)
            
            # 2. Save Text Output to File
            header = f"\n{'='*40}\nAnalysis for {neuron}\n{'='*40}\n"
            f.write(header)
            f.write(results.to_string())
            f.write("\n\n")
            
            # 3. Save Plot
            plot_filename = f"{neuron}_calibration.png"
            plot_path = os.path.join(output_folder, plot_filename)
            plot_calibration_curve(results, neuron, output_path=plot_path)
            
        # --- Create and Save Aggregate Summary ---
        if all_results_list:
            # Combine all dataframes
            all_results_df = pd.concat(all_results_list)
            
            # Group by bin and calculate the mean over all neurons
            aggregate_summary = all_results_df.groupby('bin').agg(
                avg_predicted_failure_prob=('avg_predicted_failure_prob', 'mean'),
                actual_failure_rate=('actual_failure_rate', 'mean'),
                calibration_error=('calibration_error', 'mean'),
                # Average number of samples falling in this bin per neuron
                count=('count', 'mean') 
            ).reset_index()

            # Append to text file
            header = f"\n{'='*40}\nAGGREGATE SUMMARY (Mean over all neurons)\n{'='*40}\n"
            f.write(header)
            f.write(aggregate_summary.to_string())
            f.write("\n\n")

            # Save Aggregate Plot
            agg_plot_path = os.path.join(output_folder, "AGGREGATE_calibration.png")
            plot_calibration_curve(aggregate_summary, "All Neurons (Average)", output_path=agg_plot_path)

    print(f"\nAnalysis complete. Summary saved to {log_file_path}")
            
    print(f"\nAnalysis complete. Summary saved to {log_file_path}")

def plot_calibration_curve(calibration_df, neuron_name, output_path=None):
    """
    Plots the calibration curve (Reliability Diagram) and a histogram of the sample distribution.
    Uses two subplots: top for the curve, bottom for the histogram.
    
    Args:
        calibration_df (pd.DataFrame): Output from evaluate_neuron_calibration.
        neuron_name (str): Name of the neuron for the title.
        output_path (str, optional): If provided, saves the plot to this path instead of showing it.
    """
    # Create subplots: Top for curve (3/4 height), Bottom for hist (1/4 height)
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, 
        ncols=1, 
        figsize=(10, 8), 
        sharex=True, 
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- TOP PLOT: Calibration Curve ---
    # Plot Perfect Calibration Line (Diagonal)
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    # Plot Actual Calibration Curve
    ax1.plot(
        calibration_df['avg_predicted_failure_prob'], 
        calibration_df['actual_failure_rate'], 
        marker='o', 
        linewidth=2, 
        color='tab:blue',
        label='Neuron Performance'
    )
    
    ax1.set_ylabel('Actual Failure Rate')
    ax1.set_title(f'Reliability Diagram for {neuron_name}')
    ax1.set_ylim(0, 1.05) # Little buffer at top
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # --- BOTTOM PLOT: Histogram of Samples ---
    # Align bars to the center of bins. Bins are 0.0, 0.1... so center is +0.05
    # If 'bin' column exists (it should), use it. Otherwise approximate from avg_prob.
    if 'bin' in calibration_df.columns:
        x_positions = calibration_df['bin'] + 0.05
    else:
        x_positions = calibration_df['avg_predicted_failure_prob']

    ax2.bar(
        x_positions, 
        calibration_df['count'], 
        width=0.08, # Slightly narrower than 0.1 to show gaps
        alpha=0.6, 
        color='tab:blue', 
        label='Sample Count'
    )
    ax2.set_xlabel('Predicted Failure Probability (Normalized Activation)')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)
    
    # Ensure x-axis is 0-1
    plt.xlim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free memory
    else:
        plt.show()

# --- Example Usage (Mock Data) ---
if __name__ == "__main__":
    
    df = pd.read_csv("./gsm8k_analysis_results.csv")
    
    analyze_all_neurons(df, "./plots/max_activation_calibration", clip_quantile=0.2)