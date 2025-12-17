import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from IPython.display import display, HTML
import os

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- PREDICTOR ARTIFACT CLASS (New) ---

class PredictorArtifact:
    """Standardized function-like object for predictions."""
    def __init__(self, model, scaler, features, mode='classifier', label_map=None):
        self.model = model
        self.scaler = scaler
        self.features = features
        self.mode = mode
        self.label_map = label_map or {}

    def __call__(self, df_input):
        """
        Input: DataFrame containing necessary feature columns (or dict).
        Output: (probabilities_array, labels_list)
        """
        if isinstance(df_input, dict): df_input = pd.DataFrame([df_input])
        
        # Select and Scale
        X = df_input[self.features].fillna(0)
        if self.scaler: X = self.scaler.transform(X)

        if self.mode == 'clustering':
            preds = self.model.predict(X)
            # Transform distance to "confidence" (simple heuristic) or return 1.0
            dists = self.model.transform(X)
            probs = 1.0 / (1.0 + dists.min(axis=1)) 
            # Map cluster ID to rich text label
            labels = [self.label_map.get(p, f"Cluster {p}") for p in preds]
            return probs, labels
        
        else: # Classifier
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[:, 1]
            else:
                probs = np.zeros(len(X))
            return probs, [""] * len(X) # Label empty for standard classifiers

# --- HELPER FUNCTIONS (Internal) ---

def _generate_token_html(tokens, values, metric_name):
    """Generates HTML with token-level coloring."""
    if not tokens: return "No Trace Data"
    
    if not values or len(values) == 0:
        vals = np.zeros(len(tokens))
    elif len(values) < len(tokens):
        vals = np.array(values + [0]*(len(tokens)-len(values)))
    else:
        vals = np.array(values[:len(tokens)])

    v_min, v_max = vals.min(), vals.max()
    
    html_parts = []
    for t, v in zip(tokens, vals):
        if v_max - v_min == 0: norm = 0.5
        else: norm = (v - v_min) / (v_max - v_min)
        
        hue = 120 * norm 
        color = f"hsl({hue}, 80%, 85%)"
        
        t_clean = str(t).replace('Ġ', ' ').replace('Ċ', '↵').replace('<', '&lt;').replace('>', '&gt;')
        html_parts.append(f"<span style='background-color:{color}; padding:0 2px; border-radius:2px;' title='{v:.4f}'>{t_clean}</span>")
    
    return "".join(html_parts)

def _plot_decision_boundary(ax, df, x_col, y_col, target, model=None):
    """
    Plots scatter points. 
    If model is provided, plots that model's boundary (e.g. Train boundary).
    If no model provided, fits a new one on the displayed data (e.g. Test boundary).
    """
    plot_df = df[[x_col, y_col, target]].dropna()
    X = plot_df[[x_col, y_col]].values
    y = plot_df[target].values
    
    if len(np.unique(y)) < 2: return 
    
    # 1. Scatter (The Data Points)
    sns.scatterplot(
        data=plot_df, x=x_col, y=y_col, hue=target, 
        palette={0: '#e74c3c', 1: '#2ecc71'}, 
        alpha=0.6, s=30, ax=ax, legend=False
    )
    
    # 2. Classifier Logic
    if model is not None:
        clf = model # Use the Pre-trained (Train) model
        boundary_type = "Train Boundary"
    else:
        clf = LogisticRegression()
        clf.fit(X, y) # Fit on the current (Test) data
        boundary_type = "Test Boundary"
        
    # Calculate stats on the Visible Data using the Model
    try: acc = accuracy_score(y, clf.predict(X))
    except: acc = 0.0
    try: auc = roc_auc_score(y, clf.predict_proba(X)[:,1])
    except: auc = 0.5
    
    # 3. Contours
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--', linewidths=1)
    ax.contourf(xx, yy, Z, levels=[0.0, 0.5, 1.0], colors=['#e74c3c', '#2ecc71'], alpha=0.1)
    
    clean_y_label = y_col.replace('_Z', '')
    ax.set_title(f"vs {clean_y_label}\n{boundary_type}\nAcc: {acc:.1%} | AUC: {auc:.2f}", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")


def run_full_analysis(train_df, metrics, test_df=None):
    """
    Iterates through metrics and generates report cards + plots.
    Fits baselines on train_df, but visualizes/evaluates on test_df (if provided).
    """
    # 1. Setup Data
    if train_df is None or len(train_df) < 5:
        print("⚠️ Not enough data to run analysis.")
        return {}

    viz_df = test_df if test_df is not None else train_df
    data_label = "Test Set" if test_df is not None else "Train Set"
    
    sns.set_theme(style="white", context="paper")
    warnings.filterwarnings('ignore')

    predictors = {}
    
    # 2. Filter Valid Metrics
    valid_metrics = [m for m in metrics if m in train_df.columns and f'{m}_Z' in train_df.columns]

    for m_exam in valid_metrics:
        # --- A. Statistical Power (Train on Train, Eval on Viz) ---
        try:
            # Fit single-variable LR on Train
            clf_single = LogisticRegression()
            clf_single.fit(train_df[[f'{m_exam}_Z']], train_df['Correct'])
            
            # Eval on Viz (Test)
            preds = clf_single.predict(viz_df[[f'{m_exam}_Z']])
            probs = clf_single.predict_proba(viz_df[[f'{m_exam}_Z']])[:, 1]
            
            acc_single = accuracy_score(viz_df['Correct'], preds)
            auroc_single = roc_auc_score(viz_df['Correct'], probs)

            # Store Predictor (trained artifact)
            predictors[m_exam] = PredictorArtifact(
                model=clf_single, 
                scaler=None, 
                features=[f'{m_exam}_Z'], 
                mode='classifier'
            )
        except Exception as e:
            print(f"Stats failed for {m_exam}: {e}")
            auroc_single, acc_single = 0.5, 0.0

        # --- B. Header ---
        header = f"""
        <div style='border-bottom:3px solid #3498db; padding-top:20px; margin-bottom:15px'>
            <div style='display:flex; justify-content:space-between; align-items:center'>
                <h2 style='margin:0; color:#2c3e50'>{m_exam} Analysis</h2>
                <div style='font-family:sans-serif; font-size:0.95em;'>
                    <span style='color:#7f8c8d'>Power ({data_label}):</span> 
                    <span style='background:#f1f2f6; padding:4px 8px; border-radius:4px; font-weight:bold; color:#2c3e50; border:1px solid #ccc'>
                        AUROC: {auroc_single:.3f}
                    </span>
                    <span style='background:#f1f2f6; padding:4px 8px; border-radius:4px; font-weight:bold; color:#2c3e50; border:1px solid #ccc; margin-left:5px'>
                        Acc: {acc_single:.1%}
                    </span>
                </div>
            </div>
        </div>
        """
        display(HTML(header))
        
        # --- C. Pairwise Plots (Using Viz Data) ---
        other_metrics = [m for m in valid_metrics if m != m_exam]
        if other_metrics:
            fig, axes = plt.subplots(1, len(other_metrics), figsize=(4 * len(other_metrics), 3.5))
            if len(other_metrics) == 1: axes = [axes]
            
            for i, m_other in enumerate(other_metrics):
                # 1. Train a 2D Classifier on TRAIN data
                cols_2d = [f'{m_exam}_Z', f'{m_other}_Z']
                clf_2d = LogisticRegression()
                clf_2d.fit(train_df[cols_2d], train_df['Correct'])
                
                # 2. Plot on VIZ data (Test), passing the TRAIN model
                _plot_decision_boundary(
                    axes[i], 
                    viz_df, # <--- The data points (Test)
                    cols_2d[0], 
                    cols_2d[1], 
                    'Correct',
                    model=clf_2d # <--- The boundary (Train)
                )
                
                axes[i].set_xlabel(f"{m_exam} (std)")
                axes[i].set_ylabel(f"{m_other} (std)")
            plt.tight_layout()
            plt.show()
        # --- D. Trace Examples (From Viz Data) ---
        # Map metric to trace column
        trace_map = {
            'Entropy': 'uq_entropy_trace',
            'LogitGap': 'uq_logit_gap_trace',
            'Mechanistic': 'uq_mech_trace'
        }
        trace_col = trace_map.get(m_exam, None)
        
        sorted_viz = viz_df.sort_values(by=m_exam, ascending=False)
        
        html_ex = "<div style='display:flex; gap:20px; margin-top:10px'>"
        
        def make_card(row, title_prefix):
            tokens = row.get('uq_tokens', [])
            trace_vals = row.get(trace_col, []) if trace_col and trace_col in row else []
            
            # Fallback for metrics without traces (e.g. Length)
            if not trace_vals: 
                colored_text = "".join([t.replace('Ġ', ' ') for t in tokens[:100]]) + "..."
            else:
                colored_text = _generate_token_html(tokens, trace_vals, m_exam)
                
            status = "✅" if row['Correct'] else "❌"
            val = row[m_exam]
            val_disp = f"{int(val)}" if m_exam == 'Length' else f"{val:.3f}"
            
            return f"""
            <div style='flex:1; border:1px solid #ddd; padding:10px; border-radius:5px; font-family:monospace; font-size:0.85em; background:#fafafa'>
                <div style='font-weight:bold; border-bottom:1px solid #eee; margin-bottom:5px; color:#333'>
                    {title_prefix} ({m_exam}: {val_disp}) {status}
                </div>
                <div style='max-height:150px; overflow-y:auto; line-height:1.4; background:white; padding:5px; border:1px solid #eee'>
                    {colored_text}
                </div>
            </div>
            """

        html_ex += "<div style='flex:1'><h4 style='margin:0 0 5px 0'>Highest 3 (Viz Set)</h4>"
        for _, row in sorted_viz.head(3).iterrows(): html_ex += make_card(row, "High")
        html_ex += "</div>"
        
        html_ex += "<div style='flex:1'><h4 style='margin:0 0 5px 0'>Lowest 3 (Viz Set)</h4>"
        for _, row in sorted_viz.tail(3).iterrows(): html_ex += make_card(row, "Low")
        html_ex += "</div></div>"
        
        display(HTML(html_ex))
    
    return predictors



# ==============================================================================
#  ADVANCED ANALYSIS & DASHBOARDING (Notebook Reproduction)
# ==============================================================================
import itertools
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, 
    log_loss, brier_score_loss, roc_curve
)
from IPython.display import display, Markdown, HTML

# --- CUSTOM CSS ---
DASHBOARD_CSS = """
<style>
    :root { --bg-color: #ffffff; --text-color: #2c3e50; --accent: #3498db; --border: #e0e0e0; --success: #27ae60; --warning: #f39c12; --danger: #c0392b; }
    .db-container { font-family: 'Segoe UI', sans-serif; max-width: 100%; color: var(--text-color); }
    .db-header { background: #f8f9fa; padding: 15px; border-bottom: 2px solid var(--accent); margin-bottom: 20px; }
    .db-stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-bottom: 20px; }
    .db-stat-card { background: #fff; border: 1px solid var(--border); padding: 10px; text-align: center; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .db-stat-val { font-size: 1.2em; font-weight: bold; }
    .db-stat-label { font-size: 0.8em; color: #7f8c8d; text-transform: uppercase; }
    .cluster-card { border: 1px solid var(--border); margin-bottom: 25px; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .cluster-head { padding: 10px 15px; display: flex; justify-content: space-between; align-items: center; background: #f1f2f6; border-bottom: 1px solid var(--border); }
    .cluster-title { font-weight: bold; font-size: 1.1em; }
    .acc-container { display: flex; align-items: center; gap: 10px; }
    .acc-bar-bg { width: 100px; height: 8px; background: #ddd; border-radius: 4px; overflow: hidden; }
    .acc-bar-fill { height: 100%; transition: width 0.3s; }
    .metric-table { width: 100%; border-collapse: collapse; font-size: 0.9em; margin: 10px 0; }
    .metric-table th { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; color: #7f8c8d; }
    .metric-table td { padding: 8px; border-bottom: 1px solid #eee; }
    .shift-bar-container { width: 100px; height: 6px; background: #eee; position: relative; border-radius: 2px; }
    .shift-bar { height: 100%; position: absolute; top: 0; }
    details { margin: 0; border-top: 1px solid #eee; }
    summary { padding: 10px 15px; cursor: pointer; background: #fff; font-weight: 500; color: var(--accent); user-select: none; }
    summary:hover { background: #f8f9fa; }
    .ex-content { padding: 15px; background: #fafafa; border-top: 1px solid #eee; font-family: monospace; white-space: pre-wrap; font-size: 0.85em; max-height: 400px; overflow-y: auto; }
    .status-badge-pass { background: var(--success); color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; vertical-align: middle; }
    .status-badge-fail { background: var(--danger); color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; vertical-align: middle; }
    .pred-box { border-left: 3px solid var(--accent); padding-left: 10px; margin: 5px 0; color: #444; background: rgba(52, 152, 219, 0.05); padding: 5px; }
</style>
"""

# --- 1. ADVANCED DATA LOADING ---

def load_data_universal(db_path):
    """
    Fixed Loader: 
    1. Uses correct column names (similarity_vector).
    2. Generates Z-scores automatically (for run_full_analysis).
    3. Computes Semantic Entropy (for Dashboard).
    """
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return None, None, None

    conn = sqlite3.connect(db_path)
    
    # 1. Get Latest Experiment
    try:
        exp_df = pd.read_sql("SELECT experiment_id, name FROM Experiments ORDER BY experiment_id DESC LIMIT 1", conn)
        if exp_df.empty: return None, None, None
        latest_id, name = exp_df.iloc[0]['experiment_id'], exp_df.iloc[0]['name']
    except: return None, None, None

    print(f"Loading Run: {name} (ID: {latest_id})")
    
    # 2. Fetch Data (FIXED COLUMN NAMES)
    query = f"""
    SELECT 
        r.result_id, r.question_id_external, r.question_text, r.gold_answer, 
        r.predicted_answer, r.full_trace_text, r.is_correct,
        r.uq_avg_entropy as 'Entropy', 
        r.uq_min_logit_gap as 'LogitGap',
        r.uq_heuristic_score as 'Heuristic', 
        r.uq_mech_score as 'Mechanistic',
        r.uq_coherence as 'Consistency',   -- Use pre-computed Coherence (excludes self)
        r.similarity_vector,                   -- Correct column name
        r.uq_tokens, r.uq_mech_trace, r.uq_entropy_trace, r.uq_logit_gap_trace
    FROM Results r 
    WHERE r.experiment_id = {latest_id}
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty: return None, None, None
    df['Correct'] = df['is_correct'].astype(int)
    
    # 3. Parse JSON Columns
    json_cols = ['uq_tokens', 'uq_mech_trace', 'uq_entropy_trace', 'uq_logit_gap_trace', 'similarity_vector']
    for col in json_cols:
        df[col] = df[col].apply(lambda x: json.loads(x) if x and isinstance(x, str) else [])

    # 4. Feature Engineering
    # Semantic Entropy (Group-wise)
    def calc_se(sub):
        answers = sub['predicted_answer'].fillna("").str.strip().str.lower()
        probs = answers.value_counts(normalize=True)
        return -np.sum(probs * np.log(probs + 1e-10))

    if 'question_id_external' in df.columns:
        try: se_map = df.groupby('question_id_external').apply(calc_se, include_groups=False)
        except TypeError: se_map = df.groupby('question_id_external').apply(calc_se)
        df['Semantic'] = df['question_id_external'].map(se_map)
    else:
        df['Semantic'] = 0.0

    df['Length'] = df['uq_tokens'].apply(len)
    df = df[df['Length'] <= 500] # Safety Filter

    # 5. Generate Z-Scores (CRITICAL FOR run_full_analysis)
    potential_metrics = ['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic', 'Consistency', 'Semantic', 'Length']
    valid_metrics = []
    
    for m in potential_metrics:
        if m in df.columns and df[m].var() > 0:
            valid_metrics.append(m)
            # Create _Z column
            df[f'{m}_Z'] = (df[m] - df[m].mean()) / df[m].std()

    return df, valid_metrics, (latest_id, name)

def generate_cluster_html(c_id, count, acc, metrics_html, examples_html):
    if acc >= 80: acc_color = "var(--success)"
    elif acc >= 50: acc_color = "var(--warning)"
    else: acc_color = "var(--danger)"
    
    return f"""
    <div class="cluster-card">
        <div class="cluster-head" style="border-left: 5px solid {acc_color}">
            <div class="cluster-title">CLUSTER {c_id}</div>
            <div class="acc-container">
                <span style="font-size: 0.9em; font-weight: bold;">{acc:.1f}% Acc</span>
                <div class="acc-bar-bg"><div class="acc-bar-fill" style="width: {acc}%; background: {acc_color};"></div></div>
                <span style="font-size: 0.8em; color: #666;">({count} Samples)</span>
            </div>
        </div>
        <div style="padding: 15px;">
            <table class="metric-table">
                <thead><tr><th>Metric</th><th>Shift (SD)</th><th>Mean</th><th>SD</th><th>Vis</th></tr></thead>
                <tbody>{metrics_html}</tbody>
            </table>
        </div>
        <details>
            <summary>View Representative Examples ({count})</summary>
            <div class="ex-content">{examples_html}</div>
        </details>
    </div>
    """

def get_magnitude_desc(z_score):
    if z_score > 3.0:  return "Very High"
    elif z_score > 1.5: return "High"
    elif z_score > 0.5: return "Elevated"
    elif z_score < -3.0: return "Very Low"
    elif z_score < -1.5: return "Low"
    elif z_score < -0.5: return "Reduced"
    return None

def generate_label(sub_df, features, global_means, global_sds):
    # 1. Calculate Accuracy
    
    descriptions = []
    for f in features:
        if global_sds[f] == 0: continue
        z_val = (sub_df[f].mean() - global_means[f]) / global_sds[f]
        desc = get_magnitude_desc(z_val)
        if desc:
            descriptions.append((abs(z_val), f"{desc} {f}"))
            
    descriptions.sort(key=lambda x: x[0], reverse=True)
    top_descs = [d[1] for d in descriptions[:3]]
    
    return "\n".join(top_descs) if top_descs else "Average Stats"
    

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from itertools import combinations

# --- HELPER 1: Filter Data by Observed Difficulty ---
def get_difficulty_slice(df, k):
    """
    Returns a subset of df containing only questions that appear exactly 4 times
    and were answered correctly exactly 'k' times.
    """
    if 'question_id_external' not in df.columns:
        return pd.DataFrame()
    
    # Transform to keep dataframe shape while filtering
    counts = df.groupby('question_id_external')['Correct'].transform('count')
    successes = df.groupby('question_id_external')['Correct'].transform('sum')
    
    return df[(counts == 4) & (successes == k)]

# --- HELPER 2: Stratified t-SNE Plotter ---
def plot_stratified_clusters(viz_df, tsne_proj, unique_clusters, k_levels=[0, 1, 2, 3, 4]):
    """
    Plots the same t-SNE projection multiple times, filtering by observed difficulty (k).
    """
    viz_df = viz_df.copy()
    if 'x' not in viz_df.columns:
        viz_df['x'], viz_df['y'] = tsne_proj[:, 0], tsne_proj[:, 1]
    
    palette_colors = sns.color_palette("turbo", n_colors=len(unique_clusters))
    cluster_cmap = dict(zip(unique_clusters, palette_colors))
    
    n_plots = len(k_levels)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5), sharex=True, sharey=True)
    if n_plots == 1: axes = [axes]

    print(f"Generating Stratified Plots for k={k_levels}...")

    for i, k in enumerate(k_levels):
        ax = axes[i]
        subset = get_difficulty_slice(viz_df, k)
        
        # Background
        ax.scatter(viz_df['x'], viz_df['y'], c='#ecf0f1', s=10, alpha=0.3)
        
        if len(subset) > 0:
            # Foreground
            sns.scatterplot(
                data=subset, x='x', y='y', hue='Cluster', 
                palette=cluster_cmap, s=60, alpha=0.9, 
                ax=ax, legend=False, edgecolor='k', linewidth=0.5
            )
            
            # Annotate
            top_clusters = subset['Cluster'].value_counts().head(3).index
            for c in top_clusters:
                sub_c = subset[subset['Cluster'] == c]
                cx, cy = sub_c['x'].median(), sub_c['y'].median()
                ax.text(cx, cy, f"C{c}", ha='center', va='center', 
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=cluster_cmap.get(c, 'k'), alpha=0.8))

        ax.set_title(f"Difficulty k={k}\n({k}/4 Correct)", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# --- HELPER 3: Pairwise Cluster Comparison (Corrected) ---
def get_pairwise_stats_raw(df, unique_clusters):
    """
    Calculates raw stats (N and Diff) for cluster pairs on SHARED questions.
    Returns a dataframe with columns ['c1', 'c2', 'N', 'Diff'].
    """
    if 'question_id_external' not in df.columns:
        return pd.DataFrame(columns=['c1', 'c2', 'N', 'Diff'])

    # 1. Calculate average accuracy per question per cluster
    q_stats = df.groupby(['Cluster', 'question_id_external'])['Correct'].mean()
    
    results = []
    
    # Iterate all pairs
    for c1, c2 in combinations(unique_clusters, 2):
        if c1 not in q_stats or c2 not in q_stats:
            continue
            
        # Get set of questions in each cluster
        q1 = q_stats.loc[c1]
        q2 = q_stats.loc[c2]
        
        # Find Intersection
        shared_qs = q1.index.intersection(q2.index)
        n_shared = len(shared_qs)
        
        if n_shared > 0:
            acc1 = q1.loc[shared_qs]
            acc2 = q2.loc[shared_qs]
            diff = (acc1 - acc2).mean()
            
            results.append({
                'c1': c1,
                'c2': c2,
                'N': n_shared,
                'Diff': diff
            })
            
    return pd.DataFrame(results)


from matplotlib.colors import ListedColormap

def plot_difficulty_discrimination(viz_df):
    """
    Plots a normalized stacked bar chart showing the difficulty composition of each cluster.
    Answers: "Is Cluster X mostly made of Hard questions or Easy questions?"
    """
    if 'obs_k' not in viz_df.columns:
        return

    # 1. Prepare Data: Crosstab of Cluster vs Difficulty (k)
    # We use 'question_id_external' to count uniquely if possible, otherwise use rows
    # Using rows (responses) is generally fine for response-level clustering
    cross_tab = pd.crosstab(viz_df['Cluster'], viz_df['obs_k'])
    
    # Normalize row-wise to get percentages (so each bar is 100% height)
    cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
    
    # 2. Setup Colors (Matching your previous Red->Green scheme)
    # k=0 (Hard/Red) -> k=4 (Easy/Green)
    custom_colors = ['#c0392b', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']
    cmap = ListedColormap(custom_colors)

    # 3. Plot
    ax = cross_tab_norm.plot(
        kind='bar', 
        stacked=True, 
        figsize=(12, 6), 
        colormap=cmap,
        edgecolor='black',
        linewidth=0.5
    )
    
    # 4. Styling
    plt.title("Cluster 'Difficulty' Discrimination", fontsize=14)
    plt.ylabel("Composition (%)", fontsize=12)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.xticks(rotation=0)
    
    # Legend: Reverse order so k=4 is at top (visually intuitive with the stack)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], [f"k={l} (Correct {l}/4)" for l in labels[::-1]], 
               title="Observed Difficulty", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Add percentage labels inside bars if they are big enough
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', fontsize=9, weight='bold', padding=0)

    plt.tight_layout()
    plt.show()

# --- MAIN FUNCTION ---
def run_failure_modes_dashboard(train_df, exp_info, features=['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic'], test_df=None, min_clusters = 3, plot = True):
    """
    Generates HTML dashboard, t-SNE plot, and Pairwise Cluster Analysis.
    """
    if train_df is None: return
    exp_id, exp_name = exp_info
    
    # --- 1. FIT & PREDICT (Train & Test) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].fillna(0))
    
    # Fit KMeans
    n_clusters = min(min_clusters, len(train_df)//10) if len(train_df) > 50 else 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    train_df = train_df.copy() 
    train_df['Cluster'] = kmeans.fit_predict(X_train)
    
    if test_df is not None:
        test_df = test_df.copy()
        X_test = scaler.transform(test_df[features].fillna(0))
        test_df['Cluster'] = kmeans.predict(X_test)
        viz_df = test_df 
        mode_label = "Train & Test Analysis"
        
        # Pre-calc info for stratification
        if 'question_id_external' in test_df.columns:
            test_df['obs_k'] = test_df.groupby('question_id_external')['Correct'].transform('sum')
            test_df['is_valid_k'] = test_df.groupby('question_id_external')['Correct'].transform('count') == 4
    else:
        viz_df = train_df
        mode_label = "Train Only Analysis"

    # --- 2. HTML DASHBOARD GENERATION ---
    ref_df = train_df
    global_medians = ref_df[features].median()
    global_sds = ref_df[features].std()

    if plot:
    
        html_out = [DASHBOARD_CSS, f'<div class="db-container">']
        html_out.append(f'<div class="db-header"><h2>Run: {exp_name} (ID: {exp_id})</h2><small>{mode_label}</small></div>')

        stat_cards = "".join([f'<div class="db-stat-card"><div class="db-stat-val">{global_medians[f]:.3f}</div><div class="db-stat-label">{f}</div></div>' for f in features])
        html_out.append(f'<div class="db-stat-grid">{stat_cards}</div>')

        unique_clusters = sorted(train_df['Cluster'].unique())
        train_acc_map = {} # Store for label generation later

        for c in unique_clusters:
            sub_train = train_df[train_df['Cluster'] == c]
            sub_test = test_df[test_df['Cluster'] == c] if test_df is not None else pd.DataFrame()
            
            # Stats
            n_train = len(sub_train)
            acc_train = (sub_train['Correct'].sum() / n_train * 100) if n_train > 0 else 0
            train_acc_map[c] = acc_train
            
            n_test = len(sub_test)
            if n_test > 0:
                acc_test = (sub_test['Correct'].sum() / n_test * 100)
                test_info = f"<span style='color:#2c3e50'>Test: N={n_test} ({acc_test:.1f}%)</span>"
                acc_color = "var(--success)" if acc_test >= 80 else ("var(--warning)" if acc_test >= 50 else "var(--danger)")
                bar_width = acc_test
            else:
                acc_test = 0
                test_info = "<span style='color:#95a5a6'>Test: N=0</span>"
                acc_color = "#bdc3c7"
                bar_width = 0

            # Difficulty Bar
            diff_stats_html = ""
            if n_test > 0 and 'obs_k' in sub_test.columns:
                valid_sub = sub_test[sub_test['is_valid_k'] == True]
                if len(valid_sub) > 0:
                    k_counts = valid_sub['obs_k'].value_counts().sort_index()
                    total_valid = len(valid_sub)
                    k_colors = {0: '#c0392b', 1: '#e67e22', 2: '#f1c40f', 3: '#3498db', 4: '#2ecc71'}
                    diff_bars = []
                    for k_val in range(5):
                        count = k_counts.get(k_val, 0)
                        if count > 0:
                            pct = (count / total_valid) * 100
                            diff_bars.append(f'<div style="width:{pct}%; background:{k_colors[k_val]}; height:8px;" title="k={k_val}: {pct:.1f}%"></div>')
                    
                    diff_stats_html = f"""
                    <div style="margin-top:10px; font-size:0.8em; color:#7f8c8d;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:2px;"><span>Difficulty Breakdown (Test)</span> <span style="font-size:0.9em">Observed Correctness</span></div>
                        <div style="display:flex; width:100%; border-radius:3px; overflow:hidden; background:#eee;">{''.join(diff_bars)}</div>
                        <div style="display:flex; justify-content:space-between; font-size:0.7em; margin-top:2px;">
                            <span style="color:#c0392b">Hard (0/4)</span><span style="color:#2ecc71">Easy (4/4)</span>
                        </div>
                    </div>
                    """

            # Metrics Table
            metrics_rows = ""
            for f in features:
                mu_train = sub_train[f].mean() if n_train > 0 else 0
                mu_test = sub_test[f].mean() if n_test > 0 else 0
                sd_test = sub_test[f].std() if n_test > 0 else 0
                
                if n_test > 0 and global_sds[f] != 0:
                    shift_val = (mu_test - global_medians[f]) / global_sds[f]
                elif n_train > 0 and global_sds[f] != 0:
                    shift_val = (mu_train - global_medians[f]) / global_sds[f]
                else:
                    shift_val = 0
                
                bar_color_m = "#c0392b" if shift_val > 1.0 else ("#2980b9" if shift_val < -1.0 else "#95a5a6")
                vis_width = min(abs(shift_val) * 15, 50)
                vis_left = "50%" if shift_val > 0 else f"{50 - vis_width}%"
                
                metrics_rows += f"""
                <tr>
                    <td><b>{f}</b></td>
                    <td style="color:{bar_color_m}">{shift_val:+.2f} SD</td>
                    <td style="background:#f8f9fa">{mu_train:.3f}</td> 
                    <td style="font-weight:bold">{mu_test:.3f}</td>
                    <td style="color:#7f8c8d">{sd_test:.2f}</td>
                    <td style="width:100px"><div class="shift-bar-container"><div class="shift-bar" style="left:50%; width:1px; background:#ccc;"></div><div class="shift-bar" style="left:{vis_left}; width:{vis_width}%; background:{bar_color_m}; opacity:0.8;"></div></div></td>
                </tr>
                """

            ex_source = sub_test if n_test > 0 else sub_train
            ex_label = "Test Examples" if n_test > 0 else "Train Examples (No Test Data)"
            
            ex_rows = ""
            for i, (_, row) in enumerate(ex_source.head(5).iterrows()):
                badge = '<span class="status-badge-pass">✅ Correct</span>' if row['Correct'] else '<span class="status-badge-fail">❌ Incorrect</span>'
                ex_rows += f'<div style="margin-bottom:10px;"><strong>[Ex {i+1}] {badge}</strong><br>Q: {row.get("question_text")}<br><div class="pred-box">Pred: {row.get("full_trace_text")}</div></div><hr>'

            card_html = f"""
            <div class="cluster-card">
                <div class="cluster-head" style="border-left: 5px solid {acc_color}">
                    <div class="cluster-title">CLUSTER {c}</div>
                    <div class="acc-container" style="flex:1; justify-content:flex-end; gap:15px">
                        <span style="color:#7f8c8d; font-size:0.9em">Train: N={n_train} ({acc_train:.1f}%)</span>
                        <span style="border-left:1px solid #ccc; height:15px"></span>
                        {test_info}
                        <div class="acc-bar-bg" style="width:60px"><div class="acc-bar-fill" style="width: {bar_width}%; background: {acc_color};"></div></div>
                    </div>
                </div>
                <div style="padding: 15px;">
                    {diff_stats_html}
                    <table class="metric-table">
                        <thead><tr><th>Metric</th><th>Shift (Test)</th><th style="background:#f8f9fa">Train μ</th><th>Test μ</th><th>Test σ</th><th>Vis</th></tr></thead>
                        <tbody>{metrics_rows}</tbody>
                    </table>
                </div>
                <details>
                    <summary>View {ex_label} ({len(ex_source)})</summary>
                    <div class="ex-content">{ex_rows}</div>
                </details>
            </div>
            """
            html_out.append(card_html)
            
        html_out.append("</div>")
        display(HTML("".join(html_out)))

        # --- 3. t-SNE VISUALIZATION ---
        print(f"Generating t-SNE on {mode_label}...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(viz_df)-1), random_state=42, init='pca')
        proj = tsne.fit_transform(scaler.transform(viz_df[features].fillna(0)))
        viz_df['x'], viz_df['y'] = proj[:, 0], proj[:, 1]
        
        palette_colors = sns.color_palette("turbo", n_colors=len(unique_clusters))
        cluster_cmap = dict(zip(unique_clusters, palette_colors))

        plt.figure(figsize=(14, 10))
        sns.scatterplot(data=viz_df[viz_df['Correct']==1], x='x', y='y', color='#ecf0f1', s=60, alpha=0.3, linewidth=0, zorder=0)
        sns.scatterplot(data=viz_df[viz_df['Correct']==0], x='x', y='y', hue='Cluster', palette=cluster_cmap, s=120, alpha=0.9, edgecolor='k', legend=False, zorder=10)

        global_means = train_df[features].mean()
        for c in unique_clusters:
            sub_train = train_df[train_df['Cluster'] == c]
            n_train = len(sub_train)
            acc_train = sub_train['Correct'].mean() if n_train > 0 else 0
            
            label_text = f"C{c}\nTrain: N={n_train} | {acc_train:.0%}"
            if test_df is not None:
                sub_test = test_df[test_df['Cluster'] == c]
                n_test = len(sub_test)
                acc_test = sub_test['Correct'].mean() if n_test > 0 else 0
                label_text += f"\nTest: N={n_test} | {acc_test:.0%}"

            label_text += "\n" + generate_label(sub_train, features, global_means, global_sds)

            sub_viz = viz_df[viz_df['Cluster'] == c]
            failures = sub_viz[sub_viz['Correct'] == 0]
            target = failures if len(failures) > 3 else sub_viz
            if len(target) == 0: continue
                
            cx, cy = target['x'].median(), target['y'].median()
            plt.text(cx, cy, label_text, horizontalalignment='center', verticalalignment='center',
                    fontsize=8, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=cluster_cmap[c], lw=2, alpha=0.9), zorder=20)
        plt.title(f"Response Clustering Analysis", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # --- 4. DIFFICULTY STRATIFICATION ---
        if test_df is not None and 'obs_k' in viz_df.columns:
            print("Generating Difficulty Stratification Plots...")
            plot_stratified_clusters(
                viz_df=viz_df, 
                tsne_proj=proj,
                unique_clusters=unique_clusters,
                k_levels=[0, 1, 2, 3, 4] 
            )

            print("Generating Discrimination Bar Chart...")
            plot_difficulty_discrimination(viz_df)

        # --- 5. CLUSTER PAIRWISE ANALYSIS (FIXED) ---
        print("\n--- Pairwise Cluster Comparison (Same Question Overlap) ---")
    
        # Calculate Raw Stats (N, Diff) for Train and Test separately
        train_raw = get_pairwise_stats_raw(train_df, unique_clusters)
        test_raw = pd.DataFrame(columns=['c1', 'c2', 'N', 'Diff'])
        if test_df is not None:
            test_raw = get_pairwise_stats_raw(test_df, unique_clusters)

        # Merge ON CLUSTER IDS (c1, c2) to align rows correctly
        if not train_raw.empty or not test_raw.empty:
            merged = pd.merge(
                train_raw, 
                test_raw, 
                on=['c1', 'c2'], 
                how='outer', 
                suffixes=('_Train', '_Test')
            ).fillna(0)
            
            # Now construct the display label row-by-row
            pairs = []
            for _, row in merged.iterrows():
                c1, c2 = int(row['c1']), int(row['c2'])
                acc1 = train_acc_map.get(c1, 0)
                acc2 = train_acc_map.get(c2, 0)
                
                pairs.append(f"C{c1} ({acc1:.1f}%) vs C{c2} ({acc2:.1f}%)")
            
            merged['Pair'] = pairs
            
            # Rename and Select Final Columns
            merged = merged.rename(columns={
                'N_Train': 'Train_N', 'Diff_Train': 'Train_Diff',
                'N_Test': 'Test_N', 'Diff_Test': 'Test_Diff'
            })
            
            final_df = merged[['Pair', 'Train_N', 'Train_Diff', 'Test_N', 'Test_Diff']]
            
            # Stylize
            def color_diff(val):
                if val == 0: return 'color: #95a5a6' 
                color = '#c0392b' if val < -0.1 else ('#27ae60' if val > 0.1 else 'black')
                return f'color: {color}; font-weight: bold'

            display(
                final_df.style.format({
                    'Train_Diff': '{:+.1%}', 
                    'Test_Diff': '{:+.1%}',
                    'Train_N': '{:.0f}',
                    'Test_N': '{:.0f}'
                })
                .applymap(color_diff, subset=['Train_Diff', 'Test_Diff'])
                .set_properties(**{'text-align': 'center'})
                .set_table_styles([
                    # CSS Hack to hide index on pandas < 1.4.0
                    {'selector': '.row_heading', 'props': [('display', 'none')]}, 
                    {'selector': '.blank', 'props': [('display', 'none')]},
                    dict(selector='th', props=[('text-align', 'center')])
                ])
                .set_caption("Accuracy Delta on SHARED questions (Positive = First Cluster is Better)")
            )

        else:
            print("No shared questions found between clusters (or no ID column).")

    return PredictorArtifact(
        model=kmeans,
        scaler=scaler,
        features=features,
        mode='clustering',
        label_map={}
    )



    
def run_failure_modes_dashboard_old(train_df, exp_info, features=['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic'], test_df=None, min_clusters = 3):
    """
    Generates the HTML dashboard and t-SNE plot.
    Fits clustering on train_df, but calculates and reports stats for BOTH Train and Test.
    """
    if train_df is None: return
    exp_id, exp_name = exp_info
    
    # --- 1. FIT & PREDICT (Train & Test) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features].fillna(0))
    
    # Fit KMeans on Train
    n_clusters = min(min_clusters, len(train_df)//10) if len(train_df) > 50 else 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    train_df = train_df.copy() # Avoid SettingWithCopy warnings
    train_df['Cluster'] = kmeans.fit_predict(X_train)
    
    # Predict on Test (if exists)
    if test_df is not None:
        test_df = test_df.copy()
        X_test = scaler.transform(test_df[features].fillna(0))
        test_df['Cluster'] = kmeans.predict(X_test)
        viz_df = test_df # We still use Test for the t-SNE plot
        mode_label = "Train & Test Analysis"
    else:
        viz_df = train_df
        mode_label = "Train Only Analysis"

    # --- 2. HTML DASHBOARD GENERATION ---
    # We use Test stats for the "Global" baseline if available, else Train
    ref_df = train_df
    global_medians = ref_df[features].median()
    global_sds = ref_df[features].std()
    
    html_out = [DASHBOARD_CSS, f'<div class="db-container">']
    html_out.append(f'<div class="db-header"><h2>Run: {exp_name} (ID: {exp_id})</h2><small>{mode_label}</small></div>')

    # Global Stats Cards (Reference Set)
    stat_cards = "".join([f'<div class="db-stat-card"><div class="db-stat-val">{global_medians[f]:.3f}</div><div class="db-stat-label">{f}</div></div>' for f in features])
    html_out.append(f'<div class="db-stat-grid">{stat_cards}</div>')

    # Iterate through clusters found in TRAIN (since that's the ground truth definition)
    unique_clusters = sorted(train_df['Cluster'].unique())
    
    for c in unique_clusters:
        # Get subsets
        sub_train = train_df[train_df['Cluster'] == c]
        sub_test = test_df[test_df['Cluster'] == c] if test_df is not None else pd.DataFrame()
        
        # --- A. HEADLINE STATS (TRAIN vs TEST) ---
        # Train Stats
        n_train = len(sub_train)
        acc_train = (sub_train['Correct'].sum() / n_train * 100) if n_train > 0 else 0
        
        # Test Stats
        n_test = len(sub_test)
        if n_test > 0:
            acc_test = (sub_test['Correct'].sum() / n_test * 100)
            test_info = f"<span style='color:#2c3e50'>Test: N={n_test} ({acc_test:.1f}%)</span>"
            # Color code the Test accuracy bar
            acc_color = "var(--success)" if acc_test >= 80 else ("var(--warning)" if acc_test >= 50 else "var(--danger)")
            bar_width = acc_test
        else:
            acc_test = 0
            test_info = "<span style='color:#95a5a6'>Test: N=0</span>"
            acc_color = "#bdc3c7"
            bar_width = 0

        # --- B. METRICS TABLE (SIDE-BY-SIDE) ---
        metrics_rows = ""
        for f in features:
            # Train values
            mu_train = sub_train[f].mean() if n_train > 0 else 0
            
            # Test values (or 0 if empty)
            mu_test = sub_test[f].mean() if n_test > 0 else 0
            sd_test = sub_test[f].std() if n_test > 0 else 0
            
            # Shift Calculation (Based on TEST if available, to see drift)
            if n_test > 0 and global_sds[f] != 0:
                shift_val = (mu_test - global_medians[f]) / global_sds[f]
            elif n_train > 0 and global_sds[f] != 0:
                shift_val = (mu_train - global_medians[f]) / global_sds[f]
            else:
                shift_val = 0
            
            bar_color_m = "#c0392b" if shift_val > 1.0 else ("#2980b9" if shift_val < -1.0 else "#95a5a6")
            vis_width = min(abs(shift_val) * 15, 50)
            vis_left = "50%" if shift_val > 0 else f"{50 - vis_width}%"
            
            # Row with Train vs Test Mean columns
            metrics_rows += f"""
            <tr>
                <td><b>{f}</b></td>
                <td style="color:{bar_color_m}">{shift_val:+.2f} SD</td>
                <td style="background:#f8f9fa">{mu_train:.3f}</td> 
                <td style="font-weight:bold">{mu_test:.3f}</td>
                <td style="color:#7f8c8d">{sd_test:.2f}</td>
                <td style="width:100px"><div class="shift-bar-container"><div class="shift-bar" style="left:50%; width:1px; background:#ccc;"></div><div class="shift-bar" style="left:{vis_left}; width:{vis_width}%; background:{bar_color_m}; opacity:0.8;"></div></div></td>
            </tr>
            """

        # --- C. EXAMPLES (From TEST Set if possible) ---
        ex_source = sub_test if n_test > 0 else sub_train
        ex_label = "Test Examples" if n_test > 0 else "Train Examples (No Test Data)"
        
        ex_rows = ""
        for i, (_, row) in enumerate(ex_source.head(5).iterrows()):
            badge = '<span class="status-badge-pass">✅ Correct</span>' if row['Correct'] else '<span class="status-badge-fail">❌ Incorrect</span>'
            ex_rows += f'<div style="margin-bottom:10px;"><strong>[Ex {i+1}] {badge}</strong><br>Q: {row.get("question_text")}<br><div class="pred-box">Pred: {row.get("full_trace_text")}</div></div><hr>'

        # Custom Cluster Card HTML injection
        card_html = f"""
        <div class="cluster-card">
            <div class="cluster-head" style="border-left: 5px solid {acc_color}">
                <div class="cluster-title">CLUSTER {c}</div>
                <div class="acc-container" style="flex:1; justify-content:flex-end; gap:15px">
                    <span style="color:#7f8c8d; font-size:0.9em">Train: N={n_train} ({acc_train:.1f}%)</span>
                    <span style="border-left:1px solid #ccc; height:15px"></span>
                    {test_info}
                    <div class="acc-bar-bg" style="width:60px"><div class="acc-bar-fill" style="width: {bar_width}%; background: {acc_color};"></div></div>
                </div>
            </div>
            <div style="padding: 15px;">
                <table class="metric-table">
                    <thead><tr><th>Metric</th><th>Shift (Test)</th><th style="background:#f8f9fa">Train μ</th><th>Test μ</th><th>Test σ</th><th>Vis</th></tr></thead>
                    <tbody>{metrics_rows}</tbody>
                </table>
            </div>
            <details>
                <summary>View {ex_label} ({len(ex_source)})</summary>
                <div class="ex-content">{ex_rows}</div>
            </details>
        </div>
        """
        html_out.append(card_html)
        
    html_out.append("</div>")
    display(HTML("".join(html_out)))

    # --- 3. t-SNE VISUALIZATION (Plot VIZ set, usually Test) ---
    print(f"Generating t-SNE on {mode_label}...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(viz_df)-1), random_state=42, init='pca')
    proj = tsne.fit_transform(scaler.transform(viz_df[features].fillna(0)))
    viz_df['x'], viz_df['y'] = proj[:, 0], proj[:, 1]
    
    unique_clusters = sorted(viz_df['Cluster'].unique())
    palette_colors = sns.color_palette("turbo", n_colors=len(unique_clusters))
    cluster_cmap = dict(zip(unique_clusters, palette_colors))
    cluster_labels_map = {} # Can implement if needed

    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=viz_df[viz_df['Correct']==1], x='x', y='y', color='#ecf0f1', s=60, alpha=0.3, linewidth=0, zorder=0)
    sns.scatterplot(data=viz_df[viz_df['Correct']==0], x='x', y='y', hue='Cluster', palette=cluster_cmap, s=120, alpha=0.9, edgecolor='k', legend=False, zorder=10)

    global_means = train_df[features].mean()
    for c in unique_clusters:
        # 1. Get stats for both sets
        sub_train = train_df[train_df['Cluster'] == c]
        n_train = len(sub_train)
        acc_train = sub_train['Correct'].mean() if n_train > 0 else 0
        
        label_text = f"C{c}\nTrain: N={n_train} | {acc_train:.0%}"
        
        if test_df is not None:
            sub_test = test_df[test_df['Cluster'] == c]
            n_test = len(sub_test)
            acc_test = sub_test['Correct'].mean() if n_test > 0 else 0
            label_text += f"\nTest: N={n_test} | {acc_test:.0%}"

        label_text += "\n" + generate_label(sub_train, features, global_means, global_sds)

        # 2. Position label based on the Visualization Set (to match the scatter plot)
        sub_viz = viz_df[viz_df['Cluster'] == c]
        failures = sub_viz[sub_viz['Correct'] == 0]
        
        # Prefer centering on failures if they exist, otherwise center of cluster
        target = failures if len(failures) > 3 else sub_viz
        if len(target) == 0: continue
            
        cx, cy = target['x'].median(), target['y'].median()
        
        plt.text(cx, cy, label_text, horizontalalignment='center', verticalalignment='center',
                 fontsize=8, fontweight='bold', color='black',
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=cluster_cmap[c], lw=2, alpha=0.9), zorder=20)
    plt.title(f"Response Clustering Analysis", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return PredictorArtifact(
        model=kmeans,
        scaler=scaler,
        features=features,
        mode='clustering',
        label_map={}
    )

# --- 3. METRIC DIAGNOSTICS & FEATURE SEARCH ---

def plot_metric_diagnostics(df, metrics):
    """Plots KDEs for each metric separated by correctness."""
    valid_metrics = [m for m in metrics if df[m].var() > 1e-5]
    fig, axes = plt.subplots(1, len(valid_metrics), figsize=(4 * len(valid_metrics), 4))
    if len(valid_metrics) == 1: axes = [axes]

    for i, m in enumerate(valid_metrics):
        sns.kdeplot(data=df, x=m, hue='Correct', fill=True, palette={0: "#C0392B", 1: "#2E86C1"}, alpha=0.3, warn_singular=False, ax=axes[i])
        axes[i].set_title(f"{m} Separation")
        axes[i].set_ylabel("")
        axes[i].get_yaxis().set_ticks([])

    plt.suptitle(f"Metric Diagnostics (Red=Fail, Blue=Success)", y=1.05)
    plt.tight_layout()
    plt.show()

def run_feature_subset_search(df, features=['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic', 'Length'], topn = 10):
    """Iteratively tests all feature combinations to find the best predictors."""
    display(Markdown("### Feature Subset Analysis (Exhaustive Search)"))
    
    # Prep
    train_q, test_q = train_test_split(df['question_text'].unique(), test_size=0.5, random_state=42)
    train_df = df[df['question_text'].isin(train_q)].copy().fillna(0)
    test_df  = df[df['question_text'].isin(test_q)].copy().fillna(0)
    
    scaler = StandardScaler()
    X_train_full = pd.DataFrame(scaler.fit_transform(train_df[features]), columns=features)
    X_test_full  = pd.DataFrame(scaler.transform(test_df[features]), columns=features)
    y_train, y_test = train_df['Correct'], test_df['Correct']

    results = []
    for r in range(1, len(features) + 1):
        for subset in itertools.combinations(features, r):
            cols = list(subset)
            clf = LogisticRegression(class_weight="balanced", random_state=42, penalty='l2')
            clf.fit(X_train_full[cols], y_train)
            probs = clf.predict_proba(X_test_full[cols])[:, 1]
            try: auc = roc_auc_score(y_test, probs)
            except: auc = 0.5
            results.append({'Num_Features': len(cols), 'Features': ", ".join(cols), 'AUC': auc})

    res_df = pd.DataFrame(results).sort_values('AUC', ascending=False).reset_index(drop=True)
    res_df['% of Max'] = (res_df['AUC'] / res_df.iloc[0]['AUC'])
    
    display(res_df.head(topn).style.bar(subset=['AUC'], color='#5fba7d', vmin=0.5, vmax=1.0).format("{:.4f}", subset=['AUC']).format("{:.1%}", subset=['% of Max']))
    
    # Complexity Plot
    plt.figure(figsize=(8, 4))
    sns.lineplot(x='Num_Features', y='AUC', data=res_df, marker='o', estimator='max', errorbar=None, color='navy')
    plt.title("Best Achievable AUC by Subset Size")
    plt.grid(True, linestyle='--')
    plt.show()

# --- 4. COMPREHENSIVE MODELING ---

def run_detailed_logistic_regression(train_df, test_df=None, features=['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic', 'Consistency', 'Semantic', 'Length'], plot = True):
    """
    Full Logistic Regression report with coefficients and visualizations.
    UPDATED: Now accepts explicit train/test split. Falls back to internal split if test_df is None.
    """
    if plot:
        display(Markdown("### Full Logistic Regression Analysis"))
    
    # Filter only features present in DF
    valid_features = [f for f in features if f in train_df.columns]
    
    # --- LOGIC CHANGE START: Handle explicit vs internal split ---
    if test_df is None:
        display(Markdown("*⚠️ No test set provided. Performing internal 50/50 split on training data.*"))
        # Internal split logic (Original behavior)
        train_q, test_q = train_test_split(train_df['question_text'].unique(), test_size=0.5, random_state=42)
        X_train_df = train_df[train_df['question_text'].isin(train_q)].copy().fillna(0)
        X_test_df  = train_df[train_df['question_text'].isin(test_q)].copy().fillna(0)
    else:
        # Use provided sets
        X_train_df = train_df.copy().fillna(0)
        X_test_df  = test_df.copy().fillna(0)
    # --- LOGIC CHANGE END ---

    # Use X_train_df/X_test_df for scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df[valid_features])
    X_test  = scaler.transform(X_test_df[valid_features])
    
    y_train, y_test = X_train_df['Correct'], X_test_df['Correct']
    
    clf = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    
    # 1. Feature Importance
    coef_df = pd.DataFrame({'Feature': valid_features, 'Weight': clf.coef_[0]}).sort_values('Weight', ascending=False)

    if plot: 
        plt.figure(figsize=(10, 4))
        
        # FIX: Added hue='Feature' and legend=False to silence warning
        sns.barplot(
            x='Weight', y='Feature', data=coef_df, 
            hue='Feature', legend=False,
            palette=['forestgreen' if x > 0 else 'crimson' for x in coef_df['Weight']]
        )
        plt.title("Feature Weights (Direction & Magnitude)")
        plt.axvline(0, color='black')
        plt.show()
        
        # 2. Performance
        print(classification_report(y_test, (probs > 0.5).astype(int)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(confusion_matrix(y_test, (probs > 0.5).astype(int)), annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title("Confusion Matrix")
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_score(y_test, probs):.2f}')
        ax2.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax2.set_title("ROC Curve")
        ax2.legend()
        plt.show()

    return PredictorArtifact(
        model=clf,
        scaler=scaler,
        features=valid_features,
        mode='classifier'
    )





import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, 
    accuracy_score, f1_score, precision_score, recall_score
)
from IPython.display import display, HTML

def run_expanded_comparison(train_df, classifier_artifacts, cluster_predictors, metrics_to_use, test_df=None):
    """
    Generates summary statistics for an arbitrary number of classifiers and cluster baselines.
    
    Args:
        train_df (pd.DataFrame): Training data containing 'Correct' label and metric columns.
        classifier_artifacts (dict): Dict of {name: predict_proba_function}. 
                                     Functions must accept a DF and return (probs, labels).
        cluster_predictors (dict): Dict of {name: cluster_function}. 
                                   Functions must accept a DF and return (model, labels).
        metrics_to_use (list): List of column names in DF to treat as individual feature baselines.
        test_df (pd.DataFrame, optional): Test data.
        
    Returns:
        pd.DataFrame: A comprehensive summary table of all models and metrics.
    """
    
    # --- 1. SETUP & NORMALIZATION ---
    
    # Track all "models" (classifiers, clusterers, and raw metrics)
    model_names = list(classifier_artifacts.keys()) + list(cluster_predictors.keys()) + metrics_to_use
    
    # Calculate Normalization Bounds & Direction on TRAIN
    norm_bounds = {}
    metric_flip = {}

    for m in metrics_to_use:
        if m in train_df.columns:
            s = train_df[m].fillna(train_df[m].mean())
            norm_bounds[m] = (s.min(), s.max())
            
            # Check correlation: if negative, flip so 1.0 = Good
            corr = s.corr(train_df['Correct'])
            metric_flip[m] = (corr < 0)

    def apply_normalization(df_in, metric_name):
        """Normalizes a raw metric column into a 0-1 probability proxy."""
        if metric_name not in df_in.columns: return
        
        s = df_in[metric_name].fillna(train_df[metric_name].mean())
        m_min, m_max = norm_bounds.get(metric_name, (s.min(), s.max()))
        is_inverse = metric_flip.get(metric_name, False)
        
        if m_max == m_min: 
            df_in[metric_name] = 0.5
            return

        norm_val = (s - m_min) / (m_max - m_min)
        final = (1 - norm_val) if is_inverse else norm_val
        df_in[metric_name] = np.clip(final, 0.0, 1.0)

    def compute_ece(probs, y_true, n_bins=10):
        """Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
            if not np.any(mask): continue
            bin_prob = np.mean(probs[mask])
            bin_acc = np.mean(y_true[mask])
            ece += np.abs(bin_prob - bin_acc) * (np.sum(mask) / len(probs))
        return ece

    # --- 2. GENERATE PREDICTIONS ---
    
    dfs_to_process = [('Train', train_df)]
    if test_df is not None:
        dfs_to_process.append(('Test', test_df))

    for name, df_curr in dfs_to_process:
        # A. Run Classifiers
        for model_name, predict_func in classifier_artifacts.items():
            if model_name not in df_curr.columns:
                try:
                    probs, _ = predict_func(df_curr)
                    df_curr[model_name] = probs
                except Exception:
                    df_curr[model_name] = 0.5

        # B. Run Cluster Baselines
        for clust_name, clust_func in cluster_predictors.items():
            col_name = f"{clust_name}_Prob"
            label_col = f"{clust_name}_Label"
            if col_name not in df_curr.columns:
                try:
                    _, labels = clust_func(df_curr)
                    df_curr[label_col] = labels
                    
                    # Compute historical accuracy per cluster on TRAIN
                    # Note: We always map using TRAIN statistics to prevent leakage
                    if name == 'Train':
                        cluster_accs = train_df.groupby(label_col)['Correct'].mean()
                        # Store this mapping for the Test set if needed (omitted for brevity, recalculating on train)
                    else:
                        # Re-derive train mapping for consistency
                        # In prod, you'd pass the fitted cluster object
                        _, train_labels = clust_func(train_df)
                        temp_train = train_df.copy()
                        temp_train[label_col] = train_labels
                        cluster_accs = temp_train.groupby(label_col)['Correct'].mean()

                    df_curr[col_name] = df_curr[label_col].map(cluster_accs).fillna(0.5)
                    
                    # Register this new prob column as a model to evaluate
                    if col_name not in model_names:
                        model_names.append(col_name)
                        if clust_name in model_names: model_names.remove(clust_name) # clean up
                        
                except Exception:
                    df_curr[col_name] = 0.5

        # C. Normalize Raw Metrics
        for m in metrics_to_use:
            apply_normalization(df_curr, m)

    # --- 3. COMPUTE EXTENDED STATISTICS ---
    
    stats_rows = []
    
    # Define metric functions
    # For threshold-dependent metrics (Acc, F1, etc.), we assume threshold=0.5
    metrics_calc = {
        'AUROC': lambda y, p: roc_auc_score(y, p),
        'ECE': lambda y, p: compute_ece(p, y),
        'Brier (MSE)': lambda y, p: brier_score_loss(y, p),
        'Log Loss': lambda y, p: log_loss(y, p, labels=[0,1]),
        'Accuracy': lambda y, p: accuracy_score(y, (p > 0.5).astype(int)),
        'F1-Score': lambda y, p: f1_score(y, (p > 0.5).astype(int)),
        'Precision': lambda y, p: precision_score(y, (p > 0.5).astype(int), zero_division=0),
        'Recall': lambda y, p: recall_score(y, (p > 0.5).astype(int))
    }

    # Only evaluate columns that actually exist
    valid_models = [m for m in model_names if m in train_df.columns]

    for model in valid_models:
        row = {'Model': model}
        
        # Calculate for Train
        for stat_name, func in metrics_calc.items():
            try:
                val = func(train_df['Correct'].values, train_df[model].values)
                row[f'Train {stat_name}'] = val
            except:
                row[f'Train {stat_name}'] = np.nan

        # Calculate for Test (if exists)
        if test_df is not None:
            for stat_name, func in metrics_calc.items():
                try:
                    val = func(test_df['Correct'].values, test_df[model].values)
                    row[f'Test {stat_name}'] = val
                except:
                    row[f'Test {stat_name}'] = np.nan
        
        stats_rows.append(row)

    # --- 4. DISPLAY & RETURN ---
    
    if not stats_rows:
        print("⚠️ No valid models to evaluate.")
        return pd.DataFrame()

    results_df = pd.DataFrame(stats_rows).set_index('Model')
    
    # Create display version with highlighting
    # High is Good: AUROC, Acc, F1, Precision, Recall
    # Low is Good: ECE, Brier, Log Loss
    
    high_good = [c for c in results_df.columns if any(x in c for x in ['AUROC', 'Accuracy', 'F1', 'Precision', 'Recall'])]
    low_good = [c for c in results_df.columns if any(x in c for x in ['ECE', 'Brier', 'Log Loss'])]

    print(f"Evaluated {len(valid_models)} models/features.")
    
    styled = (results_df.style
              .background_gradient(cmap='Greens', subset=high_good)
              .background_gradient(cmap='Reds', subset=low_good)
              .format("{:.4f}"))
    
    display(HTML("<h3>📊 Expanded Model Performance Summary</h3>"))
    display(HTML(styled.to_html()))
    
    return results_df




# ==============================================================================
#  ADD THIS FUNCTION TO THE BOTTOM OF YOUR CODE
# ==============================================================================

def run_comparison_dashboard(train_df, classifier_artifact, cluster_predictor, metrics_to_use, test_df=None):
    """
    Generates a comprehensive HTML dashboard comparing Logistic Regression, 
    Clustering Baselines, and Individual Metrics.
    
    UPGRADES:
    - Accepts explicit Train and Test sets.
    - Normalizes metrics based on Train statistics (prevents leakage).
    - Automatically detects metric direction (Sign Flip) via correlation.
    - Reports Train vs Test performance to spot overfitting.
    - Runs Cluster Deep Dive on Test data (if available).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from IPython.display import display, HTML
    import numpy as np
    import pandas as pd
    
    # --- 1. CONFIGURATION & HELPERS ---
    
    MODEL_ORDER = ['Logistic Fusion', 'Cluster Baseline'] + metrics_to_use

    # Calculate Normalization Bounds & DIRECTION on TRAIN to avoid leakage
    norm_bounds = {}
    metric_flip = {} # Stores whether a metric is inversely correlated with accuracy

    for m in metrics_to_use:
        if m in train_df.columns:
            # Handle potential NaNs by filling with mean for calculation
            s = train_df[m].fillna(train_df[m].mean())
            norm_bounds[m] = (s.min(), s.max())
            
            # Check correlation with Ground Truth. 
            # If Corr < 0, High Value = Bad Output (Inverse) -> Needs Flip
            corr = s.corr(train_df['Correct'])
            metric_flip[m] = (corr < 0)

    def apply_normalization(df_in, metric_name):
        """Transforms raw metric into a 0-1 Confidence Probability proxy using TRAIN bounds."""
        if metric_name not in df_in.columns: return
        
        s = df_in[metric_name].fillna(train_df[metric_name].mean())
        m_min, m_max = norm_bounds.get(metric_name, (s.min(), s.max()))
        
        # Retrieve auto-detected direction (Default to False if unknown)
        is_inverse = metric_flip.get(metric_name, False)
        
        if m_max == m_min: 
            df_in[metric_name] = 0.5
            return

        # Apply Min-Max scaling constrained to [0, 1]
        norm_val = (s - m_min) / (m_max - m_min)
        final = (1 - norm_val) if is_inverse else norm_val
        df_in[metric_name] = np.clip(final, 0.0, 1.0)

    def compute_ece(probs, y_true, n_bins=10):
        """Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
            if not np.any(mask): continue
            bin_prob = np.mean(probs[mask])
            bin_acc = np.mean(y_true[mask])
            ece += np.abs(bin_prob - bin_acc) * (np.sum(mask) / len(probs))
        return ece

    # --- 2. PREPARE PREDICTIONS (Train & Test) ---
    
    dfs_to_process = [('Train', train_df)]
    if test_df is not None:
        dfs_to_process.append(('Test', test_df))

    for name, df_curr in dfs_to_process:
        # Logistic Fusion
        if 'Logistic Fusion' not in df_curr.columns:
            try:
                probs_fusion, _ = classifier_artifact(df_curr)
                df_curr['Logistic Fusion'] = probs_fusion
            except Exception as e:
                df_curr['Logistic Fusion'] = 0.5

        # Cluster Baseline
        if 'Cluster Baseline' not in df_curr.columns:
            try:
                _, labels = cluster_predictor(df_curr)
                df_curr['Cluster_Label'] = labels
                # Note: We map cluster accuracy from TRAIN (or self if naive)
                cluster_accs = train_df.groupby('Cluster_Label')['Correct'].mean()
                df_curr['Cluster Baseline'] = df_curr['Cluster_Label'].map(cluster_accs).fillna(0.5)
            except Exception as e:
                df_curr['Cluster_Label'] = "Unknown"
                df_curr['Cluster Baseline'] = 0.5
        
        # Normalize Metrics
        for m in metrics_to_use:
            apply_normalization(df_curr, m)

    # --- 3. GLOBAL DATASET PERFORMANCE (COMPARATIVE) ---
    
    stats_rows = []
    valid_models = [m for m in MODEL_ORDER if m in train_df.columns]
    
    for model in valid_models:
        row = {'Model': model}
        
        # Train Stats
        try:
            row['Train AUROC'] = roc_auc_score(train_df['Correct'], train_df[model])
            row['Train ECE'] = compute_ece(train_df[model].values, train_df['Correct'].values)
            row['Train MSE'] = brier_score_loss(train_df['Correct'], train_df[model])
        except: pass
        
        # Test Stats
        if test_df is not None:
            try:
                row['Test AUROC'] = roc_auc_score(test_df['Correct'], test_df[model])
                row['Test ECE'] = compute_ece(test_df[model].values, test_df['Correct'].values)
                row['Test MSE'] = brier_score_loss(test_df['Correct'], test_df[model])
            except: pass
            
        stats_rows.append(row)

    if stats_rows:
        g_df = pd.DataFrame(stats_rows).set_index('Model')
        
        # visual styling
        subset_auroc = ['Train AUROC', 'Test AUROC'] if test_df is not None else ['Train AUROC']
        subset_err = ['Train ECE', 'Train MSE', 'Test ECE', 'Test MSE'] if test_df is not None else ['Train ECE', 'Train MSE']
        
        display(HTML("<h2>🌍 Overall Model Performance</h2>"))
        display(HTML(g_df.style.background_gradient(cmap='Greens', subset=subset_auroc)
                     .background_gradient(cmap='Reds', subset=[c for c in subset_err if c in g_df.columns])
                     .format("{:.4f}")
                     .to_html()))
    else:
        print("⚠️ No valid models to evaluate.")

    # --- 4. CLUSTER DEEP DIVE (Focus on Test Set if available) ---
    
    # Select Dataframe for Deep Dive (Prefer Test to show generalization)
    target_df = test_df if test_df is not None else train_df
    target_name = "TEST SET" if test_df is not None else "TRAIN SET"

    CSS = """
    <style>
        .cl-card { font-family:'Segoe UI', sans-serif; border:1px solid #ddd; border-radius:8px; margin-bottom:25px; box-shadow:0 3px 6px rgba(0,0,0,0.05); overflow:hidden; background:white; }
        .cl-head { padding:10px 15px; border-bottom:1px solid #eee; display:flex; justify-content:space-between; align-items:center; background:#f8f9fa; }
        .cl-grid { display:grid; grid-template-columns: 15% 20% 25% 20% 20%; }
        .cl-col { padding:12px; border-right:1px solid #eee; font-size:0.85em; }
        .cl-col:last-child { border-right:none; background:#fafafa; }
        .bar-row { display:flex; align-items:center; margin-bottom:5px; }
        .bar-label { width:80px; color:#555; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:0.9em; }
        .bar-track { flex:1; height:14px; background:#e0e0e0; border-radius:3px; overflow:hidden; margin-right:8px; }
        .bar-val { height:100%; display:flex; align-items:center; padding-left:4px; font-size:0.7em; color:white; font-weight:bold; }
        .sanity-tbl { width:100%; border-collapse:collapse; font-size:0.85em; }
        .sanity-tbl td { padding:2px 4px; border-bottom:1px solid #eee; color:#444; text-align:right; }
        .sanity-tbl td:first-child { text-align:left; font-weight:500; }
        .sanity-tbl th { text-align:right; padding:2px 4px; border-bottom:2px solid #ddd; color:#777; font-weight:600; font-size:0.8em; }
        .sanity-tbl th:first-child { text-align:left; }
    </style>
    """

    html_out = [CSS, f"<h2>🔍 Deep Dive: Per-Cluster Validation ({target_name})</h2>"]

    if 'Cluster_Label' in target_df.columns:
        # Use TRAIN to determine sort order (optional, but keeps consistency)
        unique_clusters = sorted(target_df['Cluster_Label'].unique())
        
        for label in unique_clusters:
            sub = target_df[target_df['Cluster_Label'] == label]
            n = len(sub)
            if n < 5: continue 
            
            # Confidence Interval
            acc = sub['Correct'].mean()
            ci = 1.96 * (sub['Correct'].std() / np.sqrt(n))
            acc_color = "#27ae60" if acc > 0.7 else ("#e67e22" if acc > 0.4 else "#c0392b")

            # --- COL 1: PROFILE (Shift from Global Mean) ---
            profile_html = "<div style='color:#777; font-weight:bold; margin-bottom:5px'>Cluster Profile</div>"
            for m in metrics_to_use:
                if m not in target_df.columns or target_df[m].std() == 0: continue
                # Calculate Z-score relative to GLOBAL mean (of the target set)
                z = (sub[m].mean() - target_df[m].mean()) / target_df[m].std()
                if abs(z) < 0.25: continue
                arrow = "⬆" if z > 0 else "⬇"
                color = "#c0392b" if z > 0 else "#2980b9"
                # Check flip for display intuition (optional, but helps readability)
                is_inv = metric_flip.get(m, False)
                
                profile_html += f"<div style='border-bottom:1px dashed #eee; padding:2px 0;'><b>{m}</b>: <span style='color:{color}'>{arrow} {abs(z):.1f}σ</span></div>"

            # --- COL 2: AVG CONFIDENCE ---
            preds_html = "<div style='color:#777; font-weight:bold; margin-bottom:5px'>Avg Confidence</div>"
            preds_html += f"<div class='bar-row'><div class='bar-label' style='font-weight:bold'>ACTUAL</div><div class='bar-track'><div class='bar-val' style='width:{acc*100}%; background:{acc_color}'>{acc:.1%}</div></div></div><hr style='margin:4px 0; border:0; border-top:1px solid #eee'>"
            for model in valid_models:
                conf = sub[model].mean()
                bg = "#2c3e50" if model == 'Logistic Fusion' else ("#7f8c8d" if model == 'Cluster Baseline' else "#95a5a6")
                # Highlight large discrepancies
                if abs(conf - acc) > 0.2 and model != 'Logistic Fusion': bg = "#e74c3c"
                preds_html += f"<div class='bar-row'><div class='bar-label'>{model}</div><div class='bar-track'><div class='bar-val' style='width:{conf*100}%; background:{bg}'>{conf:.0%}</div></div></div>"

            # --- COL 3: MSE SHOOTOUT ---
            shootout_html = "<div style='color:#777; font-weight:bold; margin-bottom:5px'>MSE (Lower is Better)</div>"
            cluster_errors = {m: brier_score_loss(sub['Correct'], sub[m]) for m in valid_models}
            if cluster_errors:
                min_err = min(cluster_errors.values())
                max_err = max(cluster_errors.values()) + 0.001
                for model in valid_models:
                    mse = cluster_errors[model]
                    width = (mse / max_err) * 100
                    is_winner = (mse == min_err)
                    star = "🏆 " if is_winner else ""
                    weight = "bold" if is_winner else "normal"
                    color = "#27ae60" if is_winner else "#95a5a6"
                    if model == 'Logistic Fusion': color = "#2c3e50"
                    shootout_html += f"<div style='display:flex; align-items:center; margin-bottom:4px; font-weight:{weight}; font-size:0.8em'><div style='width:80px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis'>{star}{model}</div><div style='flex:1; height:8px; background:#eee; margin:0 5px; border-radius:2px'><div style='width:{width}%; background:{color}; height:100%'></div></div><div style='width:35px; text-align:right; font-family:monospace'>{mse:.3f}</div></div>"

            # --- COL 4: AUROC ---
            auroc_html = "<div style='color:#777; font-weight:bold; margin-bottom:5px'>AUROC (Higher is Better)</div>"
            if len(sub['Correct'].unique()) > 1:
                cluster_aucs = {}
                for model in valid_models:
                    try: cluster_aucs[model] = roc_auc_score(sub['Correct'], sub[model])
                    except: cluster_aucs[model] = 0.5
                
                max_auc = max(cluster_aucs.values())
                for model in valid_models:
                    auc = cluster_aucs.get(model, 0.5)
                    width = max(0, (auc - 0.5) * 200) 
                    is_best = (auc == max_auc) and (auc > 0.5)
                    weight = "bold" if is_best else "normal"
                    color = "#8e44ad" if is_best else "#bdc3c7"
                    auroc_html += f"<div style='display:flex; align-items:center; margin-bottom:4px; font-weight:{weight}; font-size:0.8em'><div style='width:80px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis'>{model}</div><div style='flex:1; height:8px; background:#eee; margin:0 5px; border-radius:2px'><div style='width:{width}%; background:{color}; height:100%'></div></div><div style='width:35px; text-align:right; font-family:monospace'>{auc:.2f}</div></div>"
            else:
                auroc_html += "<div style='font-style:italic; color:#999; font-size:0.85em'>No variance in class labels.<br>AUROC undefined.</div>"

            # --- COL 5: SANITY ---
            sanity_html = "<div style='color:#777; font-weight:bold; margin-bottom:5px'>Stats</div>"
            sanity_html += "<table class='sanity-tbl'><thead><tr><th>Model</th><th>Mn</th><th>Rg</th></tr></thead><tbody>"
            for model in valid_models:
                vals = sub[model]
                mn, rng = vals.mean(), vals.max() - vals.min()
                sanity_html += f"<tr><td>{model}</td><td>{mn:.2f}</td><td>{rng:.2f}</td></tr>"
            sanity_html += "</tbody></table>"

            # --- ASSEMBLE ---
            card = f"""
            <div class="cl-card">
                <div class="cl-head" style="border-left:5px solid {acc_color}">
                    <div style="font-weight:bold; font-size:1.1em">{label.splitlines()[0]}</div>
                    <div>n={n} | Acc: {acc:.1%} ±{ci:.1%}</div>
                </div>
                <div class="cl-grid">
                    <div class="cl-col">{profile_html}</div>
                    <div class="cl-col">{preds_html}</div>
                    <div class="cl-col">{shootout_html}</div>
                    <div class="cl-col">{auroc_html}</div>
                    <div class="cl-col">{sanity_html}</div>
                </div>
            </div>
            """
            html_out.append(card)

    display(HTML("".join(html_out)))


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from scipy.stats import pointbiserialr

def plot_auarc_analysis(df, metrics_to_use):
    """
    Plots AUARC with empirical directionality correction.
    
    It calculates the correlation between the metric and correctness.
    - If Correlation > 0: High Metric = High Confidence (Keep as is)
    - If Correlation < 0: Low Metric = High Confidence (Flip sign)
    """
    
    # 1. Setup Predictors
    predictors = metrics_to_use.copy()
    # if 'Logistic Fusion' in df.columns: predictors.insert(0, 'Logistic Fusion')
    # if 'Cluster Baseline' in df.columns: predictors.insert(1, 'Cluster Baseline')
    
    plt.figure(figsize=(10, 6))
    
    # 2. Plot Baseline
    base_acc = df['Correct'].mean()
    plt.axhline(base_acc, color='black', linestyle='--', alpha=0.5, label=f'Baseline Acc ({base_acc:.1%})')
    
    for model in predictors:
        if model not in df.columns:
            print(f"Warning: {model} not found in DataFrame.")
            continue
        
        # Fill NaNs so correlation doesn't break
        raw_vals = df[model].fillna(df[model].mean())
        
        # 3. EMPIRICAL DIRECTIONALITY CHECK
        # Calculate correlation between metric and binary correctness
        # r > 0 implies higher metric values -> higher correctness
        # r < 0 implies higher metric values -> lower correctness (needs inversion)
        r, _ = pointbiserialr(raw_vals, df['Correct'])
        
        direction_label = ""
        if r < 0:
            # Negative correlation: Lower values are better (e.g., Entropy)
            confidence_scores = -1 * raw_vals
            direction_label = " (Inv)"
        else:
            # Positive correlation: Higher values are better (e.g., LogitGap)
            confidence_scores = raw_vals
            
        # 4. Sort Data by Confidence (High -> Low)
        sorted_df = pd.DataFrame({'y': df['Correct'], 'conf': confidence_scores})
        sorted_df = sorted_df.sort_values('conf', ascending=False)
        
        y_sorted = sorted_df['y'].values
        n = len(y_sorted)
        
        # 5. Calculate Curve
        rejection_rates = np.linspace(0, 0.95, 100) # Increased resolution
        accuracies = []
        
        for rr in rejection_rates:
            n_keep = int(n * (1 - rr))
            if n_keep < 10: break # Avoid instability with too few samples
            acc = np.mean(y_sorted[:n_keep])
            accuracies.append(acc)
            
        # 6. Plot
        valid_rr = rejection_rates[:len(accuracies)]
        score = auc(valid_rr, accuracies)
        
        # Formatting for special baselines
        is_baseline = 'Logistic' in model or 'Cluster' in model
        lw = 3 if is_baseline else 1.5
        alpha = 1.0 if is_baseline else 0.8
        
        plt.plot(valid_rr, accuracies, lw=lw, alpha=alpha, 
                 label=f'{model}{direction_label} (AUARC: {score:.3f})')

    plt.xlabel('Rejection Rate (Fraction Refused)')
    plt.ylabel('Accuracy of Remaining Answers')
    plt.title('Accuracy-Rejection Curve (Empirical Direction)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
# --- RUN IT ---
# plot_auarc_analysis(df, metrics_to_use)







def analyze_outlier_clusters(df):
    from IPython.display import display, HTML
    import pandas as pd
    
    # --- 0. PRE-FILTERING ---
    # Filter out generations longer than 500 tokens
    initial_len = len(df)
    df = df[df['Length'] <= 500].copy()
    filtered_len = len(df)
    
    if initial_len != filtered_len:
        print(f"ℹ️ Filtered {initial_len - filtered_len} rows with Length > 500 (New Size: {filtered_len})")

    # --- CONFIGURATION ---
    # 1. Low Entropy (High Confidence) + Low Mechanistic (High Confidence)
    mask_a = (df['Entropy_Z'] < -0.5) & (df['Mechanistic_Z'] < -0.5) 
    
    # 2. High Entropy (Low Confidence) + High Mechanistic (Low Confidence)
    mask_b = (df['Entropy_Z'] > 0.5) & (df['Mechanistic_Z'] > 0.5)
    
    # 3. Define metrics to display stats for
    metrics_to_show = ['LogitGap', 'Heuristic', 'Consistency', 'Length', 'Semantic']
    
    groups = [
        ("Group A: High Confidence (Low Ent + Low Mech)", mask_a, "#27ae60"), # Green
        ("Group B: High Uncertainty (High Ent + High Mech)", mask_b, "#e74c3c")  # Red
    ]

    # --- HTML STYLING ---
    css = """
    <style>
        .out-box { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 30px; overflow: hidden; font-family: 'Segoe UI', sans-serif; }
        .out-head { padding: 10px 15px; color: white; font-weight: bold; display: flex; justify-content: space-between; }
        .out-stats { padding: 15px; background: #f9f9f9; border-bottom: 1px solid #eee; display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; text-align: center; }
        .out-stat-item { background: white; padding: 8px; border-radius: 4px; border: 1px solid #eee; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
        .out-stat-val { font-weight: bold; font-size: 1.1em; color: #2c3e50; }
        .out-stat-lbl { font-size: 0.75em; color: #7f8c8d; text-transform: uppercase; letter-spacing: 0.5px; }
        .ex-row { padding: 10px 15px; border-bottom: 1px solid #eee; font-size: 0.9em; }
        .ex-meta { display: flex; gap: 10px; margin-bottom: 4px; font-size: 0.8em; color: #666; font-family: monospace; }
        .tag { padding: 2px 5px; border-radius: 3px; color: white; font-weight: bold; font-size: 0.8em;}
        .pass { background: #27ae60; } .fail { background: #c0392b; }
    </style>
    """
    display(HTML(css))

    for title, mask, color in groups:
        sub = df[mask].copy()
        n = len(sub)
        
        if n == 0:
            display(HTML(f"<div class='out-box'><div class='out-head' style='background:{color}'>{title} (N=0)</div><div style='padding:15px'>No samples found.</div></div>"))
            continue

        acc = sub['Correct'].mean()
        
        # --- 1. DYNAMIC STATISTICS ---
        stats_html = f"""
        <div class='out-stat-item'><div class='out-stat-val'>{n}</div><div class='out-stat-lbl'>Count</div></div>
        <div class='out-stat-item'><div class='out-stat-val'>{acc:.1%}</div><div class='out-stat-lbl'>Accuracy</div></div>
        """
        
        for m in metrics_to_show:
            if m in sub.columns:
                # Prioritize Z-score if available
                col = f"{m}_Z" if f"{m}_Z" in sub.columns else m
                val = sub[col].mean()
                disp_val = f"{val:.2f}"
                stats_html += f"<div class='out-stat-item'><div class='out-stat-val'>{disp_val}</div><div class='out-stat-lbl'>Avg {m}</div></div>"

        # --- 2. EXAMPLES (Top 10) ---
        examples_html = ""
        # Sort by Entropy to see extreme cases
        for i, row in sub.sort_values('Entropy_Z', ascending=False).head(10).iterrows():
            status = "<span class='tag pass'>PASS</span>" if row['Correct'] else "<span class='tag fail'>FAIL</span>"
            q_text = row.get('question_text', 'N/A')
            ans_text = str(row.get('full_trace_text', ''))[:]
            
            meta = f"Ent: {row.get('Entropy_Z', 0):.2f} | Mech: {row.get('Mechanistic_Z', 0):.2f}"
            
            examples_html += f"""
            <div class='ex-row'>
                <div class='ex-meta'>{status} {meta}</div>
                <div style='font-weight:bold; margin-bottom:2px; color:#2c3e50'>Q: {q_text}</div>
                <div style='color:#555; font-style:italic'>A: {ans_text}</div>
            </div>
            """

        # --- ASSEMBLE ---
        html = f"""
        <div class='out-box'>
            <div class='out-head' style='background:{color}'>
                <span>{title}</span>
                <span>Size: {n} ({n/len(df):.1%})</span>
            </div>
            <div class='out-stats'>{stats_html}</div>
            <div style='max-height:400px; overflow-y:auto'>{examples_html}</div>
        </div>
        """
        display(HTML(html))
