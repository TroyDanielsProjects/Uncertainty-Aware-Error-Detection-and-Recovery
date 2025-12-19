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

# HELPER FUNCTIONS-
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
    # Setup Data
    if train_df is None or len(train_df) < 5:
        print("Not enough data to run analysis.")
        return {}

    viz_df = test_df if test_df is not None else train_df
    data_label = "Test Set" if test_df is not None else "Train Set"
    
    sns.set_theme(style="white", context="paper")
    warnings.filterwarnings('ignore')

    predictors = {}
    
    # Filter Valid Metrics
    valid_metrics = [m for m in metrics if m in train_df.columns and f'{m}_Z' in train_df.columns]

    for m_exam in valid_metrics:

        try:
            # Fit single-variable LR on Train
            clf_single = LogisticRegression()
            clf_single.fit(train_df[[f'{m_exam}_Z']], train_df['Correct'])
            
            # Eval on Viz (Test)
            preds = clf_single.predict(viz_df[[f'{m_exam}_Z']])
            probs = clf_single.predict_proba(viz_df[[f'{m_exam}_Z']])[:, 1]
            
            acc_single = accuracy_score(viz_df['Correct'], preds)
            auroc_single = roc_auc_score(viz_df['Correct'], probs)

            # Store Predictor
            predictors[m_exam] = PredictorArtifact(
                model=clf_single, 
                scaler=None, 
                features=[f'{m_exam}_Z'], 
                mode='classifier'
            )
        except Exception as e:
            print(f"Stats failed for {m_exam}: {e}")
            auroc_single, acc_single = 0.5, 0.0

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
        
        # Pairwise Plots 
        other_metrics = [m for m in valid_metrics if m != m_exam]
        if other_metrics:
            fig, axes = plt.subplots(1, len(other_metrics), figsize=(4 * len(other_metrics), 3.5))
            if len(other_metrics) == 1: axes = [axes]
            
            for i, m_other in enumerate(other_metrics):
                # Train a 2D Classifier on TRAIN data
                cols_2d = [f'{m_exam}_Z', f'{m_other}_Z']
                clf_2d = LogisticRegression()
                clf_2d.fit(train_df[cols_2d], train_df['Correct'])
                
                # Plot on test, with train boudnary
                _plot_decision_boundary(
                    axes[i], 
                    viz_df, # test
                    cols_2d[0], 
                    cols_2d[1], 
                    'Correct',
                    model=clf_2d # train
                )
                
                axes[i].set_xlabel(f"{m_exam} (std)")
                axes[i].set_ylabel(f"{m_other} (std)")
            plt.tight_layout()
            plt.show()

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
    
    # 2. Collect Data
    query = f"""
    SELECT 
        r.result_id, r.question_id_external, r.question_text, r.gold_answer, 
        r.predicted_answer, r.full_trace_text, r.is_correct,
        r.uq_avg_entropy as 'Entropy', 
        r.uq_min_logit_gap as 'LogitGap',
        r.uq_heuristic_score as 'Heuristic', 
        r.uq_mech_score as 'Mechanistic',
        r.uq_semantic_entropy as 'Semantic',
        r.gpt_eval_reason as 'Reason',
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
    
    # Parse JSON Columns
    json_cols = ['uq_tokens', 'uq_mech_trace', 'uq_entropy_trace', 'uq_logit_gap_trace', 'similarity_vector']
    for col in json_cols:
        df[col] = df[col].apply(lambda x: json.loads(x) if x and isinstance(x, str) else [])

    df['Length'] = df['uq_tokens'].apply(len)
    df = df[df['Length'] <= 500]

    # Generate Z-Scores
    potential_metrics = ['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic', 'Consistency', 'Semantic', 'Length']
    valid_metrics = []
    
    for m in potential_metrics:
        if m in df.columns and df[m].var() > 0:
            valid_metrics.append(m)
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

# filter by observed difficulty
def get_difficulty_slice(df, k):
    """
    Returns a subset of df containing only questions that appear exactly 4 times
    and were answered correctly exactly 'k' times.
    """
    if 'question_id_external' not in df.columns:
        return pd.DataFrame()
    
    counts = df.groupby('question_id_external')['Correct'].transform('count')
    successes = df.groupby('question_id_external')['Correct'].transform('sum')
    
    return df[(counts == 4) & (successes == k)]

# tsne plotter
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

# pairwise cluster comparison
def get_pairwise_stats_raw(df, unique_clusters):
    """
    Calculates raw stats (N and Diff) for cluster pairs on SHARED questions.
    Returns a dataframe with columns ['c1', 'c2', 'N', 'Diff'].
    """
    if 'question_id_external' not in df.columns:
        return pd.DataFrame(columns=['c1', 'c2', 'N', 'Diff'])

    # Calculate average accuracy per question per cluster
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

    # cluster vs observed difficulty
    cross_tab = pd.crosstab(viz_df['Cluster'], viz_df['obs_k'])
    
    # Normalize row-wise to get percentages (so each bar is 100% height)
    cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
    
    # k=0 (Hard/Red) -> k=4 (Easy/Green)
    custom_colors = ['#c0392b', '#e67e22', '#f1c40f', '#3498db', '#2ecc71']
    cmap = ListedColormap(custom_colors)

    ax = cross_tab_norm.plot(
        kind='bar', 
        stacked=True, 
        figsize=(12, 6), 
        colormap=cmap,
        edgecolor='black',
        linewidth=0.5
    )
    
    plt.title("Cluster 'Difficulty' Discrimination", fontsize=14)
    plt.ylabel("Composition (%)", fontsize=12)
    plt.xlabel("Cluster ID", fontsize=12)
    plt.xticks(rotation=0)
    
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], [f"k={l} (Correct {l}/4)" for l in labels[::-1]], 
               title="Observed Difficulty", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', fontsize=9, weight='bold', padding=0)

    plt.tight_layout()
    plt.show()

# main clustering dashboard
def run_failure_modes_dashboard(train_df, exp_info, features=['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic'], test_df=None, min_clusters = 3, plot = True):
    """
    Generates HTML dashboard, t-SNE plot, and Pairwise Cluster Analysis.
    """
    if train_df is None: return
    exp_id, exp_name = exp_info
    
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

    # nice dashboard generation code
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
        sns.scatterplot(data=viz_df[viz_df['Correct']==1], x='x', y='y', hue='Cluster', palette=cluster_cmap, s=60, alpha=0.2, edgecolor='k', legend=False, linewidth=0, zorder=0)#color='#ecf0f1', s=60, alpha=0.3, linewidth=0, zorder=0)
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

        # difficulty stratification
        if test_df is not None and 'obs_k' in viz_df.columns:
            print("Generating Difficulty Stratification Plots.")
            plot_stratified_clusters(
                viz_df=viz_df, 
                tsne_proj=proj,
                unique_clusters=unique_clusters,
                k_levels=[0, 1, 2, 3, 4] 
            )

            print("Generating Discrimination Bar Chart.")
            plot_difficulty_discrimination(viz_df)

        # pairwise cluster analysis
        print("\n--- Pairwise Cluster Comparison (Same Question Overlap) ---")
    
        train_raw = get_pairwise_stats_raw(train_df, unique_clusters)
        test_raw = pd.DataFrame(columns=['c1', 'c2', 'N', 'Diff'])
        if test_df is not None:
            test_raw = get_pairwise_stats_raw(test_df, unique_clusters)

        # Merge ON cluster ids (c1, c2) to align rows correctly
        if not train_raw.empty or not test_raw.empty:
            merged = pd.merge(
                train_raw, 
                test_raw, 
                on=['c1', 'c2'], 
                how='outer', 
                suffixes=('_Train', '_Test')
            ).fillna(0)
            
            pairs = []
            for _, row in merged.iterrows():
                c1, c2 = int(row['c1']), int(row['c2'])
                acc1 = train_acc_map.get(c1, 0)
                acc2 = train_acc_map.get(c2, 0)
                
                pairs.append(f"C{c1} ({acc1:.1f}%) vs C{c2} ({acc2:.1f}%)")
            
            merged['Pair'] = pairs
            
            merged = merged.rename(columns={
                'N_Train': 'Train_N', 'Diff_Train': 'Train_Diff',
                'N_Test': 'Test_N', 'Diff_Test': 'Test_Diff'
            })
            
            final_df = merged[['Pair', 'Train_N', 'Train_Diff', 'Test_N', 'Test_Diff']]
            
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



def run_detailed_logistic_regression(train_df, test_df=None, features=['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic', 'Consistency', 'Semantic', 'Length'], plot = True):
    """
    Full Logistic Regression report with coefficients and visualizations.
    """
    if plot:
        display(Markdown("### Full Logistic Regression Analysis"))
    
    # Filter only features present in df
    valid_features = [f for f in features if f in train_df.columns]
    
    X_train_df = train_df.copy().fillna(0)
    X_test_df  = test_df.copy().fillna(0)

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
    
    predictors = metrics_to_use.copy()
    
    plt.figure(figsize=(10, 6))
    
    base_acc = df['Correct'].mean()
    plt.axhline(base_acc, color='black', linestyle='--', alpha=0.5, label=f'Baseline Acc ({base_acc:.1%})')
    
    for model in predictors:
        if model not in df.columns:
            print(f"Warning: {model} not found in DataFrame.")
            continue
        
        # Fill NaNs so correlation doesn't break
        raw_vals = df[model].fillna(df[model].mean())
        
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
        
        # Calculate Curve
        rejection_rates = np.linspace(0, 0.95, 100)
        accuracies = []
        
        for rr in rejection_rates:
            n_keep = int(n * (1 - rr))
            if n_keep < 10: break
            acc = np.mean(y_sorted[:n_keep])
            accuracies.append(acc)
            
        valid_rr = rejection_rates[:len(accuracies)]
        score = auc(valid_rr, accuracies)
        
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




def analyze_outlier_clusters(df):
    from IPython.display import display, HTML
    import pandas as pd
    

    filtered_len = len(df)

    # 1. Low Entropy (High Confidence) + Low Mechanistic (High Confidence)
    mask_a = (df['Entropy_Z'] < -0.5) & (df['Mechanistic_Z'] < -0.5) 
    
    # 2. High Entropy (Low Confidence) + High Mechanistic (Low Confidence)
    mask_b = (df['Entropy_Z'] > 0.5) & (df['Mechanistic_Z'] > 0.5)
    
    # 3. Define metrics to display stats for
    metrics_to_show = ["Entropy", "Mechanistic", 'LogitGap', 'Heuristic', 'Consistency', 'Length', 'Semantic']
    
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

        # examples
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





import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_detailed_consistency_pie(df, cluster_classifier=None):
    work_df = df.copy()
    if cluster_classifier:
        _, labels = cluster_classifier(work_df)
        work_df['Cluster'] = labels
    elif 'Cluster' not in work_df.columns:
        print("Error: DataFrame must have a 'Cluster' column.")
        return

    group_col = 'question_id_external' if 'question_id_external' in work_df.columns else 'question_text'
    sns.set_theme(style="white", context="paper")

    def classify(x):
        if len(x) < 2: return 'Single Sample'
        n_clusters = x['Cluster'].nunique()
        mean_acc = x['Correct'].mean()
        
        if mean_acc == 1.0:
            return 'Multi-Cluster (Always Correct)' if n_clusters > 1 else 'Single Cluster (Always Correct)'
        elif mean_acc == 0.0:
            return 'Multi-Cluster (Always Wrong)' if n_clusters > 1 else 'Single Cluster (Always Wrong)'
        else:
            return 'Multi-Cluster (Mixed Acc)' if n_clusters > 1 else 'Single Cluster (Mixed Acc)'

    group_class = work_df.groupby(group_col).apply(classify)
    counts = group_class.value_counts()
    total = len(group_class)
    
    color_map = {
        'Multi-Cluster (Always Correct)':  '#229954',
        'Single Cluster (Always Correct)': '#2ecc71',
        'Multi-Cluster (Mixed Acc)':       '#8e44ad',
        'Single Cluster (Mixed Acc)':      '#f39c12',
        'Multi-Cluster (Always Wrong)':    '#922b21',
        'Single Cluster (Always Wrong)':   '#e74c3c',
        'Single Sample':                   '#bdc3c7'
    }
    
    order = [
        'Multi-Cluster (Always Correct)', 'Single Cluster (Always Correct)',
        'Multi-Cluster (Mixed Acc)', 'Single Cluster (Mixed Acc)',
        'Multi-Cluster (Always Wrong)', 'Single Cluster (Always Wrong)',
        'Single Sample'
    ]
    
    labels = [l for l in order if l in counts.index]
    sizes = [counts[l] for l in labels]
    colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(11, 7))
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=colors, pctdistance=0.85,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 9}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
        
    for i, l in enumerate(labels):
        if l == 'Single Sample': autotexts[i].set_color('#555')

    fig.gca().add_artist(plt.Circle((0,0), 0.65, fc='white'))
    
    target_count = counts.get('Multi-Cluster (Mixed Acc)', 0)
    ax.text(0, 0, f"Target Set\n{(target_count/total)*100:.1f}%", 
            ha='center', va='center', fontsize=12, fontweight='bold', color='#8e44ad')

    ax.set_title(f"Detailed Cluster Consistency\n(N={total} Questions)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()









from IPython.display import HTML, display
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_intra_question_selection(test_df, cluster_predictor):
    probs, labels = cluster_predictor(test_df)
    
    analysis_df = test_df.copy()
    analysis_df['Safety_Score'] = probs
    
    group_col = 'question_id_external' if 'question_id_external' in analysis_df.columns else 'question_text'
    ans_col = 'predicted_answer' if 'predicted_answer' in analysis_df.columns else 'full_trace_text'

    grouped = analysis_df.groupby(group_col).filter(lambda x: len(x) > 1 and x['Safety_Score'].nunique() > 1)
    
    if grouped.empty:
        print("No valid multi-answer questions found for intra-question analysis.")
        return

    best_idx = grouped.groupby(group_col)['Safety_Score'].idxmax()
    worst_idx = grouped.groupby(group_col)['Safety_Score'].idxmin()
    
    acc_best = grouped.loc[best_idx, 'Correct'].mean()
    acc_worst = grouped.loc[worst_idx, 'Correct'].mean()
    acc_base = grouped['Correct'].mean()

    def get_voting_ev(sub):
        counts = sub[ans_col].value_counts()
        candidates = counts[counts == counts.max()].index
        return sub[sub[ans_col].isin(candidates)].groupby(ans_col)['Correct'].mean().mean()

    acc_vote = grouped.groupby(group_col).apply(get_voting_ev).mean()

    gain = acc_best - acc_base
    n_q = grouped[group_col].nunique()
    
    def color_score(val):
        if val > acc_base + 0.005: return "#27ae60"
        if val < acc_base - 0.005: return "#c0392b"
        return "#7f8c8d"
    
    display(HTML(f"""
    <div style='border:1px solid #ddd; padding:20px; border-radius:10px; background:white; font-family:sans-serif; max-width: 850px'>
        <div style='border-bottom:1px solid #eee; padding-bottom:10px; margin-bottom:15px'>
            <h3 style='margin:0; color:#2c3e50'>Selection Strategy Showdown</h3>
            <div style='color:#7f8c8d; font-size:0.9em; margin-top:5px'>
                Comparing Cluster Safety against Random and Voting baselines (N={n_q} Questions).
            </div>
        </div>
        <div style='display:grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap:10px; text-align:center; align-items:end'>
            <div style='opacity:0.7'>
                <div style='font-size:1.8em; color:#c0392b; font-weight:bold'>{acc_worst:.1%}</div>
                <div style='font-size:0.8em; font-weight:bold; color:#c0392b'>Riskiest</div>
            </div>
            <div>
                <div style='font-size:1.8em; color:#95a5a6; font-weight:bold'>{acc_base:.1%}</div>
                <div style='font-size:0.8em; font-weight:bold; color:#95a5a6'>Random</div>
            </div>
            <div>
                <div style='font-size:1.8em; color:{color_score(acc_vote)}; font-weight:bold'>{acc_vote:.1%}</div>
                <div style='font-size:0.8em; font-weight:bold; color:#2980b9'>Majority Vote</div>
            </div>
            <div>
                <div style='font-size:2.2em; color:{color_score(acc_best)}; font-weight:bold'>{acc_best:.1%}</div>
                <div style='font-size:0.8em; font-weight:bold; color:#8e44ad'>Safest Cluster</div>
            </div>
        </div>
        <div style='margin-top:20px; background:#f9f9f9; padding:10px; border-radius:6px; text-align:center'>
            <span style='color:#2c3e50; font-weight:600'>Cluster Gain vs Random:</span> 
            <span style='color:{color_score(acc_best)}; font-weight:bold; font-size:1.1em'> {gain:+.1%} </span>
            &nbsp;|&nbsp; 
            <span style='color:#2c3e50; font-weight:600'>vs Voting:</span>
            <span style='color:{color_score(acc_best - acc_vote)}; font-weight:bold; font-size:1.1em'> {acc_best - acc_vote:+.1%} </span>
        </div>
    </div>
    """))

    plot_data = pd.DataFrame({
        'Strategy': ['Riskiest', 'Random', 'Majority Vote', 'Safest'],
        'Accuracy': [acc_worst, acc_base, acc_vote, acc_best]
    })
    
    plt.figure(figsize=(9, 4))
    sns.set_theme(style="white", context="paper")
    ax = sns.barplot(data=plot_data, x='Strategy', y='Accuracy', palette=['#c0392b', '#95a5a6', '#5dade2', '#8e44ad'])
    plt.ylim(0, 1.1)
    plt.axhline(acc_base, color='k', linestyle='--', alpha=0.3, label='Baseline')
    
    for i, v in enumerate(plot_data['Accuracy']):
        weight = 'bold' if v == plot_data['Accuracy'].max() else 'normal'
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight=weight, color='#333')
        
    sns.despine()
    plt.title("Impact of Selection Strategy")
    plt.show()




from IPython.display import HTML, display
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_intra_question_selection_filtered(test_df, cluster_predictor):
    probs, labels = cluster_predictor(test_df)
    
    analysis_df = test_df.copy()
    analysis_df['Safety_Score'] = probs
    analysis_df['Cluster_Label'] = labels
    
    group_col = 'question_id_external' if 'question_id_external' in analysis_df.columns else 'question_text'
    ans_col = 'predicted_answer' if 'predicted_answer' in analysis_df.columns else 'full_trace_text'
    
    def classify_question_group(x):
        if len(x) < 2: return "Excluded: Single Sample"
        if x['Cluster_Label'].nunique() <= 1: return "Excluded: Single Cluster"
        if x['Correct'].nunique() <= 1:
            if x['Correct'].mean() == 1.0: return "Excluded: All Correct"
            if x['Correct'].mean() == 0.0: return "Excluded: All Wrong"
        return "Kept: Inconsistent Set"

    group_status = analysis_df.groupby(group_col).apply(classify_question_group)
    analysis_df = analysis_df.merge(group_status.rename('Status'), on=group_col)
    
    status_counts = group_status.value_counts().reset_index()
    status_counts.columns = ['Category', 'Count']
    status_counts['Percent'] = (status_counts['Count'] / status_counts['Count'].sum())
    
    log_html = f"<div style='margin-bottom:20px; font-family:sans-serif; border:1px solid #eee; border-radius:8px; overflow:hidden;'><div style='background:#f8f9fa; padding:10px 15px; border-bottom:1px solid #eee; font-weight:bold; color:#555;'>Data Funnel: Exclusion Log</div><table style='width:100%; border-collapse:collapse; font-size:0.9em;'><tr style='background:#fff; border-bottom:2px solid #f0f0f0;'><th style='text-align:left; padding:8px 15px; color:#999;'>Category</th><th style='text-align:right; padding:8px 15px; color:#999;'>Questions</th><th style='text-align:right; padding:8px 15px; color:#999;'>%</th></tr>"
    for _, row in status_counts.iterrows():
        cat = row['Category']
        bg = "#eafaf1" if "Kept" in cat else "#fff"
        weight = "bold" if "Kept" in cat else "normal"
        color = "#27ae60" if "Kept" in cat else "#7f8c8d"
        log_html += f"<tr style='background:{bg}; border-bottom:1px solid #f9f9f9;'><td style='padding:8px 15px; font-weight:{weight}; color:{color}'>{cat}</td><td style='text-align:right; padding:8px 15px; font-weight:{weight}; color:{color}'>{row['Count']}</td><td style='text-align:right; padding:8px 15px; color:#999'>{row['Percent']:.1%}</td></tr>"
    log_html += "</table></div>"
    display(HTML(log_html))

    grouped_df = analysis_df[analysis_df['Status'] == "Kept: Inconsistent Set"]
    if grouped_df.empty:
        print("No inconsistent questions found.")
        return

    best_idx = grouped_df.groupby(group_col)['Safety_Score'].idxmax()
    worst_idx = grouped_df.groupby(group_col)['Safety_Score'].idxmin()
    acc_best = grouped_df.loc[best_idx, 'Correct'].mean()
    acc_worst = grouped_df.loc[worst_idx, 'Correct'].mean()
    
    acc_base = grouped_df.groupby(group_col)['Correct'].mean().mean()

    def get_voting_ev(sub):
        counts = sub[ans_col].value_counts()
        max_votes = counts.max()
        candidates = counts[counts == max_votes].index
        cand_correctness = sub[sub[ans_col].isin(candidates)].groupby(ans_col)['Correct'].mean()
        return cand_correctness.mean()

    acc_vote = grouped_df.groupby(group_col).apply(get_voting_ev).mean()

    n_q = grouped_df[group_col].nunique()
    
    def color_score(val):
        if val > acc_base + 0.005: return "#27ae60"
        if val < acc_base - 0.005: return "#c0392b"
        return "#7f8c8d"

    display(HTML(f"""
    <div style='border:1px solid #ddd; padding:20px; border-radius:10px; background:white; font-family:sans-serif; max-width: 800px; box-shadow: 0 4px 6px rgba(0,0,0,0.05)'>
        <div style='border-bottom:1px solid #eee; padding-bottom:10px; margin-bottom:15px'>
            <h3 style='margin:0; color:#2c3e50'>Selection Strategy Showdown (4-Way)</h3>
            <div style='color:#7f8c8d; font-size:0.9em; margin-top:5px'>
                Comparing Cluster Safety against Random and Voting baselines (Inconsistent Set, N={n_q}).
            </div>
        </div>
        
        <div style='display:grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap:10px; text-align:center; align-items:end'>
            <div style='opacity:0.7'>
                <div style='font-size:2em; color:#c0392b; font-weight:bold'>{acc_worst:.1%}</div>
                <div style='font-size:0.8em; font-weight:bold; color:#c0392b; margin-top:5px'>Riskiest</div>
            </div>
            <div>
                <div style='font-size:2em; color:#95a5a6; font-weight:bold'>{acc_base:.1%}</div>
                <div style='font-size:0.8em; font-weight:bold; color:#95a5a6; margin-top:5px'>Random</div>
            </div>
            <div>
                <div style='font-size:2em; color:{color_score(acc_vote)}; font-weight:bold'>{acc_vote:.1%}</div>
                <div style='font-size:0.8em; font-weight:bold; color:#2980b9; margin-top:5px'>Majority Vote</div>
            </div>
            <div>
                <div style='font-size:2.4em; color:{color_score(acc_best)}; font-weight:bold'>{acc_best:.1%}</div>
                <div style='font-size:0.8em; font-weight:bold; color:#8e44ad; margin-top:5px'>Safest Cluster</div>
            </div>
        </div>
        
        <div style='margin-top:20px; background:#f9f9f9; padding:10px; border-radius:6px; text-align:center'>
            <span style='color:#2c3e50; font-weight:600'>Cluster Gain vs Random:</span> 
            <span style='color:{color_score(acc_best)}; font-weight:bold; font-size:1.1em'> {acc_best - acc_base:+.1%} </span>
            &nbsp;|&nbsp; 
            <span style='color:#2c3e50; font-weight:600'>vs Voting:</span>
            <span style='color:{color_score(acc_best - acc_vote)}; font-weight:bold; font-size:1.1em'> {acc_best - acc_vote:+.1%} </span>
        </div>
    </div>
    """))

    plot_data = pd.DataFrame({
        'Strategy': ['Riskiest', 'Random', 'Majority Vote', 'Safest'],
        'Accuracy': [acc_worst, acc_base, acc_vote, acc_best]
    })
    
    plt.figure(figsize=(9, 5))
    colors = ['#c0392b', '#bdc3c7', '#5dade2', '#8e44ad']
    sns.set_theme(style="white", context="paper")

    ax = sns.barplot(data=plot_data, x='Strategy', y='Accuracy', palette=colors)
    plt.ylim(0, 1.1)
    plt.axhline(acc_base, color='k', linestyle='--', alpha=0.4, label='Baseline')
    
    for i, v in enumerate(plot_data['Accuracy']):
        weight = 'bold' if v == plot_data['Accuracy'].max() else 'normal'
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight=weight, color='#333')
        
    sns.despine()
    plt.ylabel("Accuracy on Inconsistent Set")
    plt.title(f"Strategy Comparison on Inconsistent Questions (N={n_q})")
    plt.show()








import pandas as pd
import numpy as np
import re
from itertools import combinations
from IPython.display import display, HTML

def plot_pairwise_improvement_table(train_df, cluster_classifier, test_df=None):
    train_work = train_df.copy()
    
    _, train_labels = cluster_classifier(train_work)
    train_work['Cluster'] = train_labels

    if test_df is not None:
        test_work = test_df.copy()
        _, test_labels = cluster_classifier(test_work)
        test_work['Cluster'] = test_labels
    else:
        test_work = None

    css = """
    <style>
        .pw-container { font-family: 'Segoe UI', sans-serif; margin-top: 20px; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.03); max-width: 1000px; }
        .pw-header { padding: 20px; background: #fafafa; border-bottom: 1px solid #eee; }
        .pw-title { margin: 0; color: #2c3e50; font-size: 1.2em; font-weight: 600; }
        .pw-subtitle { font-size: 0.9em; color: #7f8c8d; margin-top: 5px; line-height: 1.4; }
        .pw-table { width: 100%; border-collapse: collapse; background: white; }
        .pw-table th { background: white; color: #95a5a6; font-weight: 600; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; padding: 12px 15px; text-align: center; border-bottom: 2px solid #f0f0f0; }
        .pw-table th:first-child { text-align: left; padding-left: 20px; }
        .pw-table td { padding: 10px 15px; border-bottom: 1px solid #f9f9f9; vertical-align: middle; font-size: 0.9em; color: #34495e; text-align: center; }
        .pw-table td:first-child { text-align: left; padding-left: 20px; }
        .pw-table tr:hover { background-color: #f8f9fa; transition: background 0.1s; }
        .strategy-flow { display: flex; align-items: center; gap: 8px; }
        .c-pill { padding: 4px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85em; color: white; min-width: 30px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
        .c-acc { color: #7f8c8d; font-size: 0.9em; }
        .arrow-icon { color: #bdc3c7; font-size: 1.1em; font-weight: bold; margin: 0 4px; }
        .exp-gain { color: black; font-weight: bold; font-size: 0.95em; }
        .delta-box { padding: 4px 8px; border-radius: 6px; font-weight: 700; font-size: 0.9em; display: inline-block; min-width: 65px; text-align: center; }
        .d-pos { background: #eafaf1; color: #27ae60; border: 1px solid #d5f5e3; }
        .d-neg { background: #fdedec; color: #c0392b; border: 1px solid #fadbd8; }
        .d-neu { background: #f4f6f6; color: #95a5a6; border: 1px solid #eaeded; font-weight: normal;}
        .n-tag { background: #fdfefe; border: 1px solid #e0e0e0; color: #7f8c8d; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-family: monospace; }
        .bg-0 { background-color: #3498db; }
        .bg-1 { background-color: #e74c3c; }
        .bg-2 { background-color: #f1c40f; }
        .bg-3 { background-color: #2ecc71; }
        .bg-4 { background-color: #9b59b6; }
        .bg-5 { background-color: #e67e22; }
        .bg-def { background-color: #95a5a6; }
    </style>
    """

    def get_cid(label):
        if isinstance(label, (int, float)): return int(label)
        match = re.search(r'\d+', str(label))
        return int(match.group()) if match else 0

    global_acc = train_work.groupby('Cluster')['Correct'].mean()
    unique_clusters = sorted(train_work['Cluster'].unique())

    def get_stats(df_in, c_low, c_high):
        if df_in is None: return 0, np.nan
        q_stats = df_in.groupby(['Cluster', 'question_id_external'])['Correct'].mean()
        if c_low not in q_stats or c_high not in q_stats: return 0, np.nan
        
        shared = q_stats.loc[c_low].index.intersection(q_stats.loc[c_high].index)
        n = len(shared)
        if n == 0: return 0, np.nan
        
        diff = (q_stats.loc[c_high].loc[shared] - q_stats.loc[c_low].loc[shared]).mean()
        return n, diff

    rows_html = ""
    valid_pairs = 0
    
    for c1, c2 in combinations(unique_clusters, 2):
        acc1, acc2 = global_acc.get(c1, 0), global_acc.get(c2, 0)
        
        if acc1 < acc2:
            left, right = c1, c2
            left_acc, right_acc = acc1, acc2
        else:
            left, right = c2, c1
            left_acc, right_acc = acc2, acc1
            
        n_train, diff_train = get_stats(train_work, left, right)
        if n_train == 0: continue 
        
        n_test, diff_test = get_stats(test_work, left, right)
        valid_pairs += 1
        
        exp_gain = right_acc - left_acc
        
        def fmt_d(val):
            if pd.isna(val): return '<span class="delta-box d-neu">-</span>'
            cls = "d-pos" if val > 0.005 else ("d-neg" if val < -0.005 else "d-neu")
            sign = "+" if val > 0 else ""
            return f'<span class="delta-box {cls}">{sign}{val:.1%}</span>'
        
        c_l_bg = f"bg-{get_cid(left) % 6}"
        c_r_bg = f"bg-{get_cid(right) % 6}"

        rows_html += f"""
        <tr>
            <td style="width: 30%">
                <div class="strategy-flow">
                    <div class="c-pill {c_l_bg}">{left}</div>
                    <div class="c-acc">{left_acc:.1%}</div>
                    <div class="arrow-icon">➜</div>
                    <div class="c-pill {c_r_bg}">{right}</div>
                    <div class="c-acc">{right_acc:.1%}</div>
                </div>
            </td>
            <td style="width: 15%;" class="exp-gain">+{exp_gain:.1%}</td>
            <td style="width: 13%"><span class="n-tag">N={n_train}</span></td>
            <td style="width: 13%">{fmt_d(diff_train)}</td>
            <td style="width: 13%"><span class="n-tag">N={n_test if test_work is not None else 0}</span></td>
            <td style="width: 13%">{fmt_d(diff_test)}</td>
        </tr>
        """

    if valid_pairs == 0:
        rows_html = '<tr><td colspan="6" style="text-align:center; padding:20px; color:#95a5a6">No shared questions found between clusters.</td></tr>'

    full_html = f"""
    {css}
    <div class="pw-container">
        <div class="pw-header">
            <h3 class="pw-title">Cluster Pairwise Comparison</h3>
            <div class="pw-subtitle">
                Isolating <b>Shared Questions</b> to measure the pure performance gain of switching clusters.<br>
                <span style="color:#27ae60; font-weight:bold">Positive (+%)</span> indicates the right-hand cluster is strictly better on identical inputs.
            </div>
        </div>
        <table class="pw-table">
            <thead>
                <tr>
                    <th>Strategy Path (Worse ➜ Better)</th>
                    <th>Theoretical 'Expected' Gain</th>
                    <th>Train Support</th>
                    <th>Train Gain</th>
                    <th>Test Support</th>
                    <th>Test Gain</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    
    display(HTML(full_html))






import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
from IPython.display import display, HTML

def compare_metric_models(train_df, test_df, model_feature_map, k_clusters=5):
    """
    Races arbitrary lists of features against each other.
    Returns a styled HTML leaderboard.
    """
    results = []
    
    for model_name, features in model_feature_map.items():
        valid_feats = [f for f in features if f in train_df.columns]
        if not valid_feats: continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[valid_feats].fillna(0))
        X_test = scaler.transform(test_df[valid_feats].fillna(0))
        y_train, y_test = train_df['Correct'], test_df['Correct']
        
        lr = LogisticRegression(class_weight='balanced', random_state=42)
        lr.fit(X_train, y_train)
        try: score_sup = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        except: score_sup = 0.5
            
        km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        train_labels = km.fit_predict(X_train)
        cluster_map = pd.DataFrame({'c': train_labels, 'y': y_train}).groupby('c')['y'].mean().to_dict()
        test_labels = km.predict(X_test)
        probs_unsup = np.array([cluster_map.get(l, 0.5) for l in test_labels])
        
        try: score_unsup = roc_auc_score(y_test, probs_unsup)
        except: score_unsup = 0.5

        results.append({
            'Model': model_name,
            'Features': ", ".join(valid_feats),
            'Sup': score_sup,
            'Unsup': score_unsup
        })

    results.sort(key=lambda x: x['Sup'], reverse=True)

    css = """
    <style>
        .lb-container { font-family: 'Segoe UI', sans-serif; margin-top: 20px; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.03); max-width: 850px; }
        .lb-header { padding: 15px 20px; background: #fafafa; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }
        .lb-title { margin: 0; color: #2c3e50; font-size: 1.2em; font-weight: 700; }
        .lb-table { width: 100%; border-collapse: collapse; background: white; }
        .lb-table th { background: white; color: #95a5a6; font-weight: 600; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; padding: 12px 20px; text-align: left; border-bottom: 2px solid #f0f0f0; }
        .lb-table td { padding: 12px 20px; border-bottom: 1px solid #f9f9f9; vertical-align: middle; font-size: 0.9em; color: #34495e; }
        .lb-table tr:hover { background-color: #f8f9fa; }
        
        .score-bar-bg { width: 100px; height: 6px; background: #eee; border-radius: 3px; display: inline-block; vertical-align: middle; margin-right: 8px; overflow: hidden; }
        .score-bar-fill { height: 100%; background: #27ae60; border-radius: 3px; }
        .feat-tag { font-size: 0.8em; color: #7f8c8d; font-family: monospace; background: #f4f6f6; padding: 2px 5px; border-radius: 3px; }
        .rank-badge { background: #2c3e50; color: white; width: 20px; height: 20px; text-align: center; line-height: 20px; border-radius: 50%; font-size: 0.75em; display: inline-block; margin-right: 8px; }
    </style>
    """

    rows_html = ""
    for i, r in enumerate(results):
        sup_pct = (r['Sup'] - 0.5) * 200 # Scale 0.5-1.0 to 0-100% for visual contrast
        unsup_pct = (r['Unsup'] - 0.5) * 200
        sup_width = max(0, min(100, sup_pct))
        unsup_width = max(0, min(100, unsup_pct))
        
        # Color coding for high performance
        sup_color = "#27ae60" if r['Sup'] > 0.7 else "#95a5a6"
        
        rows_html += f"""
        <tr>
            <td style="width: 35%">
                <div style="font-weight:600;"><span class="rank-badge">{i+1}</span>{r['Model']}</div>
                <div class="feat-tag" style="margin-top:4px">{r['Features']}</div>
            </td>
            <td style="width: 32%">
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{sup_width}%; background:{sup_color}"></div></div>
                <span style="font-weight:bold">{r['Sup']:.3f}</span>
            </td>
            <td style="width: 32%">
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{unsup_width}%; background:#9b59b6"></div></div>
                <span>{r['Unsup']:.3f}</span>
            </td>
        </tr>
        """

    full_html = f"""
    {css}
    <div class="lb-container">
        <div class="lb-header">
            <h3 class="lb-title">🏆 Feature Configuration Leaderboard</h3>
            <div style="font-size:0.85em; color:#7f8c8d">Metric: AUROC (Test Set)</div>
        </div>
        <table class="lb-table">
            <thead>
                <tr>
                    <th>Configuration</th>
                    <th>Supervised (LogReg)</th>
                    <th>Unsupervised (Cluster)</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    
    display(HTML(full_html))
    return pd.DataFrame(results)





import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_histograms(df, metrics):
    sns.set_theme(style="white", context="paper")
    
    valid_metrics = [m for m in metrics if m in df.columns]
    n = len(valid_metrics)
    
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1: axes = [axes]

    for ax, m in zip(axes, valid_metrics):
        sns.histplot(
            data=df, x=m, hue='Correct',
            palette={0: "#c0392b", 1: "#2ecc71"},
            kde=True,
            element="step",
            common_norm=False,
            alpha=0.2, linewidth=0,
            ax=ax
        )
        
        ax.set_title(m, fontweight='bold', color='#333')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_yaxis().set_ticks([])
        sns.despine(ax=ax, left=True)
        
        if ax.get_legend(): ax.get_legend().remove()

    fig.legend(
        handles=[
            plt.Rectangle((0,0),1,1, color="#2ecc71", alpha=0.5),
            plt.Rectangle((0,0),1,1, color="#c0392b", alpha=0.5)
        ], 
        labels=['Correct', 'Incorrect'], 
        loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.1), frameon=False
    )

    plt.tight_layout()
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_full_correlation_matrix(df, metrics=None):
    if metrics:
        work_df = df[metrics]
    else:
        work_df = df.select_dtypes(include=[np.number])
    
    corr = work_df.corr()
    
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white", context="paper")
    
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    
    sns.heatmap(
        corr, 
        cmap=cmap, 
        vmax=1.0, 
        vmin=-1.0, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .6},
        annot=True, 
        fmt=".2f",
        annot_kws={"size": 8}
    )
    
    plt.title("Correlation Matrix", fontsize=11, fontweight='bold', color='#2c3e50')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()
