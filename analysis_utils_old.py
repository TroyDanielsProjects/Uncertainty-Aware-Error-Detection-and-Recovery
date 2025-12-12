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

def _plot_decision_boundary(ax, df, x_col, y_col, target):
    """Plots scatter and LR decision boundary."""
    plot_df = df[[x_col, y_col, target]].dropna()
    X = plot_df[[x_col, y_col]].values
    y = plot_df[target].values
    
    if len(np.unique(y)) < 2: return 
    
    # Scatter
    sns.scatterplot(
        data=plot_df, x=x_col, y=y_col, hue=target, 
        palette={0: '#e74c3c', 1: '#2ecc71'}, 
        alpha=0.6, s=30, ax=ax, legend=False
    )
    
    # Classifier
    clf = LogisticRegression()
    clf.fit(X, y)
    try: acc = accuracy_score(y, clf.predict(X))
    except: acc = 0.0
    try: auc = roc_auc_score(y, clf.predict_proba(X)[:,1])
    except: auc = 0.5
    
    # Contours
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linestyles='--', linewidths=1)
    ax.contourf(xx, yy, Z, levels=[0.0, 0.5, 1.0], colors=['#e74c3c', '#2ecc71'], alpha=0.1)
    
    clean_y_label = y_col.replace('_Z', '')
    ax.set_title(f"vs {clean_y_label}\nPair Acc: {acc:.1%} | Pair AUC: {auc:.2f}", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")


def run_full_analysis(df, metrics):
    """Iterates through metrics and generates report cards + plots."""
    if df is None or len(df) < 5:
        print("⚠️ Not enough data to run analysis.")
        return

    sns.set_theme(style="white", context="paper")
    warnings.filterwarnings('ignore')

    for m_exam in metrics:
        # 1. Stats
        try:
            clf_single = LogisticRegression()
            clf_single.fit(df[[f'{m_exam}_Z']], df['is_correct'])
            acc_single = accuracy_score(df['is_correct'], clf_single.predict(df[[f'{m_exam}_Z']]))
            probs = clf_single.predict_proba(df[[f'{m_exam}_Z']])[:, 1]
            auroc_single = roc_auc_score(df['is_correct'], probs)
        except:
            auroc_single, acc_single = 0.5, 0.0

        # 2. Header
        header = f"""
        <div style='border-bottom:3px solid #3498db; padding-top:20px; margin-bottom:15px'>
            <div style='display:flex; justify-content:space-between; align-items:center'>
                <h2 style='margin:0; color:#2c3e50'>{m_exam} Analysis</h2>
                <div style='font-family:sans-serif; font-size:0.95em;'>
                    <span style='color:#7f8c8d'>Discriminative Power:</span> 
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
        
        # 3. Plots
        other_metrics = [m for m in metrics if m != m_exam]
        fig, axes = plt.subplots(1, len(other_metrics), figsize=(4 * len(other_metrics), 3.5))
        for i, m_other in enumerate(other_metrics):
            _plot_decision_boundary(axes[i], df, f'{m_exam}_Z', f'{m_other}_Z', 'is_correct')
            axes[i].set_xlabel(f"{m_exam} (std)")
            axes[i].set_ylabel(f"{m_other} (std)")
        plt.tight_layout()
        plt.show()
        
        # 4. Examples
        if m_exam == 'Entropy': trace_col = 'uq_entropy_trace'
        elif m_exam == 'LogitGap': trace_col = 'uq_logit_gap_trace'
        else: trace_col = 'uq_mech_trace'
        
        sorted_df = df.sort_values(by=m_exam, ascending=False)
        
        html_ex = "<div style='display:flex; gap:20px; margin-top:10px'>"
        
        def make_card(row, title_prefix):
            tokens = row['uq_tokens']
            trace_vals = row.get(trace_col, [])
            colored_text = _generate_token_html(tokens, trace_vals, m_exam)
            status = "✅" if row['is_correct'] else "❌"
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

        html_ex += "<div style='flex:1'><h4 style='margin:0 0 5px 0'>Top 3 (Highest)</h4>"
        for _, row in sorted_df.head(3).iterrows(): html_ex += make_card(row, "High")
        html_ex += "</div>"
        
        html_ex += "<div style='flex:1'><h4 style='margin:0 0 5px 0'>Bottom 3 (Lowest)</h4>"
        for _, row in sorted_df.tail(3).iterrows(): html_ex += make_card(row, "Low")
        html_ex += "</div></div>"
        
        display(HTML(html_ex))




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
    acc = (sub_df['Correct'].sum() / len(sub_df)) * 100
    
    descriptions = []
    for f in features:
        if global_sds[f] == 0: continue
        z_val = (sub_df[f].mean() - global_means[f]) / global_sds[f]
        desc = get_magnitude_desc(z_val)
        if desc:
            descriptions.append((abs(z_val), f"{desc} {f}"))
            
    descriptions.sort(key=lambda x: x[0], reverse=True)
    top_descs = [d[1] for d in descriptions[:3]]
    
    chars = "\n".join(top_descs) if top_descs else "Average Stats"
    
    # 2. Return string WITH accuracy included
    return f"N={len(sub_df)} | Acc: {acc:.1f}%\n{chars}"

def run_failure_modes_dashboard(df, exp_info, features=['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic']):
    """Generates the HTML dashboard and t-SNE plot."""
    if df is None: return
    exp_id, exp_name = exp_info
    
    # 1. Clustering
    X = StandardScaler().fit_transform(df[features].fillna(0))
    # Adaptive cluster count
    n_clusters = min(12, len(df)//10) if len(df) > 50 else 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    df['Cluster'] = kmeans.fit_predict(X)

    # 2. HTML Generation
    global_medians = df[features].median()
    global_sds = df[features].std()
    
    html_out = [DASHBOARD_CSS, f'<div class="db-container">']
    html_out.append(f'<div class="db-header"><h2>Run: {exp_name} (ID: {exp_id})</h2><small>{len(df)} samples</small></div>')

    # Global Stats
    stat_cards = "".join([f'<div class="db-stat-card"><div class="db-stat-val">{global_medians[f]:.3f}</div><div class="db-stat-label">{f}</div></div>' for f in features])
    html_out.append(f'<div class="db-stat-grid">{stat_cards}</div>')

    # Per Cluster
    for c in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == c]
        n_total = len(sub)
        acc = (sub['Correct'].sum() / n_total) * 100
        
        metrics_rows = ""
        for f in features:
            mu, sd = sub[f].mean(), sub[f].std()
            shift_val = (mu - global_medians[f]) / global_sds[f] if global_sds[f] != 0 else 0
            
            bar_color = "#c0392b" if shift_val > 1.0 else ("#2980b9" if shift_val < -1.0 else "#95a5a6")
            vis_width = min(abs(shift_val) * 15, 50) 
            vis_left = "50%" if shift_val > 0 else f"{50 - vis_width}%"
            
            metrics_rows += f"""<tr><td><b>{f}</b></td><td style="color:{bar_color}">{shift_val:+.2f} SD</td><td>{mu:.3f}</td><td>{sd:.3f}</td>
                <td><div class="shift-bar-container"><div class="shift-bar" style="left:50%; width:1px; background:#ccc;"></div><div class="shift-bar" style="left:{vis_left}; width:{vis_width}%; background:{bar_color}; opacity:0.8;"></div></div></td></tr>"""
        
        ex_rows = ""
        for i, (_, row) in enumerate(sub.head(5).iterrows()):
            badge = '<span class="status-badge-pass">✅ Correct</span>' if row['Correct'] else '<span class="status-badge-fail">❌ Incorrect</span>'
            ex_rows += f'<div style="margin-bottom:10px;"><strong>[Ex {i+1}] {badge}</strong><br>Q: {row.get("question_text")}<br><div class="pred-box">Pred: {row.get("full_trace_text")}</div></div><hr>'
            
        html_out.append(generate_cluster_html(c, n_total, acc, metrics_rows, ex_rows))
        
    html_out.append("</div>")
    display(HTML("".join(html_out)))

    # 3. t-SNE Visualization
    print("Generating t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(df)-1), random_state=42, init='pca')
    proj = tsne.fit_transform(X)
    df['x'], df['y'] = proj[:, 0], proj[:, 1]
    
    unique_clusters = sorted(df['Cluster'].unique())
    palette_colors = sns.color_palette("turbo", n_colors=len(unique_clusters))
    cluster_cmap = dict(zip(unique_clusters, palette_colors))

    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=df[df['Correct']==1], x='x', y='y', color='#ecf0f1', s=60, alpha=0.3, linewidth=0, zorder=0)
    sns.scatterplot(data=df[df['Correct']==0], x='x', y='y', hue='Cluster', palette=cluster_cmap, s=120, alpha=0.9, edgecolor='k', legend=False, zorder=10)

    # Labels
    global_means = df[features].mean()
    for c in unique_clusters:
        sub = df[df['Cluster'] == c]
        failures = sub[sub['Correct'] == 0]
        target = failures if len(failures) > 3 else sub
        if len(target) == 0: continue
            
        cx, cy = target['x'].median(), target['y'].median()
        label_text = f"C{c}\n" + generate_label(sub, features, global_means, global_sds)
        
        plt.text(cx, cy, label_text, horizontalalignment='center', verticalalignment='center',
                 fontsize=9, fontweight='bold', color='black',
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=cluster_cmap[c], lw=3, alpha=0.9), zorder=20)

    plt.title(f"Failure Modes Analysis: {exp_name}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

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

def run_detailed_logistic_regression(df, features=['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic', 'Length']):
    """Full Logistic Regression report with coefficients and visualizations."""
    display(Markdown("### Full Logistic Regression Analysis"))
    
    train_q, test_q = train_test_split(df['question_text'].unique(), test_size=0.5, random_state=42)
    train_df = df[df['question_text'].isin(train_q)].copy().fillna(0)
    test_df  = df[df['question_text'].isin(test_q)].copy().fillna(0)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test  = scaler.transform(test_df[features])
    y_train, y_test = train_df['Correct'], test_df['Correct']
    
    clf = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    
    # 1. Feature Importance
    coef_df = pd.DataFrame({'Feature': features, 'Weight': clf.coef_[0], 'Odds Ratio': np.exp(clf.coef_[0])}).sort_values('Weight', ascending=False)
    plt.figure(figsize=(10, 4))
    sns.barplot(x='Weight', y='Feature', data=coef_df, palette=['forestgreen' if x>0 else 'crimson' for x in coef_df['Weight']])
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

def run_random_forest_diagnostics(df, features=['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic', 'Length']):
    """Random Forest analysis to catch non-linear patterns and hallucinations."""
    display(Markdown("### Random Forest Diagnostics"))
    
    train_q, test_q = train_test_split(df['question_text'].unique(), test_size=0.2, random_state=42)
    train_df, test_df = df[df['question_text'].isin(train_q)].fillna(0), df[df['question_text'].isin(test_q)].fillna(0)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    # Grid Search for robustness
    param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'class_weight': ['balanced']}
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc').fit(X_train, train_df['Correct'])
    best_clf = clf.best_estimator_
    
    probs = best_clf.predict_proba(X_test)[:, 1]
    test_df['Prob_Correct'] = probs
    
    # Plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.kdeplot(x=test_df[test_df['Correct']==1]['Prob_Correct'], fill=True, color='green', label='Actual Correct', alpha=0.3)
    sns.kdeplot(x=test_df[test_df['Correct']==0]['Prob_Correct'], fill=True, color='red', label='Actual Incorrect', alpha=0.3)
    plt.title('Prediction Confidence Density')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    pd.Series(best_clf.feature_importances_, index=features).sort_values().plot(kind='barh', color='#34495e')
    plt.title('Feature Importance (RF)')
    plt.tight_layout()
    plt.show()

# --- 5. TRACE SPIKE ANALYSIS ---

def analyze_trace_spikes(db_path, result_id=None, threshold=0.05):
    """Visualizes token-level spikes in mechanistic scores."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    query = "SELECT * FROM Results WHERE result_id = ?" if result_id else "SELECT * FROM Results ORDER BY result_id DESC LIMIT 1"
    params = (result_id,) if result_id else ()
    row = conn.execute(query, params).fetchone()
    conn.close()
    
    if not row: return print("No data found.")
    
    try:
        mech_trace = json.loads(row['uq_mech_trace'])
        tokens = json.loads(row['uq_tokens'])
    except: return print("Trace data missing.")

    deltas = [0] + [mech_trace[i] - mech_trace[i-1] for i in range(1, len(mech_trace))]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(mech_trace, label='Mechanistic Score', color='#2c3e50')
    plt.title(f"Trace Trajectory (ID: {row['result_id']})")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(deltas)), deltas, color=['red' if d > 0 else 'blue' for d in deltas])
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.axhline(y=-threshold, color='b', linestyle='--')
    plt.title("Token-wise Deltas")
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== MAJOR SPIKES (> {threshold}) ===")
    print(f"{'Pos':<5} | {'Delta':<8} | {'Score':<8} | {'Token'}")
    print("-" * 45)
    limit = min(len(tokens), len(deltas))
    for i in range(limit):
        if abs(deltas[i]) >= threshold:
            ind = "🔺" if deltas[i] > 0 else "🔻"
            tok = str(tokens[i]).replace('Ġ', ' ').replace('Ċ', '\\n')
            print(f"{i:<5} | {deltas[i]:+.4f} {ind} | {mech_trace[i]:.4f}   | \"{tok}\"")