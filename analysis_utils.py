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

# --- EXPORTED FUNCTIONS ---

def load_data(db_path):
    """Loads data, calculates Consistency, filters length, and standardizes."""
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return None, None

    conn = sqlite3.connect(db_path)
    try:
        exp_id = pd.read_sql("SELECT experiment_id FROM Experiments ORDER BY experiment_id DESC LIMIT 1", conn).iloc[0,0]
    except:
        print("❌ No experiments found.")
        return None, None

    print(f"Loading Experiment ID: {exp_id}...")
    
    # Fetch all relevant columns including similarity_vector
    query = f"""
    SELECT 
        r.result_id, r.is_correct, r.predicted_answer,
        r.uq_avg_entropy as 'Entropy',
        r.uq_min_logit_gap as 'LogitGap',
        r.uq_heuristic_score as 'Heuristic', 
        r.uq_mech_score as 'Mechanistic',
        r.similarity_vector, 
        r.uq_tokens, r.uq_mech_trace, r.uq_entropy_trace, r.uq_logit_gap_trace
    FROM Results r 
    WHERE r.experiment_id = {exp_id}
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Preprocess
    df['is_correct'] = df['is_correct'].astype(int)
    
    for col in ['uq_tokens', 'uq_mech_trace', 'uq_entropy_trace', 'uq_logit_gap_trace', 'similarity_vector']:
        df[col] = df[col].apply(lambda x: json.loads(x) if x and isinstance(x, str) else [])

    # Metric: Consistency
    df['Consistency'] = df['similarity_vector'].apply(lambda x: np.mean(x) if x else 0.0)

    # Metric: Length
    df['Length'] = df['uq_tokens'].apply(len)
    
    # Filter Cutoff
    df = df[df['Length'] <= 500]

    # Z-Score Standardization
    metrics = ['Entropy', 'LogitGap', 'Heuristic', 'Mechanistic', 'Length', 'Consistency']
    for m in metrics:
        if df[m].std() == 0: df[f'{m}_Z'] = 0
        else: df[f'{m}_Z'] = (df[m] - df[m].mean()) / df[m].std()
        
    return df, metrics

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