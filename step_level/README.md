# Step-Level Divergence Analysis (Exploratory Work)

This folder contains **exploratory/preliminary work** on step-level divergence analysis, extending semantic uncertainty quantification to localize *where* in multi-step reasoning uncertainty emerges.

> **Note**: This work is described in our report as a diagnostic framework for future intervention strategies. The step-level metrics are not included in the main fusion classifier but provide explainability for understanding uncertainty patterns. 

---

## Overview

While our main pipeline (in `ensemble-data-collection/`) measures *how much* uncertainty exists at the response level, this module addresses *where* in the reasoning process uncertainty emerges.

### Key Idea
For each problem, we generate K=5 reasoning trajectories, parse each into reasoning steps, and use **Dynamic Time Warping (DTW)** to align variable-length trajectories. We then compute divergence at each aligned position to create a "divergence profile."

---

## Files

| File | Description |
|------|-------------|
| `step_level_analysis.ipynb` | Main notebook implementing trajectory generation, DTW alignment, and divergence analysis |
| `results/` | Experimental results and summary statistics |

---

## Methodology

### 1. Trajectory Generation
- Generate K=5 independent reasoning trajectories per problem using GPT-4o-mini (temperature=0.7)
- Each trajectory is parsed into a sequence of reasoning steps

### 2. DTW Alignment
Since trajectories have variable lengths, we align them using Dynamic Time Warping:
```
T0: [A, B, C, D]      ->  Aligned to N positions
T1: [A, B, C, C, D]   ->  where each position
T2: [A, B, D]         ->  contains K steps
```

### 3. Divergence Metrics
At each aligned position, we compute:
- **Mean pairwise cosine distance** between step embeddings
- **Max step divergence**: Point of greatest disagreement
- **Divergence location**: Early/middle/late in reasoning

---

## Key Findings (Preliminary)

From analysis of 20 GSM8K problems:

| Pattern | Count | Interpretation |
|---------|-------|----------------|
| Final >> Trajectory | 15/20 | Similar reasoning paths, but final answers diverge |
| Trajectory >> Final | 1/20 | Different reasoning, same answer (robust) |
| Neither | 4/20 | Comparable uncertainty at both levels |

**Insight**: Uncertainty often concentrates at the final answer formulation step rather than being distributed throughout reasoning.

---

## Running the Analysis

### Requirements
```bash
pip install datasets openai sentence-transformers scikit-learn numpy pandas matplotlib seaborn plotly dtaidistance
```

### Environment
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Execution
```bash
jupyter notebook step_level_analysis.ipynb
```

---

## Relation to Main Pipeline

This module provides **complementary diagnostic information**:

| Main Pipeline (`ensemble-data-collection/`) | Step-Level Analysis |
|---------------------------------------------|---------------------|
| Response-level uncertainty metrics | Step-by-step divergence profiles |
| Predicts *if* model will fail | Shows *where* uncertainty emerges |
| Used in fusion classifier | Diagnostic/explainability tool |

---

## Future Work

As noted in the paper (Section VII-C), step-level divergence analysis opens avenues for targeted intervention:
1. Re-prompt the model at high-divergence steps
2. Route difficult sub-problems to more capable models
3. Request multiple samples and aggregate at critical steps

---

## Citation

See main project README for citation information.