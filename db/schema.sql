
-- Database Schema for Holistic Uncertainty Vector (HUV) Framework Results

PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- Table: Models
-- Tracks unique model configurations and their associated UQ calibrations.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,          -- e.g., "deepseek-math-7b-rl"
    architecture TEXT,                 -- e.g., "Llama"
    training_method TEXT,              -- e.g., "SFT", "RLHF"
    mechanistic_config_path TEXT,      -- Path to entropy neuron indices
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, architecture, training_method, mechanistic_config_path)
);

-- -----------------------------------------------------------------------------
-- Table: Experiments
-- Tracks experimental runs, configurations, dates, and code versions.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    repository_version TEXT,           -- Git Hash or version tag
    dataset_name TEXT NOT NULL,
    configuration TEXT                 -- JSON string of parameters
);

-- -----------------------------------------------------------------------------
-- Table: Results
-- The core table storing the HUV and outcomes.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    
    -- Question/Input Details
    question_id_external TEXT,
    question_text TEXT NOT NULL,
    gold_answer TEXT,
    
    -- Generation Outcome
    predicted_answer TEXT,
    is_correct BOOLEAN,
    full_trace_text TEXT,
    trace_file_path TEXT,              -- Path to raw trace file (JSON/NPY)
    
    -- Holistic Uncertainty Vector (HUV)
    uq_mech_score REAL,
    uq_avg_entropy REAL,
    uq_min_logit_gap REAL,
    uq_saup_score REAL,
    uq_semantic_div REAL,
    uq_semantic_entropy REAL,
    uq_heuristic_score REAL,
    
    FOREIGN KEY (experiment_id) REFERENCES Experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES Models(model_id)
);

-- -----------------------------------------------------------------------------
-- Table: Stage_Results
-- Dedicated table for Chain-of-Thought (CoT) stage-wise analysis.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS Stage_Results (
    stage_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    stage_index INTEGER NOT NULL,
    stage_name TEXT,
    token_count INTEGER,
    
    -- Stage-specific UQ metrics
    uq_avg_entropy REAL,
    uq_min_logit_gap REAL,
    uq_saup_score REAL,
    
    FOREIGN KEY (result_id) REFERENCES Results(result_id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_results_exp_model ON Results(experiment_id, model_id);
CREATE INDEX IF NOT EXISTS idx_stage_results_result ON Stage_Results(result_id);
