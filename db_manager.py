"""
SQLite database manager. 
Updated schema to support local LLM grading.
"""

from __future__ import annotations

import sqlite3
import json
import os
from typing import Dict, Any, List, Optional

class DBManager:
    def __init__(self, db_path: str = 'db/results.sqlite'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.initialize_db()

    def get_connection(self) -> Optional[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;") 
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None

    def initialize_db(self) -> None:
        conn = self.get_connection()
        if not conn: return
        
        schema = """
        CREATE TABLE IF NOT EXISTS Experiments (
            experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            dataset_name TEXT,
            configuration TEXT,
            repository_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS Models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE NOT NULL,
            architecture TEXT,
            training_method TEXT,
            mechanistic_config_path TEXT
        );

        CREATE TABLE IF NOT EXISTS Results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            model_id INTEGER,
            question_id_external TEXT,
            sample_index INTEGER,
            question_text TEXT,
            gold_answer TEXT,
            predicted_answer TEXT,
            
            -- JUDGEMENT COLUMNS
            is_correct BOOLEAN,
            eval_reason TEXT,     -- Explanation from Llama/Local Grader
            eval_method TEXT,     -- 'Exact Match' or 'Model: Llama-3-8B'
            
            full_trace_text TEXT,
            trace_file_path TEXT,
            
            -- METRICS
            uq_mech_score REAL,
            uq_avg_entropy REAL,
            uq_min_logit_gap REAL,
            uq_semantic_entropy REAL,
            uq_heuristic_score REAL,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(experiment_id) REFERENCES Experiments(experiment_id),
            FOREIGN KEY(model_id) REFERENCES Models(model_id)
        );

        CREATE TABLE IF NOT EXISTS Stage_Results (
            stage_id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER,
            stage_index INTEGER,
            stage_name TEXT,
            token_count INTEGER,
            uq_avg_entropy REAL,
            uq_min_logit_gap REAL,
            FOREIGN KEY(result_id) REFERENCES Results(result_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS Token_Metrics (
            metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id INTEGER,
            position INTEGER,
            token_text TEXT,
            entropy REAL,
            top1_logprob REAL,
            top2_logprob REAL,
            logit_gap REAL,
            mechanistic_score REAL, 
            FOREIGN KEY(result_id) REFERENCES Results(result_id) ON DELETE CASCADE
        );
        """
        try:
            cursor = conn.cursor()
            cursor.executescript(schema)
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
        finally:
            conn.close()

    # --- REGISTRATION METHODS (Unchanged) ---
    def register_model(self, model_name: str, architecture: str = None, training_method: str = None, mech_config: str = None) -> Optional[int]:
        conn = self.get_connection()
        if not conn: return None
        cursor = conn.cursor()
        try:
            cursor.execute(
                """INSERT OR IGNORE INTO Models (model_name, architecture, training_method, mechanistic_config_path) VALUES (?, ?, ?, ?)""",
                (model_name, architecture, training_method, mech_config)
            )
            conn.commit()
            cursor.execute("SELECT model_id FROM Models WHERE model_name = ?", (model_name,))
            res = cursor.fetchone()
            return res[0] if res else None
        finally:
            conn.close()

    def create_experiment(self, experiment_name: str, dataset_name: str, config: Dict[str, Any], repo_version: str = "1.0") -> Optional[int]:
        conn = self.get_connection()
        if not conn: return None
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO Experiments (experiment_name, dataset_name, configuration, repository_version) VALUES (?, ?, ?, ?)",
                (experiment_name, dataset_name, json.dumps(config), repo_version)
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    # --- LOGGING METHODS ---
    def log_result(self, experiment_id: int, model_id: int, q_data: Dict[str, Any], p_data: Dict[str, Any], uq: Any, trace_path: str = None, sample_index: int = 0) -> Optional[int]:
        conn = self.get_connection()
        if not conn: return None
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
            INSERT INTO Results (
                experiment_id, model_id, question_id_external, sample_index, 
                question_text, gold_answer, predicted_answer, 
                is_correct, eval_reason, eval_method,
                full_trace_text, trace_file_path,
                uq_mech_score, uq_avg_entropy, uq_min_logit_gap, 
                uq_semantic_entropy, uq_heuristic_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id, model_id, q_data.get('id_external'), sample_index,
                    q_data.get('question'), q_data.get('gold_answer'),
                    p_data.get('predicted_answer'),
                    p_data.get('is_correct'), p_data.get('reason'), p_data.get('eval_method'),
                    p_data.get('full_text'), trace_path,
                    uq.mechanistic_score, uq.avg_entropy, uq.min_logit_gap,
                    uq.semantic_entropy, uq.heuristic_score,
                )
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error logging result: {e}")
            return None
        finally:
            conn.close()

    def update_grading(self, result_id: int, is_correct: bool, reason: str, method: str) -> None:
        """New method to update a row after the grader runs."""
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE Results SET is_correct = ?, eval_reason = ?, eval_method = ? WHERE result_id = ?",
                (is_correct, reason, method, result_id)
            )
            conn.commit()
        finally:
            conn.close()

    def get_ungraded_results(self, experiment_id: int) -> List[Any]:
        """Fetch results where is_correct is possibly ambiguous (or fetch all if needed)."""
        conn = self.get_connection()
        if not conn: return []
        # We fetch everything so we can decide in python which to skip (e.g. Exact Matches)
        df = conn.execute(f"SELECT result_id, question_text, gold_answer, predicted_answer, full_trace_text, eval_method FROM Results WHERE experiment_id = {experiment_id}").fetchall()
        conn.close()
        return df

    # (Keep log_stage_results and log_token_metrics unchanged)
    def log_stage_results(self, result_id: int, stage_data: List[Dict[str, Any]]) -> None:
        conn = self.get_connection(); 
        if not conn: return
        try:
            rows = [(result_id, i, s['stage_name'], s['token_count'], s['avg_entropy'], s['min_logit_gap']) for i, s in enumerate(stage_data)]
            conn.cursor().executemany("INSERT INTO Stage_Results (result_id, stage_index, stage_name, token_count, uq_avg_entropy, uq_min_logit_gap) VALUES (?, ?, ?, ?, ?, ?)", rows)
            conn.commit()
        finally: conn.close()

    def log_token_metrics(self, result_id: int, token_data: List[Dict[str, Any]]) -> None:
        conn = self.get_connection(); 
        if not conn: return
        try:
            rows = [(result_id, t['position'], t['token'], t['entropy'], t['top1_logprob'], t['top2_logprob'], t['logit_gap'], t['mechanistic_score']) for t in token_data]
            conn.cursor().executemany("INSERT INTO Token_Metrics (result_id, position, token_text, entropy, top1_logprob, top2_logprob, logit_gap, mechanistic_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            conn.commit()
        finally: conn.close()