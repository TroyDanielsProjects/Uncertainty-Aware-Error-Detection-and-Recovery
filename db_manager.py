"""
db_manager.py
SQLite interface. Stores results for Phase 1 (Gen) and Phase 2 (Analysis).
Optimized with persistent connection to prevent I/O thrashing.
"""
import sqlite3
import json
import os
from typing import Dict, Any

class DBManager:
    def __init__(self, db_path: str = 'db/results.sqlite'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Open ONCE and reuse. check_same_thread=False allows simple multithreading if needed.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def __del__(self):
        """Cleanup connection on exit."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def _init_db(self):
        schema = """
        CREATE TABLE IF NOT EXISTS Experiments (
            experiment_id INTEGER PRIMARY KEY, name TEXT, config TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS Models (
            model_id INTEGER PRIMARY KEY, name TEXT UNIQUE, mech_config TEXT
        );
        CREATE TABLE IF NOT EXISTS Results (
            result_id INTEGER PRIMARY KEY, 
            experiment_id INTEGER, model_id INTEGER, 
            question_id_external TEXT, question_text TEXT, 
            gold_answer TEXT, predicted_answer TEXT, full_trace_text TEXT, 
            is_correct BOOLEAN, eval_method TEXT, gpt_eval_reason TEXT,
            uq_avg_entropy REAL, uq_min_logit_gap REAL, 
            uq_semantic_entropy REAL, uq_heuristic_score REAL, uq_mech_score REAL
        );
        """
        try:
            with self.conn:
                self.conn.executescript(schema)
        except sqlite3.Error as e:
            print(f"DB Init Error: {e}")

    def new_experiment(self, name: str, config: Dict) -> int:
        with self.conn:
            cur = self.conn.execute("INSERT INTO Experiments (name, config) VALUES (?, ?)", (name, json.dumps(config)))
            return cur.lastrowid

    def register_model(self, name: str, mech_config: str = None) -> int:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO Models (name, mech_config) VALUES (?, ?)", (name, mech_config))
            # Separate select to ensure we get ID even if IGNORE triggered
            return self.conn.execute("SELECT model_id FROM Models WHERE name = ?", (name,)).fetchone()[0]

    def log_result(self, exp_id: int, mod_id: int, q_data: Dict, pred: str, trace_txt: str, is_corr: bool, uq: Any):
        with self.conn:
            self.conn.execute("""
                INSERT INTO Results (
                    experiment_id, model_id, question_id_external, question_text, gold_answer, 
                    predicted_answer, full_trace_text, is_correct, eval_method, 
                    uq_avg_entropy, uq_min_logit_gap, uq_semantic_entropy, uq_heuristic_score, uq_mech_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_id, mod_id, q_data['id'], q_data['q'], q_data['gold'], 
                pred, trace_txt, is_corr, 'Exact Match' if is_corr else 'Pending',
                uq.avg_entropy, uq.min_logit_gap, getattr(uq, 'semantic_entropy', 0.0), 
                uq.heuristic_score, uq.mechanistic_score
            ))