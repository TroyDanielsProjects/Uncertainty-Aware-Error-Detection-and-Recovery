"""
SQLite database manager for storing experiment results.
"""

from __future__ import annotations

import sqlite3
import json
import os
from typing import Dict, Any, List, Optional


class DBManager:
    """
    Manages interactions with the SQLite database for storing HUV results.
    """

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
        if not conn:
            return
        
        # Defined inline to ensure correctness without external file dependence
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
            question_text TEXT,
            gold_answer TEXT,
            predicted_answer TEXT,
            is_correct BOOLEAN,
            full_trace_text TEXT,
            trace_file_path TEXT,
            
            -- METRICS (Duplicates removed)
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

        -- NEW: Granular Token Logging
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

    def register_model(self, model_name: str, architecture: Optional[str] = None, training_method: Optional[str] = None, mech_config: Optional[str] = None) -> Optional[int]:
        conn = self.get_connection()
        if not conn:
            return None
        cursor = conn.cursor()
        try:
            cursor.execute(
                """INSERT OR IGNORE INTO Models 
                (model_name, architecture, training_method, mechanistic_config_path) 
                VALUES (?, ?, ?, ?)""",
                (model_name, architecture, training_method, mech_config)
            )
            conn.commit()
            cursor.execute("SELECT model_id FROM Models WHERE model_name = ?", (model_name,))
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            print(f"Error registering model: {e}")
            return None
        finally:
            conn.close()

    def create_experiment(self, experiment_name: str, dataset_name: str, config: Dict[str, Any], repo_version: str = "1.0") -> Optional[int]:
        conn = self.get_connection()
        if not conn:
            return None
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO Experiments (experiment_name, dataset_name, configuration, repository_version) VALUES (?, ?, ?, ?)",
                (experiment_name, dataset_name, json.dumps(config), repo_version)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error creating experiment: {e}")
            return None
        finally:
            conn.close()

    def log_result(self, experiment_id: int, model_id: int, question_data: Dict[str, Any], prediction_data: Dict[str, Any], uq_vector: Any, trace_path: Optional[str] = None) -> Optional[int]:
        conn = self.get_connection()
        if not conn:
            return None
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
            INSERT INTO Results (
                experiment_id, model_id, question_id_external, question_text, gold_answer, 
                predicted_answer, is_correct, full_trace_text, trace_file_path,
                uq_mech_score, uq_avg_entropy, uq_min_logit_gap, 
                uq_semantic_entropy, uq_heuristic_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    model_id,
                    question_data.get('id_external'),
                    question_data.get('question'),
                    question_data.get('gold_answer'),
                    prediction_data.get('predicted_answer'),
                    prediction_data.get('is_correct'),
                    prediction_data.get('full_text'),
                    trace_path,
                    uq_vector.mechanistic_score,
                    uq_vector.avg_entropy,
                    uq_vector.min_logit_gap,
                    uq_vector.semantic_entropy,
                    uq_vector.heuristic_score,
                )
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Error logging result: {e}")
            return None
        finally:
            conn.close()

    def log_stage_results(self, result_id: int, stage_data: List[Dict[str, Any]]) -> None:
        conn = self.get_connection()
        if not conn:
            return
        cursor = conn.cursor()
        try:
            rows = [
                (
                    result_id,
                    i,
                    stage.get('stage_name'),
                    stage.get('token_count'),
                    stage.get('avg_entropy'),
                    stage.get('min_logit_gap'),
                )
                for i, stage in enumerate(stage_data)
            ]
            cursor.executemany(
                """
                INSERT INTO Stage_Results (
                    result_id, stage_index, stage_name, token_count, uq_avg_entropy, uq_min_logit_gap
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error logging stage results: {e}")
        finally:
            conn.close()

    def log_token_metrics(self, result_id: int, token_data: List[Dict[str, Any]]) -> None:
        """
        Bulk insert granular token metrics including mechanistic scores.
        """
        conn = self.get_connection()
        if not conn or not token_data:
            return
        cursor = conn.cursor()
        try:
            rows = [
                (
                    result_id,
                    t.get('position'),
                    t.get('token'),
                    t.get('entropy'),
                    t.get('top1_logprob'),
                    t.get('top2_logprob'),
                    t.get('logit_gap'),
                    t.get('mechanistic_score')
                )
                for t in token_data
            ]
            
            cursor.executemany(
                """
                INSERT INTO Token_Metrics (
                    result_id, position, token_text, entropy, top1_logprob, top2_logprob, logit_gap, mechanistic_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error logging token metrics: {e}")
        finally:
            conn.close()