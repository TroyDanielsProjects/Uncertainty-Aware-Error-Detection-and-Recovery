"""
SQLite database manager for storing experiment results.

This module wraps a small SQLite database used by the main pipeline to
record experiment configurations, model metadata, holistic uncertainty
vector components and chain‑of‑thought stage summaries.  It creates
the schema on initialisation using the bundled ``schema.sql`` file.
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
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.initialize_db()

    def get_connection(self) -> Optional[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            return None

    def initialize_db(self) -> None:
        # Locate the schema file relative to this script
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        if not os.path.exists(schema_path):
            print(f"Error: Schema file not found at {schema_path}.")
            return
        try:
            with open(schema_path, 'r') as f:
                schema_script = f.read()
            conn = self.get_connection()
            if conn:
                cursor = conn.cursor()
                cursor.executescript(schema_script)
                conn.commit()
                conn.close()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")

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
            cursor.execute(
                """SELECT model_id FROM Models WHERE 
                model_name = ? AND 
                (architecture IS ? OR architecture = ?) AND 
                (training_method IS ? OR training_method = ?) AND
                (mechanistic_config_path IS ? OR mechanistic_config_path = ?)""",
                (model_name, architecture, architecture, training_method, training_method, mech_config, mech_config)
            )
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
                uq_mech_score, uq_avg_entropy, uq_min_logit_gap, uq_saup_score, 
                uq_semantic_div, uq_semantic_entropy, uq_heuristic_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    # Map HUV components
                    uq_vector.mechanistic_score,
                    uq_vector.avg_entropy,
                    uq_vector.min_logit_gap,
                    uq_vector.saup_score,
                    uq_vector.semantic_divergence,
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
            rows = []
            for i, stage in enumerate(stage_data):
                rows.append(
                    (
                        result_id,
                        i,
                        stage.get('stage_name'),
                        stage.get('token_count'),
                        stage.get('avg_entropy'),
                        stage.get('min_logit_gap'),
                        stage.get('saup_score'),
                    )
                )
            cursor.executemany(
                """
                INSERT INTO Stage_Results (
                    result_id, stage_index, stage_name, token_count, uq_avg_entropy, uq_min_logit_gap, uq_saup_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error logging stage results: {e}")
        finally:
            conn.close()


if __name__ == '__main__':
    # Initialise DB if run directly (for testing)
    dbm = DBManager(db_path=os.path.join(os.path.dirname(__file__), 'results.sqlite'))
    print("Database initialised (if needed).")