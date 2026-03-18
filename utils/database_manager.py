import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


class ExperimentDatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def initialize(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    strategy TEXT,
                    acquisition_function TEXT,
                    evaluator_type TEXT,
                    optimization_mode TEXT,
                    best_distance REAL,
                    best_parameters_json TEXT,
                    n_candidates INTEGER,
                    n_batches INTEGER,
                    export_dir TEXT,
                    report_html TEXT,
                    summary_json TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    seq INTEGER NOT NULL,
                    trial_index INTEGER,
                    source_trial_index INTEGER,
                    is_experimental INTEGER,
                    distance REAL,
                    parameters_json TEXT,
                    objective_values_json TEXT,
                    uncertainties_json TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL,
                    batch INTEGER,
                    batch_min_distance REAL,
                    batch_mean_distance REAL,
                    best_overall_distance REAL,
                    n_candidates_evaluated INTEGER,
                    generation_strategy TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
                """
            )
            conn.commit()

    def save_experiment_run(
        self,
        config: Any,
        result: Any,
        artifacts: Dict[str, Any],
    ) -> int:
        strategy = getattr(getattr(config, "generation_strategy", None), "value", str(getattr(config, "generation_strategy", "")))
        acquisition = getattr(getattr(config, "acquisition_function", None), "value", str(getattr(config, "acquisition_function", "")))
        evaluator_type = getattr(getattr(config, "evaluator_type", None), "value", str(getattr(config, "evaluator_type", "")))
        optimization_mode = getattr(getattr(config, "optimization_mode", None), "value", str(getattr(config, "optimization_mode", "")))

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO experiments (
                    experiment_name, created_at, strategy, acquisition_function,
                    evaluator_type, optimization_mode, best_distance,
                    best_parameters_json, n_candidates, n_batches,
                    export_dir, report_html, summary_json
                ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    getattr(config, "experiment_name", "Unknown"),
                    strategy,
                    acquisition,
                    evaluator_type,
                    optimization_mode,
                    float(getattr(result, "best_distance", float("nan"))),
                    json.dumps(getattr(result, "best_parameters", {}) or {}),
                    len(getattr(result, "all_candidates", []) or []),
                    len(getattr(result, "history", []) or []),
                    artifacts.get("export_dir"),
                    artifacts.get("report_html"),
                    artifacts.get("summary_json"),
                ),
            )
            experiment_id = int(cur.lastrowid)

            for idx, cand in enumerate(getattr(result, "all_candidates", []) or [], start=1):
                cur.execute(
                    """
                    INSERT INTO candidates (
                        experiment_id, seq, trial_index, source_trial_index, is_experimental,
                        distance, parameters_json, objective_values_json, uncertainties_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        idx,
                        cand.get("trial_index"),
                        cand.get("source_trial_index"),
                        1 if bool(cand.get("is_experimental", False)) else 0,
                        float(cand.get("distance")) if cand.get("distance") is not None else None,
                        json.dumps(cand.get("parameters") or {}),
                        json.dumps(cand.get("objective_values") or {}),
                        json.dumps(cand.get("uncertainties") or {}),
                    ),
                )

            for batch_item in getattr(result, "history", []) or []:
                cur.execute(
                    """
                    INSERT INTO batch_history (
                        experiment_id, batch, batch_min_distance, batch_mean_distance,
                        best_overall_distance, n_candidates_evaluated, generation_strategy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        batch_item.get("batch"),
                        batch_item.get("batch_min_distance"),
                        batch_item.get("batch_mean_distance"),
                        batch_item.get("best_overall_distance"),
                        batch_item.get("n_candidates_evaluated"),
                        batch_item.get("generation_strategy"),
                    ),
                )
            conn.commit()

        return experiment_id

    def list_experiments(self, limit: int = 100) -> pd.DataFrame:
        query = """
            SELECT id, experiment_name, created_at, strategy, acquisition_function,
                   evaluator_type, optimization_mode, best_distance,
                   n_candidates, n_batches, export_dir
            FROM experiments
            ORDER BY id DESC
            LIMIT ?
        """
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=(limit,))

    def get_candidates(self, experiment_id: int) -> pd.DataFrame:
        query = """
            SELECT seq, trial_index, source_trial_index, is_experimental, distance,
                   parameters_json, objective_values_json, uncertainties_json
            FROM candidates
            WHERE experiment_id = ?
            ORDER BY seq
        """
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=(experiment_id,))

    def get_batch_history(self, experiment_id: int) -> pd.DataFrame:
        query = """
            SELECT batch, batch_min_distance, batch_mean_distance,
                   best_overall_distance, n_candidates_evaluated, generation_strategy
            FROM batch_history
            WHERE experiment_id = ?
            ORDER BY batch
        """
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=(experiment_id,))

