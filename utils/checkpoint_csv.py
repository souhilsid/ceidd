import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return value


def _flatten_prefixed(row: Dict[str, Any], prefix: str, value: Any) -> None:
    if not isinstance(value, dict):
        return
    for key, val in value.items():
        if key is None:
            continue
        row[f"{prefix}{str(key)}"] = _json_cell(val)


def _infer_batch_for_seq(seq: int, history: List[Dict[str, Any]]) -> Any:
    for item in history:
        try:
            cutoff = int(item.get("n_candidates_evaluated", 0))
        except Exception:
            cutoff = 0
        if cutoff > 0 and seq <= cutoff:
            return item.get("batch")
    return None


def checkpoint_payload_to_dataframes(payload: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    optimizer_state = payload.get("optimizer_state") if isinstance(payload, dict) else None
    if not isinstance(optimizer_state, dict):
        optimizer_state = {}

    history = optimizer_state.get("history")
    if not isinstance(history, list):
        history = []

    all_candidates = optimizer_state.get("all_candidates")
    if not isinstance(all_candidates, list):
        all_candidates = []

    rows: List[Dict[str, Any]] = []
    for idx, cand in enumerate(all_candidates, start=1):
        candidate = cand if isinstance(cand, dict) else {}
        row: Dict[str, Any] = {
            "seq": idx,
            "batch": _infer_batch_for_seq(idx, history),
            "trial_index": candidate.get("trial_index"),
            "source_trial_index": candidate.get("source_trial_index"),
            "distance": candidate.get("distance"),
            "is_experimental": bool(candidate.get("is_experimental", False)),
            "candidate_modified": bool(candidate.get("candidate_modified", False)),
            "proposed_trial_blocked": bool(candidate.get("proposed_trial_blocked", False)),
            "parameters_adjusted": bool(candidate.get("parameters_adjusted", False)),
            "constraints_adjusted": bool(candidate.get("constraints_adjusted", False)),
            "adaptive_adjusted": bool(candidate.get("adaptive_adjusted", False)),
        }

        predictions = candidate.get("predictions")
        if isinstance(predictions, list):
            for i, val in enumerate(predictions):
                row[f"pred__{i}"] = _json_cell(val)

        _flatten_prefixed(row, "param__", candidate.get("parameters"))
        _flatten_prefixed(row, "obj__", candidate.get("objective_values"))
        _flatten_prefixed(row, "measured__", candidate.get("measured_objectives"))
        _flatten_prefixed(row, "uncert__", candidate.get("uncertainties"))
        _flatten_prefixed(row, "meta__", candidate.get("measurement_metadata"))
        _flatten_prefixed(row, "model__", candidate.get("best_models_used"))
        rows.append(row)

    candidates_df = pd.DataFrame(rows)
    history_df = pd.DataFrame(history if isinstance(history, list) else [])

    progress = payload.get("progress")
    progress_rows: List[Dict[str, Any]] = []
    if isinstance(progress, dict):
        row: Dict[str, Any] = {}
        for key, val in progress.items():
            row[str(key)] = _json_cell(val)
        progress_rows.append(row)
    progress_df = pd.DataFrame(progress_rows)

    return {
        "candidates": candidates_df,
        "history": history_df,
        "progress": progress_df,
    }


def export_checkpoint_csvs(
    payload: Dict[str, Any],
    output_prefix: str,
) -> Dict[str, str]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    tables = checkpoint_payload_to_dataframes(payload)

    candidates_path = prefix.parent / f"{prefix.name}_candidates.csv"
    history_path = prefix.parent / f"{prefix.name}_history.csv"
    progress_path = prefix.parent / f"{prefix.name}_progress.csv"

    tables["candidates"].to_csv(candidates_path, index=False)
    tables["history"].to_csv(history_path, index=False)
    tables["progress"].to_csv(progress_path, index=False)

    return {
        "candidates_csv": str(candidates_path),
        "history_csv": str(history_path),
        "progress_csv": str(progress_path),
    }
