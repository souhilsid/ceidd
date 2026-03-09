import base64
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return cleaned[:80] or "experiment"


def _candidate_objective_value(candidate: Dict[str, Any], objective_name: str, objective_idx: int) -> Any:
    objective_values = candidate.get("objective_values") or {}
    if objective_name in objective_values:
        return objective_values.get(objective_name)

    predictions = candidate.get("predictions")
    if isinstance(predictions, np.ndarray):
        predictions = predictions.flatten().tolist()
    if isinstance(predictions, list) and objective_idx < len(predictions):
        return predictions[objective_idx]
    return np.nan


def build_result_tables(result: Any, config: Any) -> Dict[str, pd.DataFrame]:
    parameter_names = [p.name for p in getattr(config, "parameters", [])]
    objective_names = [o.name for o in getattr(config, "objectives", [])]

    candidate_rows: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(getattr(result, "all_candidates", []) or [], start=1):
        row = {
            "seq": idx,
            "trial_index": candidate.get("trial_index"),
            "source_trial_index": candidate.get("source_trial_index"),
            "distance": candidate.get("distance"),
            "is_experimental": bool(candidate.get("is_experimental", False)),
            "candidate_modified": bool(candidate.get("candidate_modified", False)),
            "proposed_trial_blocked": bool(candidate.get("proposed_trial_blocked", False)),
        }
        for name in parameter_names:
            row[f"param__{name}"] = (candidate.get("parameters") or {}).get(name)

        uncertainties = candidate.get("uncertainties") or {}
        for obj_idx, obj_name in enumerate(objective_names):
            row[f"obj__{obj_name}"] = _candidate_objective_value(candidate, obj_name, obj_idx)
            row[f"uncert__{obj_name}"] = uncertainties.get(obj_name, np.nan)
        candidate_rows.append(row)

    candidates_df = pd.DataFrame(candidate_rows)
    history_df = pd.DataFrame(getattr(result, "history", []) or [])

    pareto_rows: List[Dict[str, Any]] = []
    for idx, point in enumerate(getattr(result, "pareto_front", []) or [], start=1):
        row = {"rank": idx}
        for pname, pval in (point.get("parameters") or {}).items():
            row[f"param__{pname}"] = pval
        for oname, oval in (point.get("outcomes") or {}).items():
            row[f"obj__{oname}"] = oval
        pareto_rows.append(row)
    pareto_df = pd.DataFrame(pareto_rows)

    return {
        "candidates": candidates_df,
        "history": history_df,
        "pareto": pareto_df,
    }


def _save_tables(tables: Dict[str, pd.DataFrame], tables_dir: Path) -> Dict[str, str]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    for name, df in tables.items():
        csv_path = tables_dir / f"{name}.csv"
        json_path = tables_dir / f"{name}.json"
        if df.empty:
            df = pd.DataFrame()
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        saved[f"{name}_csv"] = str(csv_path)
        saved[f"{name}_json"] = str(json_path)
    return saved


def _save_distance_progress_chart(history_df: pd.DataFrame, out_path: Path) -> bool:
    if history_df.empty:
        return False
    required = {"batch", "best_overall_distance"}
    if not required.issubset(set(history_df.columns)):
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_df["batch"], history_df["best_overall_distance"], marker="o", label="Best overall")
    if "batch_min_distance" in history_df.columns:
        ax.plot(history_df["batch"], history_df["batch_min_distance"], marker="o", label="Batch min")
    if "batch_mean_distance" in history_df.columns:
        ax.plot(history_df["batch"], history_df["batch_mean_distance"], marker="o", label="Batch mean")
    ax.set_title("Optimization Distance Progress")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _save_candidate_distance_chart(candidates_df: pd.DataFrame, out_path: Path) -> bool:
    if candidates_df.empty or "distance" not in candidates_df.columns:
        return False
    distances = pd.to_numeric(candidates_df["distance"], errors="coerce")
    if distances.dropna().empty:
        return False

    best_so_far = distances.copy()
    best_so_far = best_so_far.where(~best_so_far.isna(), np.inf)
    best_so_far = best_so_far.cummin().replace(np.inf, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(candidates_df["seq"], distances, alpha=0.6, marker=".", label="Candidate distance")
    ax.plot(candidates_df["seq"], best_so_far, linewidth=2.5, label="Best so far")
    ax.set_title("Candidate Distances")
    ax.set_xlabel("Candidate #")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _save_multi_trace_chart(df: pd.DataFrame, prefix: str, title: str, ylabel: str, out_path: Path, max_cols: int = 6) -> bool:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return False
    cols = cols[:max_cols]

    fig, ax = plt.subplots(figsize=(11, 6))
    for col in cols:
        ax.plot(df["seq"], pd.to_numeric(df[col], errors="coerce"), marker=".", label=col.replace(prefix, ""))
    ax.set_title(title)
    ax.set_xlabel("Candidate #")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def _save_pareto_chart(candidates_df: pd.DataFrame, out_path: Path) -> bool:
    obj_cols = [c for c in candidates_df.columns if c.startswith("obj__")]
    if len(obj_cols) < 2:
        return False

    x = pd.to_numeric(candidates_df[obj_cols[0]], errors="coerce")
    y = pd.to_numeric(candidates_df[obj_cols[1]], errors="coerce")
    mask = ~(x.isna() | y.isna())
    if not mask.any():
        return False

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x[mask], y[mask], alpha=0.75, s=35)
    ax.set_title("Objective Space Scatter")
    ax.set_xlabel(obj_cols[0].replace("obj__", ""))
    ax.set_ylabel(obj_cols[1].replace("obj__", ""))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def save_charts_png(tables: Dict[str, pd.DataFrame], charts_dir: Path) -> Dict[str, str]:
    charts_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    candidates_df = tables["candidates"]
    history_df = tables["history"]

    chart_specs: List[Tuple[str, bool]] = [
        ("distance_progress", _save_distance_progress_chart(history_df, charts_dir / "distance_progress.png")),
        ("candidate_distances", _save_candidate_distance_chart(candidates_df, charts_dir / "candidate_distances.png")),
        (
            "objective_traces",
            _save_multi_trace_chart(
                candidates_df,
                prefix="obj__",
                title="Objective Traces",
                ylabel="Objective value",
                out_path=charts_dir / "objective_traces.png",
            ),
        ),
        (
            "parameter_traces",
            _save_multi_trace_chart(
                candidates_df,
                prefix="param__",
                title="Parameter Traces",
                ylabel="Parameter value",
                out_path=charts_dir / "parameter_traces.png",
            ),
        ),
        (
            "uncertainty_traces",
            _save_multi_trace_chart(
                candidates_df,
                prefix="uncert__",
                title="Uncertainty Traces",
                ylabel="Uncertainty",
                out_path=charts_dir / "uncertainty_traces.png",
            ),
        ),
        ("objective_scatter", _save_pareto_chart(candidates_df, charts_dir / "objective_scatter.png")),
    ]

    for name, ok in chart_specs:
        if ok:
            saved[name] = str(charts_dir / f"{name}.png")
    return saved


def _img_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _render_table_html(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df.empty:
        return "<p><em>No data available.</em></p>"
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0)


def generate_html_report(
    config: Any,
    result: Any,
    tables: Dict[str, pd.DataFrame],
    chart_paths: Dict[str, str],
    report_path: Path,
) -> str:
    summary = {
        "Experiment": getattr(config, "experiment_name", "Unknown"),
        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Strategy": getattr(getattr(config, "generation_strategy", None), "value", str(getattr(config, "generation_strategy", ""))),
        "Acquisition": getattr(getattr(config, "acquisition_function", None), "value", str(getattr(config, "acquisition_function", ""))),
        "Best Distance": getattr(result, "best_distance", np.nan),
        "Candidates": len(getattr(result, "all_candidates", []) or []),
        "Batches": len(getattr(result, "history", []) or []),
    }

    summary_rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary.items()
    )

    charts_html = ""
    for name, path in chart_paths.items():
        img = _img_base64(path)
        charts_html += (
            f"<h3>{name.replace('_', ' ').title()}</h3>"
            f"<img src='data:image/png;base64,{img}' style='max-width:100%; border:1px solid #ddd;'/>"
        )

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Optimization Report - {summary['Experiment']}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2, h3 {{ color: #0f172a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0 24px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; }}
    .data-table tr:nth-child(even) {{ background: #f8fafc; }}
    .section {{ margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Bayesian Optimization Report</h1>
  <div class="section">
    <h2>Run Summary</h2>
    <table>{summary_rows}</table>
  </div>
  <div class="section">
    <h2>Charts</h2>
    {charts_html or "<p><em>No charts were generated.</em></p>"}
  </div>
  <div class="section">
    <h2>Candidates (preview)</h2>
    {_render_table_html(tables["candidates"])}
  </div>
  <div class="section">
    <h2>Batch History (preview)</h2>
    {_render_table_html(tables["history"])}
  </div>
  <div class="section">
    <h2>Pareto Front (preview)</h2>
    {_render_table_html(tables["pareto"])}
  </div>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)


def _zip_folder(source_dir: Path, zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                abs_path = Path(root) / file_name
                rel_path = abs_path.relative_to(source_dir)
                zf.write(abs_path, arcname=str(rel_path))
    return str(zip_path)


def export_run_artifacts(
    config: Any,
    result: Any,
    export_root: str,
    config_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    root = Path(export_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = _safe_name(getattr(config, "experiment_name", "experiment"))
    export_dir = root / f"{exp_name}_{stamp}"
    tables_dir = export_dir / "tables"
    charts_dir = export_dir / "charts"

    export_dir.mkdir(parents=True, exist_ok=True)
    tables = build_result_tables(result=result, config=config)
    table_paths = _save_tables(tables=tables, tables_dir=tables_dir)
    chart_paths = save_charts_png(tables=tables, charts_dir=charts_dir)

    report_path = generate_html_report(
        config=config,
        result=result,
        tables=tables,
        chart_paths=chart_paths,
        report_path=export_dir / "report.html",
    )

    summary_payload = {
        "experiment_name": getattr(config, "experiment_name", "Unknown"),
        "created_at": datetime.now().isoformat(),
        "best_distance": getattr(result, "best_distance", None),
        "n_candidates": len(getattr(result, "all_candidates", []) or []),
        "n_batches": len(getattr(result, "history", []) or []),
        "tables": table_paths,
        "charts": chart_paths,
        "report_html": report_path,
    }
    if config_payload is not None:
        summary_payload["config"] = config_payload

    summary_path = export_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    zip_path = _zip_folder(export_dir, export_dir / "bundle.zip")

    return {
        "export_dir": str(export_dir),
        "tables": table_paths,
        "charts": chart_paths,
        "report_html": report_path,
        "summary_json": str(summary_path),
        "bundle_zip": zip_path,
    }

