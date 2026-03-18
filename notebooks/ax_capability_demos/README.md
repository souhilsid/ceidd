# Ax Capability Demos

These notebooks are standalone Ax demos. They do not modify the CEID app.

Recommended interpreter:
- `C:\Users\aisci\AppData\Local\Programs\Python\Python311\python.exe`

Notebook order:
1. `00_index.ipynb`
2. `01_outcome_constraints_status_quo.ipynb`
3. `02_tracking_metrics.ipynb`
4. `03_objective_thresholds_pareto.ipynb`
5. `04_early_stopping_progression.ipynb`
6. `05_external_generation_node_mock_llm.ipynb`
7. `06_scheduler_and_sql_storage.ipynb`

Compatibility notes:
- These notebooks target your actual runtime: `ax-platform 0.4.3`.
- In this version, `Scheduler` is the practical closed-loop orchestration tool.
- SQL storage is currently blocked in your Python 3.11 env because Ax 0.4.3 disables SQL integration when `SQLAlchemy >= 2`.
