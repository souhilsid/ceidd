# CEID
Cooperative Explorer for Inverse Design platform with Bayesian Optimization and Self-Driving Lab (SDL) integration.

CEID is a Streamlit platform for Bayesian Optimization (BO) workflows across:
- Virtual evaluators (ML surrogates)
- Self-Driving Lab (SDL) evaluators (hardware/remote loop)
- Embedded SDL execution (run the RYB hardware agent in-process)

It supports full experiment setup, optimization execution, checkpoint/resume, report export, and SQLite run archival.

## 1. What Is In This Repository

```text
CEID/
  app.py                      # Main Streamlit application
  RYB_SDL.py                  # Headless RYB SDL agent (printer + relay + Arduino + Unity)
  core/
    config.py                 # Config schema, enums, validation
    optimization.py           # BO engine (Ax/BoTorch), trial loop, checkpoint state
    evaluators.py             # Virtual evaluator training/inference/uncertainty
    models.py                 # Model registry + custom model loaders
    sdl.py                    # SDL connector (mqtt/http/tcp/serial/embedded)
    visualization.py          # Plotting/dashboard helpers
  utils/
    data_loader.py            # Data loading and validation helpers
    state_manager.py          # JSON + Ax state save/load
    reporting.py              # Report/tables/charts export bundle
    database_manager.py       # SQLite run persistence
  checkpoints/               # Saved optimization checkpoints
  exports/                   # Generated reports/bundles/database
```

## 2. CEID Workflow (UI)

The app has 4 pages:
1. **Overview**
2. **Setup**
3. **Optimize**
4. **Results & Export**

Typical flow:
1. Upload data (optional in SDL and some virtual scenarios).
2. Configure parameters/objectives/models/constraints/evaluator.
3. Run optimization in batch or sequential mode.
4. Save checkpoints during run.
5. Export report/tables/charts and optionally save run to SQLite.

## 3. Modes and Scenarios

### 3.1 Evaluator Modes

CEID supports these evaluator types:
- `virtual`
- `self_driving_lab`
- `third_party_simulator` (UI placeholder; integration is intentionally not complete)

### 3.2 Virtual Mode Scenarios

1. **Data-driven virtual BO**
- Upload dataset with parameter and objective columns.
- CEID trains enabled models and optimizes from historical data.

2. **Custom-model virtual BO with no dataset**
- Upload pre-trained custom model(s).
- CEID can optimize using those model predictions without uploaded historical data.

3. **Cold-start virtual fallback**
- If no dataset and no custom model exist, CEID can still run with a synthetic deterministic objective signal.
- This is useful for smoke-testing the optimization pipeline only.

### 3.3 SDL Mode Scenarios

1. **External SDL service** via protocol:
- `mqtt`
- `http`
- `tcp`
- `serial`

2. **Embedded SDL**
- CEID starts an embedded RYB SDL runner from `core/sdl.py` (`protocol = embedded`).
- Supports Unity transport controls and digital twin gating.

3. **Automatic SDL control with Unity gating**
- `digital_twin_control = true`: trial execution waits for Start/Continue commands.
- `require_continue_each_trial = true`: pause after each automatic trial.

### 3.4 Optimization Execution Modes

- `batch`: evaluate multiple candidates per batch.
- `sequential`: 1 candidate per step.

### 3.5 Objective Handling Scenarios

1. **Direct objective optimization**
- If at least one objective is `minimize` or `maximize`, CEID optimizes direct outputs (Pareto-capable for multi-objective).

2. **Target-only objective optimization**
- If objectives are only `target_range`/`target_value`, CEID optimizes a composite distance objective.

## 4. Installation

## 4.1 Python
- Recommended: **Python 3.10+**

## 4.2 Install

```powershell
cd CEID
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4.3 Run

```powershell
streamlit run app.py --server.port 8501
```

Open:
- `http://localhost:8501`

If port is busy, use:
```powershell
python port_checker.py
```

## 4.4 Runtime Storage

CEID now uses a managed runtime storage root for uploaded files, checkpoints, exports, and SQLite state.

Default resolution order:
- `CEID_STORAGE_DIR` if set
- `./.ceid_runtime` in the project root if writable
- system temp directory as a fallback

Useful overrides:
- `CEID_STORAGE_DIR`
- `CEID_EXPORTS_DIR`
- `CEID_CHECKPOINTS_DIR`
- `CEID_DB_PATH`
- `CEID_TEMP_DIR`

Uploads are persisted under the managed storage root so dataset/config/model uploads are not tied to anonymous temp files.

## 4.5 Docker

Build and run:

```powershell
docker build -t ceid-app .
docker run --rm -p 8501:8501 -e CEID_STORAGE_DIR=/data -v ${PWD}/docker-data:/data ceid-app
```

Notes:
- Mount `/data` for persistent checkpoints, exports, and SQLite state.
- Hardware-backed SDL modes require additional device/network access and are not suitable for most hosted container platforms.

## 4.6 Streamlit Cloud / Hosted Streamlit

Entry point:
- `app.py`

Recommended environment variables:
- `CEID_STORAGE_DIR=/tmp/ceid`

Hosted Streamlit is suitable for:
- virtual evaluator workflows
- uploaded datasets/configs/models
- report and ZIP download generation

Hosted Streamlit is not suitable for:
- direct serial/COM hardware access
- embedded SDL hardware control
- guaranteed persistent disk across app restarts unless the platform provides it

## 4.7 VM Deployment

For a Windows or Linux VM:
- create a Python 3.10+ environment
- install `requirements.txt`
- set `CEID_STORAGE_DIR` to a writable persistent folder
- run `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`

For Linux VMs using SDL serial mode, set `SDL_SERIAL_PORT` to the correct device path such as `/dev/ttyUSB0`.

## 5. Data Requirements

### 5.1 Minimum expectations
- Virtual mode usually needs historical samples for meaningful model quality.
- App allows small datasets, but quality warnings appear for low sample counts.

### 5.2 Columns
For configured parameter/objective names:
- Parameter names must match data columns.
- Objective names must match data columns.

### 5.3 Optional uncertainty columns
CEID can read SEM/STD columns and convert to SEM using config.

Defaults include suffix/key matching such as:
- SEM suffixes: `_sem`, `_stderr`, `_se`, `_uncertainty`
- STD suffixes: `_std`, `_stdev`, `_sigma`

Optional replicate count column can be configured for `std -> sem` conversion.

### 5.4 File formats
- Main UI upload currently uses **CSV**.
- Utility loader supports `csv/xlsx/xls/json` programmatically.

## 6. Experiment Configuration (YAML Import/Export)

CEID imports/exports YAML configs from UI.

### 6.1 Required top-level sections
- `experiment_name`
- `parameters`
- `objectives`

Other sections are filled with defaults if omitted.

### 6.2 Parameter schema
Each parameter:
- `name`
- `type`: `continuous | discrete | categorical`
- `bounds` (continuous/discrete)
- `step` (discrete)
- `categories` (categorical)

### 6.3 Objective schema
Each objective:
- `name`
- `type`: `minimize | maximize | target_range | target_value`
- `weight`
- `target_range` (for `target_range`)
- `target_value` + `tolerance` (for `target_value`)

### 6.4 Parameter constraints
Constraint types:
- `sum`
- `order`
- `linear`
- `composition`

Stored as expressions (Ax-compatible constraint strings), e.g.:
- `x1 + x2 <= 10`
- `x1 <= x2`
- `2*x1 + 3*x2 <= 15`

### 6.5 Optimization settings highlights
- `generation_strategy`
- `acquisition_function`
- `initialization_strategy`
- `initialization_trials`
- `batch_iterations`, `batch_size`, `max_iterations`
- `use_adaptive_search` + `adaptive_search_config`
- `use_evolving_constraints` + `evolving_constraints_config`
- `uncertainty_config`
- `distance_normalization_config`

## 7. BO Strategies and Compatibility Notes

Available strategies include:
- `default`, `GPEI`, `SAASBO`, `FULLYBAYESIAN`, `BOTORCH_MODULAR`, `BO_MIXED`
- `ST_MTGP`, `SAAS_MTGP`
- `THOMPSON`, `EMPIRICAL_BAYES_THOMPSON`, `EB_ASHR`, `FACTORIAL`, `UNIFORM`

Key constraints enforced by code:
- MTGP (`ST_MTGP`, `SAAS_MTGP`) requires task parameter and virtual mode.
- Discrete-only strategies reject continuous parameters.
- `FACTORIAL` is capped for very large combinatorics.
- Custom acquisition functions only apply to compatible strategies.

Initialization strategies:
- `none`, `sobol`, `uniform`

Acquisition selection supports single-objective vs multi-objective compatibility checks.

## 8. SDL Integration Details

## 8.1 Connector protocols (`core/sdl.py`)
- `mqtt`, `http`, `serial`, `tcp`, `embedded`

## 8.2 Candidate payload (CEID -> SDL)

```json
{
  "trial_index": 12,
  "parameters": {
    "vol_r": 2,
    "vol_y": 1,
    "vol_b": 1,
    "vol_w": 4
  },
  "control": {
    "digital_twin_control": true,
    "require_continue_each_trial": true
  },
  "ts": 1710000000.0
}
```

## 8.3 Expected response (SDL -> CEID)

CEID accepts both simple and rich payloads. Preferred format:

```json
{
  "trial_index": 12,
  "status": "ok",
  "objectives": {
    "frequency": 743.0
  },
  "objective_uncertainties": {
    "frequency": 1.0
  },
  "uncertainty_type": "sem",
  "measurement_metadata": {
    "frequency_samples_collected": 3,
    "frequency_samples_requested": 3,
    "frequency_sem": 1.0,
    "frequency_std": 1.73
  }
}
```

CEID normalizes uncertainty handling and can parse multiple key styles.

## 8.4 Running external RYB SDL agent

Example TCP:

```powershell
python RYB_SDL.py ^
  --protocol tcp ^
  --tcp-host 0.0.0.0 ^
  --tcp-port 7000 ^
  --arduino-port COM5 ^
  --arduino-baud 9600 ^
  --sensor-replicates 3 ^
  --sensor-uncertainty-type sem ^
  --sensor-timeout 30 ^
  --log-level INFO
```

Example HTTP:

```powershell
python RYB_SDL.py --protocol http --http-host 0.0.0.0 --http-port 8000
```

Then configure CEID SDL settings to match protocol/host/port/endpoint.

## 8.5 Embedded SDL mode

In CEID Setup -> Evaluators -> Self-driving labs:
- choose protocol: `Embedded (RYB SDL in CEID)`
- set embedded Arduino, Unity transport, LiveKit/TCP fields as needed

Note: embedded mode imports `sdl_agent.EmbeddedRYBSDL`; ensure that package/module is available in your workspace.

## 9. RYB_SDL Agent Capabilities (This Repo)

`RYB_SDL.py` provides:
- Headless lab agent for MQTT/HTTP/TCP/Serial server modes
- Motion + relay actuator control
- Continuous Arduino sensor streaming
- Frequency replicate stats (`mean/std/sem`)
- Unity digital twin streaming (LiveKit or TCP)
- Manual vs automatic candidate handling
- Digital twin command loop (start/continue/stop/validate/manual adjust)
- Optional CSV trace logging

## 10. Checkpointing and Resume

CEID checkpoints include:
- Serialized experiment config
- `X`, `Y`, and `Y_sem`
- Optimizer state (including Ax state)
- Progress metadata

UI controls:
- **Start Fresh**
- **Resume / Next Batch**
- **Pause & Save**
- **Stop & Save**
- **Save**

You can upload a checkpoint JSON and continue from saved progress.

## 11. Results, Export, and Database

Generated export bundle includes:
- tables (`candidates`, `history`, `pareto`) as CSV + JSON
- PNG charts (distance/objective/parameter/uncertainty traces, scatter)
- HTML report
- `summary.json`
- zipped bundle (`bundle.zip`)

SQLite support (`utils/database_manager.py`):
- `experiments`
- `candidates`
- `batch_history`

You can initialize/repair DB and browse saved runs directly in UI.

## 12. requirements.txt Scope

`requirements.txt` now includes:
- Core CEID runtime dependencies
- SDL connector dependencies
- Optional model/interop packages used by advanced/custom features
- Hardware-specific optional package marker for FTDI relay tooling

## 13. Troubleshooting

### SDL timeout/disconnect
- Increase `response_timeout` in SDL settings.
- For physical runs, use larger values (e.g., 300s+).

### Missing uncertainty in results
- Check uncertainty config suffixes/keys.
- Verify dataset SEM/STD columns and replicate column mapping.

### Strategy incompatibility errors
- Recheck parameter types, objective setup, and task parameter requirements.
- MTGP and discrete-only strategy constraints are enforced in `core/optimization.py`.

### No virtual results with no dataset
- Upload data or enable custom models.
- Otherwise CEID runs cold-start synthetic objective mode for pipeline testing.

### Embedded mode import failure
- Ensure `sdl_agent` package/module is importable from workspace root.

---

For a complete setup: configure in **Setup**, run in **Optimize**, inspect in **Results**, and archive in **Export & Database**.

