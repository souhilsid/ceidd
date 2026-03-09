# CEID
Cooperative Explorer for Inverse Design platform with Bayesian Optimization and Self-Driving Lab (SDL) integration.

## What This Repository Contains
This repository combines:
- A Streamlit-based CEID optimization platform (`app.py`)
- Core BO engine based on Ax/BoTorch (`core/`)
- SDL communication layer for HTTP/MQTT/TCP/Serial/embedded modes (`core/sdl.py`)
- A headless hardware controller for RYB mixing on an Elegoo-based setup (`RYB_SDL.py`)

The typical workflow is:
1. CEID proposes a candidate formulation.
2. SDL executes physical mixing and sensing.
3. SDL returns measured objective mean and uncertainty.
4. CEID updates optimization and proposes the next candidate.

## Repository Structure
```text
CEID/
  app.py                 # Streamlit UI
  RYB_SDL.py             # Headless hardware SDL agent (printer + relay + Arduino + Unity)
  core/
    config.py            # Config models and validation
    optimization.py      # Ax/BoTorch optimization loop
    evaluators.py        # Virtual/model-based evaluator logic
    sdl.py               # SDL connector and protocol adapters
    models.py            # ML model loading/training utilities
    visualization.py     # Plotting helpers
  utils/
    data_loader.py
    state_manager.py
    reporting.py
    database_manager.py
```

## Requirements
Python 3.10+ is recommended.

Install core dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install streamlit pandas numpy plotly pyyaml requests scikit-learn scipy
pip install ax-platform torch botorch gpytorch
pip install paho-mqtt pyserial
```

Optional dependencies (only if you use these features):
```powershell
pip install optuna xgboost pygam skops onnxruntime joblib
```

## Quick Start
### 1) Run CEID UI
From repository root:
```powershell
streamlit run app.py --server.port 8501
```
Open: `http://localhost:8501`

### 2) Run RYB SDL Agent (example: TCP mode)
In another terminal:
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
  --log-level INFO ^
  --log-file sdl_agent.log
```

HTTP example:
```powershell
python RYB_SDL.py --protocol http --http-host 0.0.0.0 --http-port 8000 --arduino-port COM5 --arduino-baud 9600
```

### 3) Configure CEID SDL Settings
In CEID Advanced/SDL settings:
- `protocol`: match your agent mode (`tcp` or `http`)
- For TCP: set host/port (default in this project: `localhost:7000`)
- For HTTP: set endpoint (default expected by CEID: `http://localhost:8000/bo`)
- Increase `response_timeout` for physical runs (recommended: `300`+ seconds)

## CEID <-> SDL Message Contract
### Candidate request (CEID -> SDL)
```json
{
  "trial_index": 12,
  "parameters": {
    "vol_r": 2,
    "vol_y": 1,
    "vol_b": 1,
    "vol_w": 4
  }
}
```

### Measurement response (SDL -> CEID)
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
    "frequency_std": 1.7320508075688772,
    "frequency_latest_color": "Violet/Purple"
  }
}
```

## Frequency Objective and Uncertainty in `RYB_SDL.py`
`RYB_SDL.py` computes frequency statistics from replicate sensor reads:
- Mean: used as objective value sent to CEID (`objectives.frequency`)
- SEM or STD: sent in `objective_uncertainties.frequency`
- Detailed fields are included in `measurement_metadata`

Formulas used:
- `mean = sum(samples) / n`
- `std = sqrt(sum((x - mean)^2) / (n - 1))` (sample standard deviation, for `n >= 2`)
- `sem = std / sqrt(n)`

Relevant CLI flags:
- `--sensor-replicates` (default `3`)
- `--sensor-sample-interval`
- `--sensor-max-age-sec`
- `--sensor-require-fresh-samples` / `--no-sensor-require-fresh-samples`
- `--sensor-allow-partial-samples` / `--no-sensor-allow-partial-samples`
- `--sensor-uncertainty-type sem|std`
- `--sensor-sem-fallback`

## Recommended Configuration for One Objective (`frequency`)
If your only target objective is frequency:
- Use objective: `frequency`
- Keep uncertainty enabled
- Start with `SEM` uncertainty mode
- Start with `sensor_replicates = 3` (increase to `5` if readings are noisy)
- In CEID distance normalization, keep scaling enabled with conservative clipping

## Unity / Digital Twin Notes
`RYB_SDL.py` supports:
- LiveKit transport (`--unity-transport livekit` + token)
- TCP transport fallback (`--unity-transport tcp`)

Frequency stream and candidate updates are broadcast to Unity when enabled.

## Troubleshooting
### TCP disconnect / `WinError 10054`
Cause: CEID closed socket before SDL reply (usually timeout too short for physical cycle).
Fix:
- Increase CEID `response_timeout` (e.g., `300` to `900` seconds)
- Keep one candidate per request-response cycle

### Very high `std`/`sem`
Likely causes:
- Sensor instability between replicates
- Reading stale/noisy transitions
- Mixed lighting conditions or sample motion
Fix:
- Increase `sensor-replicates`
- Add `--sensor-sample-interval`
- Ensure stable optics and consistent illumination
- Use `--sensor-require-fresh-samples`

### Arduino not detected
Check:
- Correct COM port and baud
- Serial monitor is closed
- `pyserial` installed

## Notes
- The SDL script is headless and intended for hardware-safe operation with explicit run commands.
- Current project includes exported artifacts/checkpoints in-repo for reproducibility.
