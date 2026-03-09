# app.py - COMPLETE FIXED VERSION WITH DUPLICATE UPLOAD PROTECTION

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import io
import json
import base64
import sys
import os
import time
import tempfile
from pathlib import Path

# Add the core directory to the path so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from core.config import (
    ConfigManager,
    ParameterConfig,
    ObjectiveConfig,
    ModelConfig,
    ParameterConstraint,
    ParameterType,
    ObjectiveType,
    BOGenerationStrategy,
    BOAcquisitionFunction,
    BOInitializationStrategy,
    ACQF_CUSTOMIZABLE_STRATEGIES,
    SINGLE_OBJECTIVE_ACQFS,
    MULTI_OBJECTIVE_ACQFS,
    OptimizationConfig,
    EvaluatorType,
    OptimizationMode,
)
from core.models import ModelRegistry, ModelFactory, CustomModelLoader
from core.evaluators import GeneralizedEvaluator, EvaluationMetrics, OPTUNA_AVAILABLE
from core.sdl import SDLConnector, SDLSettings
from core.optimization import BayesianOptimizer, OptimizationResult
from core.visualization import VisualizationEngine
from utils.data_loader import DataLoader
from utils.state_manager import save_json, load_json
from utils.reporting import export_run_artifacts
from utils.database_manager import ExperimentDatabaseManager

class BOPlatform:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.model_registry = ModelRegistry()
        self.model_factory = ModelFactory(self.model_registry)
        self.data_loader = DataLoader()
        self.viz_engine = VisualizationEngine()
        
        self.current_experiment = None
        self.experiment_data = None
        self.evaluator = None
        self.optimizer = None
        self.optimization_result = None
        
        # Initialize session state
        if 'experiment_config' not in st.session_state:
            st.session_state.experiment_config = None
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = None
        if 'parameters' not in st.session_state:
            st.session_state.parameters = []
        if 'objectives' not in st.session_state:
            st.session_state.objectives = []
        if 'models' not in st.session_state:
            st.session_state.models = self._default_builtin_models()
        if 'optimization_progress' not in st.session_state:
            st.session_state.optimization_progress = {
                'current_batch': 0,
                'total_batches': 0,
                'current_trial': None,
                'status': 'idle',
                'candidates_completed': 0,
                'best_distance': float('inf'),
                'start_time': time.time(),
                'current_parameters': {},
                'current_uncertainties': {},
                'current_objectives': {},
                'uses_direct_objectives': False,
                'distance_history': [],
                'batch_history': [],
                'uncertainty_history': [],
                'objective_history': {}
            }
        if 'custom_models' not in st.session_state:
            st.session_state.custom_models = {}
        if 'processed_uploads' not in st.session_state:
            st.session_state.processed_uploads = set()
        if 'custom_model_counter' not in st.session_state:
            st.session_state.custom_model_counter = 0
        if 'parameter_constraints' not in st.session_state:
            st.session_state.parameter_constraints = []
        if 'checkpoint_info' not in st.session_state:
            st.session_state.checkpoint_info = None
        if 'loaded_checkpoint' not in st.session_state:
            st.session_state.loaded_checkpoint = None
        if 'run_status' not in st.session_state:
            st.session_state.run_status = 'idle'
        if 'evaluator_category' not in st.session_state:
            st.session_state.evaluator_category = "Virtual Evaluators (AI models)"
        if 'evaluator_type' not in st.session_state:
            st.session_state.evaluator_type = EvaluatorType.VIRTUAL.value
        if 'sdl_settings' not in st.session_state:
            st.session_state.sdl_settings = SDLSettings().__dict__
        if 'sdl_connector_ready' not in st.session_state:
            st.session_state.sdl_connector_ready = False
        if 'sdl_connector' not in st.session_state:
            st.session_state.sdl_connector = None
        if 'optimization_mode' not in st.session_state:
            st.session_state.optimization_mode = OptimizationMode.BATCH.value
        if 'task_parameter_name' not in st.session_state:
            st.session_state.task_parameter_name = None
        if 'acquisition_function_select' not in st.session_state:
            st.session_state.acquisition_function_select = BOAcquisitionFunction.AUTO.value
        if 'initialization_strategy_select' not in st.session_state:
            st.session_state.initialization_strategy_select = BOInitializationStrategy.SOBOL.value
        if 'initialization_trials_input' not in st.session_state:
            st.session_state.initialization_trials_input = 5
        if 'adaptive_search_config' not in st.session_state:
            st.session_state.adaptive_search_config = {
                "warmup_batches": 2,
                "update_frequency": 1,
                "top_fraction": 0.3,
                "min_candidates": 5,
                "margin_fraction": 0.2,
                "min_relative_span": 0.15,
                "include_experimental": True,
            }
        if 'evolving_constraints_config' not in st.session_state:
            st.session_state.evolving_constraints_config = {
                "schedule_type": "linear",
                "progress_power": 1.0,
                "r2_lower_start": 0.2,
                "r2_lower_end": 0.7,
                "r2_upper_start": 0.965,
                "r2_upper_end": 0.965,
                "nrmse_start": 0.5,
                "nrmse_end": 0.1,
                "static_r2_lower": 0.2,
                "static_r2_upper": 0.965,
                "static_nrmse_threshold": 0.5,
                "constraint_bonus": 0.5,
            }
        if 'uncertainty_config' not in st.session_state:
            st.session_state.uncertainty_config = {
                "enabled": True,
                "fallback_sem": 0.0,
                "min_sem": 0.0,
                "std_mode": "as_sem",
                "default_replicates": 1,
                "replicates_column": "",
                "data_sem_suffixes": ["_sem", "_stderr", "_se", "_uncertainty"],
                "data_std_suffixes": ["_std", "_stdev", "_sigma"],
                "sdl_sem_keys": ["sem", "stderr", "se", "uncertainty"],
                "sdl_std_keys": ["std", "stdev", "sigma"],
                "virtual_sem_scale": 1.0,
            }
        if 'distance_normalization_config' not in st.session_state:
            st.session_state.distance_normalization_config = {
                "enabled": True,
                "method": "quantile",
                "q_low": 0.05,
                "q_high": 0.95,
                "min_scale": 1e-6,
                "clip_component": 10.0,
                "normalize_weight_norm": True,
                "normalize_target_components": False,
                "max_scale_samples": 2000,
            }
        if 'export_root_dir' not in st.session_state:
            st.session_state.export_root_dir = os.path.join(os.getcwd(), "exports")
        if 'export_db_path' not in st.session_state:
            st.session_state.export_db_path = os.path.join(os.getcwd(), "exports", "bo_platform.db")
        if 'last_export_artifacts' not in st.session_state:
            st.session_state.last_export_artifacts = None
        if 'nav_page' not in st.session_state:
            st.session_state.nav_page = "Overview"

    def _default_builtin_models(self) -> List[Dict[str, Any]]:
        """Build default built-in model list based on installed dependencies."""
        available = set(self.model_registry.get_available_models())
        presets = {
            "random_forest": "Random Forest",
            "gaussian_process": "Gaussian Process",
            "xgboost": "XGBoost",
            "svm": "SVM",
            "gam": "GAM",
        }
        defaults: List[Dict[str, Any]] = []
        for model_type in ["random_forest", "gaussian_process", "xgboost"]:
            if model_type in available:
                defaults.append(
                    {
                        "name": presets.get(model_type, model_type),
                        "type": model_type,
                        "enabled": True,
                        "hyperparameters": {},
                    }
                )

        if not defaults and available:
            fallback_type = sorted(available)[0]
            defaults.append(
                {
                    "name": presets.get(fallback_type, fallback_type),
                    "type": fallback_type,
                    "enabled": True,
                    "hyperparameters": {},
                }
            )
        return defaults

    def _workflow_state(self) -> Dict[str, Any]:
        """Compute simple workflow state used for UI guidance."""
        uploaded_data = st.session_state.get("uploaded_data")
        has_data = uploaded_data is not None and len(uploaded_data) > 0
        has_config = st.session_state.get("experiment_config") is not None
        has_result = st.session_state.get("optimization_result") is not None
        return {
            "has_data": has_data,
            "has_config": has_config,
            "has_result": has_result,
            "progress": (int(has_data) + int(has_config) + int(has_result)) / 3.0,
        }

    def _render_workflow_summary(self):
        """Render compact workflow status cards."""
        state = self._workflow_state()
        cols = st.columns(4)
        cols[0].metric("Data", "Loaded" if state["has_data"] else "Optional")
        cols[1].metric("Configuration", "Ready" if state["has_config"] else "Pending")
        cols[2].metric("Optimization", "Done" if state["has_result"] else "Not Run")
        cols[3].metric("Completion", f"{int(state['progress'] * 100)}%")

    def render_sidebar(self):
        """Render the sidebar navigation with logo"""
        st.sidebar.title("CEID Platform")
        
        # Add logo to sidebar
        self._render_logo()

        workflow = self._workflow_state()
        st.sidebar.markdown("### Workflow Status")
        st.sidebar.progress(workflow["progress"])
        st.sidebar.markdown(
            f"Data: **{'Loaded' if workflow['has_data'] else 'Optional'}**  \n"
            f"Config: **{'Ready' if workflow['has_config'] else 'Pending'}**  \n"
            f"Run: **{'Done' if workflow['has_result'] else 'Pending'}**"
        )

        st.sidebar.markdown("### Navigation")
        pages = {
            "Overview": self.render_home,
            "Setup": self.render_build_workspace,
            "Optimize": self.render_optimization,
            "Results & Export": self.render_analysis_workspace,
        }
        options = list(pages.keys())
        current = st.session_state.get("nav_page", options[0])
        index = options.index(current) if current in options else 0
        selection = st.sidebar.radio("Go to", options, index=index, label_visibility="collapsed")
        st.session_state.nav_page = selection
        pages[selection]()

    def _render_logo(self):
        """Render logo in sidebar"""
        try:
            # Try to load SVG logo from multiple possible locations
            possible_logo_paths = [
                "logoside.svg",
                "./logoside.svg", 
                "assets/logoside.svg",
                "./assets/logoside.svg",
                "images/logoside.svg",
                "./images/logoside.svg"
            ]
            
            logo_path = None
            for path in possible_logo_paths:
                if os.path.exists(path):
                    logo_path = path
                    break
            
            if logo_path:
                with open(logo_path, "r") as f:
                    svg_logo = f.read()
                
                # Display SVG in sidebar
                st.sidebar.markdown(
                    f"""
                    <div style="text-align: left; margin-bottom: 20px;">
                        {svg_logo}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                # Fallback: display text logo
                st.sidebar.markdown(
                    """
                    <div style="text-align: center; margin-bottom: 20px;">
                        <h3>BO Platform</h3>
                        <p style="font-size: 0.8em; color: #666;">Advanced Bayesian Optimization</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        except Exception as e:
            # Simple text fallback
            st.sidebar.markdown("### BO Platform")

    def render_home(self):
        """Render workflow-first landing page."""
        st.title("Overview")
        st.caption(
            "Use the guided flow: Setup -> Optimize -> Results & Export."
        )
        self._render_workflow_summary()

        st.markdown("### Recommended Sequence")
        st.markdown(
            "1. Open `Setup` to upload data (optional) and configure experiment settings.\n"
            "2. Open `Optimize` to execute sequential or batch optimization.\n"
            "3. Open `Results & Export` to inspect results, generate reports, and save to database."
        )

        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            if st.button("Go to Setup", use_container_width=True, key="go_build_btn"):
                st.session_state.nav_page = "Setup"
                st.rerun()
        with nav_col2:
            if st.button("Go to Optimize", use_container_width=True, key="go_run_btn"):
                st.session_state.nav_page = "Optimize"
                st.rerun()
        with nav_col3:
            if st.button("Go to Results & Export", use_container_width=True, key="go_analyze_btn"):
                st.session_state.nav_page = "Results & Export"
                st.rerun()

        st.markdown("### Quick Templates")
        template_col1, template_col2, template_col3 = st.columns(3)
        with template_col1:
            if st.button("Material Science", use_container_width=True, key="tpl_material"):
                self.load_template("material_science", trigger_rerun=False)
                st.session_state.nav_page = "Setup"
                st.rerun()
        with template_col2:
            if st.button("Drug Discovery", use_container_width=True, key="tpl_drug"):
                self.load_template("drug_discovery", trigger_rerun=False)
                st.session_state.nav_page = "Setup"
                st.rerun()
        with template_col3:
            if st.button("Process Optimization", use_container_width=True, key="tpl_process"):
                self.load_template("process_optimization", trigger_rerun=False)
                st.session_state.nav_page = "Setup"
                st.rerun()

    def render_build_workspace(self):
        """Guided build page: data + setup."""
        st.header("Setup")
        st.caption("Data upload is optional. Configure experiment settings below.")
        self._render_workflow_summary()
        st.markdown("### Data (Optional)")
        self.render_data_upload(show_header=False)
        st.divider()
        st.markdown("### Experiment Configuration")
        self.render_experiment_setup(show_header=False)

    def render_analysis_workspace(self):
        """Guided analysis page: results + export."""
        st.header("Results & Export")
        st.caption("Review optimization outputs and generate persistent artifacts.")
        self._render_workflow_summary()
        st.markdown("### Results")
        self.render_results(show_header=False)
        st.divider()
        st.markdown("### Export and Database")
        self.render_export(show_header=False)

    def load_template(self, template_name: str, trigger_rerun: bool = True):
        """Load a pre-configured template"""
        templates = {
            "material_science": {
                "parameters": [
                    {"name": "composition_a", "type": "continuous", "bounds": [0, 100]},
                    {"name": "composition_b", "type": "continuous", "bounds": [0, 100]},
                    {"name": "temperature", "type": "continuous", "bounds": [100, 500]},
                    {"name": "pressure", "type": "continuous", "bounds": [1, 100]}
                ],
                "objectives": [
                    {"name": "strength", "type": "maximize", "weight": 1.0},
                    {"name": "conductivity", "type": "target_range", "target_range": [10, 50], "weight": 0.8}
                ]
            },
            "drug_discovery": {
                "parameters": [
                    {"name": "molecular_weight", "type": "continuous", "bounds": [100, 800]},
                    {"name": "logp", "type": "continuous", "bounds": [-2, 6]},
                    {"name": "hbd", "type": "discrete", "bounds": [0, 10], "step": 1},
                    {"name": "hba", "type": "discrete", "bounds": [0, 15], "step": 1}
                ],
                "objectives": [
                    {"name": "activity", "type": "maximize", "weight": 1.0},
                    {"name": "solubility", "type": "minimize", "weight": 0.7},
                    {"name": "toxicity", "type": "minimize", "weight": 0.9}
                ]
            },
            "process_optimization": {
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [50, 200]},
                    {"name": "pressure", "type": "continuous", "bounds": [1, 10]},
                    {"name": "flow_rate", "type": "continuous", "bounds": [10, 100]},
                    {"name": "catalyst", "type": "categorical", "categories": ["A", "B", "C", "D"]}
                ],
                "objectives": [
                    {"name": "yield", "type": "maximize", "weight": 1.0},
                    {"name": "purity", "type": "target_range", "target_range": [95, 99], "weight": 0.9},
                    {"name": "cost", "type": "minimize", "weight": 0.6}
                ]
            }
        }
        
        if template_name in templates:
            st.session_state.template_config = templates[template_name]
            # Apply template to parameters and objectives
            st.session_state.parameters = templates[template_name]["parameters"]
            st.session_state.objectives = templates[template_name]["objectives"]
            st.success(f":material/check_circle: Loaded {template_name.replace('_', ' ').title()} template.")
            if trigger_rerun:
                st.rerun()

    def render_data_upload(self, show_header: bool = True):
        """Render data upload section"""
        if show_header:
            st.header("Data Upload")
        st.info("Data upload is optional. When using Self-driving Labs you can start without a dataset; Sobol points and live measurements will seed the optimizer.")
        
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV)", 
            type=['csv'],
            help="Upload a CSV file with your experimental data. At least 5 rows are recommended."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Keep upload flexible: allow small datasets but flag lower confidence.
                if len(df) < 5:
                    st.warning(
                        f"Dataset has {len(df)} rows. This is allowed, but model quality may be unstable."
                    )
                
                st.session_state.uploaded_data = df
                
                st.success(f":material/check_circle: Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                
                # Show basic statistics
                st.subheader("Basic Statistics")
                st.dataframe(df.describe())
                
                # Column information
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(col_info)
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        elif st.session_state.get('uploaded_data') is not None:
            df = st.session_state.uploaded_data
            st.info(f"Using previously uploaded dataset: {df.shape[0]} rows Ã— {df.shape[1]} columns")
            st.dataframe(df.head(5))

    def _default_uncertainty_config(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "fallback_sem": 0.0,
            "min_sem": 0.0,
            "std_mode": "as_sem",
            "default_replicates": 1,
            "replicates_column": "",
            "data_sem_suffixes": ["_sem", "_stderr", "_se", "_uncertainty"],
            "data_std_suffixes": ["_std", "_stdev", "_sigma"],
            "sdl_sem_keys": ["sem", "stderr", "se", "uncertainty"],
            "sdl_std_keys": ["std", "stdev", "sigma"],
            "virtual_sem_scale": 1.0,
        }

    def _split_csv_items(self, raw_value: Any, fallback: List[str]) -> List[str]:
        if isinstance(raw_value, list):
            values = [str(v).strip() for v in raw_value if str(v).strip()]
            return values or fallback
        if isinstance(raw_value, str):
            values = [v.strip() for v in raw_value.split(",") if v.strip()]
            return values or fallback
        return fallback

    def _resolve_uncertainty_config(self, config_value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resolved = self._default_uncertainty_config()
        if isinstance(config_value, dict):
            resolved.update(config_value)
        resolved["data_sem_suffixes"] = self._split_csv_items(
            resolved.get("data_sem_suffixes"),
            self._default_uncertainty_config()["data_sem_suffixes"],
        )
        resolved["data_std_suffixes"] = self._split_csv_items(
            resolved.get("data_std_suffixes"),
            self._default_uncertainty_config()["data_std_suffixes"],
        )
        resolved["sdl_sem_keys"] = self._split_csv_items(
            resolved.get("sdl_sem_keys"),
            self._default_uncertainty_config()["sdl_sem_keys"],
        )
        resolved["sdl_std_keys"] = self._split_csv_items(
            resolved.get("sdl_std_keys"),
            self._default_uncertainty_config()["sdl_std_keys"],
        )
        resolved["std_mode"] = str(resolved.get("std_mode", "as_sem")).lower()
        resolved["default_replicates"] = int(max(1, int(resolved.get("default_replicates", 1))))
        resolved["fallback_sem"] = max(0.0, float(resolved.get("fallback_sem", 0.0)))
        resolved["min_sem"] = max(0.0, float(resolved.get("min_sem", 0.0)))
        resolved["virtual_sem_scale"] = max(1e-9, float(resolved.get("virtual_sem_scale", 1.0)))
        resolved["replicates_column"] = str(resolved.get("replicates_column", "") or "").strip()
        resolved["enabled"] = bool(resolved.get("enabled", True))
        return resolved

    def _default_distance_normalization_config(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "method": "quantile",
            "q_low": 0.05,
            "q_high": 0.95,
            "min_scale": 1e-6,
            "clip_component": 10.0,
            "normalize_weight_norm": True,
            "normalize_target_components": False,
            "max_scale_samples": 2000,
        }

    def _resolve_distance_normalization_config(
        self,
        config_value: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved = self._default_distance_normalization_config()
        if isinstance(config_value, dict):
            resolved.update(config_value)

        resolved["enabled"] = bool(resolved.get("enabled", True))
        method = str(resolved.get("method", "quantile")).lower()
        if method not in {"quantile", "range", "std"}:
            method = "quantile"
        resolved["method"] = method

        q_low = float(resolved.get("q_low", 0.05))
        q_high = float(resolved.get("q_high", 0.95))
        q_low = min(max(q_low, 0.0), 1.0)
        q_high = min(max(q_high, 0.0), 1.0)
        if q_high <= q_low:
            q_low, q_high = 0.05, 0.95
        resolved["q_low"] = q_low
        resolved["q_high"] = q_high
        resolved["min_scale"] = max(1e-12, float(resolved.get("min_scale", 1e-6)))
        resolved["clip_component"] = max(0.0, float(resolved.get("clip_component", 10.0)))
        resolved["normalize_weight_norm"] = bool(resolved.get("normalize_weight_norm", True))
        resolved["normalize_target_components"] = bool(resolved.get("normalize_target_components", False))
        resolved["max_scale_samples"] = max(10, int(resolved.get("max_scale_samples", 2000)))
        return resolved

    def _extract_objective_sem_matrix(
        self,
        df: pd.DataFrame,
        objective_columns: List[str],
        uncertainty_config: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        if df is None or len(df) == 0 or not objective_columns:
            return np.zeros((0, len(objective_columns)), dtype=np.float64)

        cfg = self._resolve_uncertainty_config(uncertainty_config)
        sem_matrix = np.full((len(df), len(objective_columns)), np.nan, dtype=np.float64)
        if not cfg.get("enabled", True):
            return sem_matrix

        std_mode = str(cfg.get("std_mode", "as_sem")).lower()
        replicate_col = str(cfg.get("replicates_column", "") or "").strip()
        default_replicates = max(1, int(cfg.get("default_replicates", 1)))
        min_sem = max(0.0, float(cfg.get("min_sem", 0.0)))
        lower_to_original = {str(col).lower(): str(col) for col in df.columns}

        def _find_by_suffix(obj_name: str, suffixes: List[str]) -> Optional[str]:
            for suffix in suffixes:
                candidate = f"{obj_name}{suffix}"
                if candidate in df.columns:
                    return candidate
                candidate_l = candidate.lower()
                if candidate_l in lower_to_original:
                    return lower_to_original[candidate_l]
            return None

        def _auto_match_uncertainty_column(obj_name: str, mode: str) -> Optional[str]:
            name = str(obj_name or "").strip()
            if not name:
                return None
            name_l = name.lower()
            reps = ["uncertainty", "sem", "stderr", "se"] if mode == "sem" else ["std", "stdev", "sigma"]
            generated: List[str] = []

            # common naming transforms: *_mean_* -> *_uncertainty_* (or *_std_*), etc.
            for marker in ["_mean_", "_avg_", "_average_"]:
                if marker in name_l:
                    for rep in reps:
                        generated.append(name_l.replace(marker, f"_{rep}_"))

            for suffix in ["_mean", "_avg", "_average"]:
                if name_l.endswith(suffix):
                    stem = name_l[: -len(suffix)]
                    for rep in reps:
                        generated.append(f"{stem}_{rep}")

            for prefix in ["mean_", "avg_", "average_"]:
                if name_l.startswith(prefix):
                    stem = name_l[len(prefix):]
                    for rep in reps:
                        generated.append(f"{rep}_{stem}")

            for rep in reps:
                generated.append(f"{name_l}_{rep}")
                generated.append(name_l.replace("mean", rep))

            seen = set()
            for cand in generated:
                if cand in seen:
                    continue
                seen.add(cand)
                if cand in lower_to_original:
                    return lower_to_original[cand]

            # relaxed fallback: shared prefix token + uncertainty keyword.
            core_token = name_l.split("_")[0]
            if core_token:
                keyword_matches = [
                    col for col in df.columns
                    if core_token in str(col).lower() and any(rep in str(col).lower() for rep in reps)
                ]
                if len(keyword_matches) == 1:
                    return keyword_matches[0]

            return None

        replicate_values = None
        if replicate_col and replicate_col in df.columns:
            replicate_values = pd.to_numeric(df[replicate_col], errors="coerce").fillna(default_replicates).clip(lower=1.0).values

        for j, obj_name in enumerate(objective_columns):
            sem_col = _find_by_suffix(obj_name, cfg.get("data_sem_suffixes", []))
            if sem_col is None:
                sem_col = _auto_match_uncertainty_column(obj_name, mode="sem")

            if sem_col:
                sem_vals = pd.to_numeric(df[sem_col], errors="coerce").values
                sem_matrix[:, j] = sem_vals
                continue

            std_col = _find_by_suffix(obj_name, cfg.get("data_std_suffixes", []))
            if std_col is None:
                std_col = _auto_match_uncertainty_column(obj_name, mode="std")

            if std_col:
                std_vals = pd.to_numeric(df[std_col], errors="coerce").values
                if std_mode == "std_to_sem":
                    reps = replicate_values if replicate_values is not None else np.full(len(df), float(default_replicates))
                    sem_vals = std_vals / np.sqrt(reps)
                else:
                    sem_vals = std_vals
                sem_matrix[:, j] = sem_vals

        sem_matrix = np.abs(sem_matrix.astype(np.float64))
        finite_mask = np.isfinite(sem_matrix)
        sem_matrix[finite_mask] = np.maximum(sem_matrix[finite_mask], min_sem)
        return sem_matrix

    def _update_ui_from_config(self, config: OptimizationConfig):
     """Update UI session state from loaded configuration"""
    
    # Update experiment name
     st.session_state.experiment_name_input = config.experiment_name
    
    # Update parameters
     st.session_state.parameters = []
     for param in config.parameters:
        param_dict = {
            'name': param.name,
            'type': param.type.value,
            'bounds': param.bounds,
            'categories': param.categories,
            'step': param.step
        }
        st.session_state.parameters.append(param_dict)
    
    # Update objectives
     st.session_state.objectives = []
     for obj in config.objectives:
        obj_dict = {
            'name': obj.name,
            'type': obj.type.value,
            'target_range': obj.target_range,
            'target_value': obj.target_value,
            'tolerance': obj.tolerance,
            'weight': obj.weight
        }
        st.session_state.objectives.append(obj_dict)
    
    # Update models
     st.session_state.models = []
     for model in config.models:
        model_dict = {
            'name': model.name,
            'type': model.type,
            'hyperparameters': model.hyperparameters,
            'enabled': model.enabled
        }
        st.session_state.models.append(model_dict)

     available_model_types = set(self.model_registry.get_available_models())
     st.session_state.models = [
        m for m in st.session_state.models
        if m.get("type") == "custom_model" or m.get("type") in available_model_types
     ]
     if not st.session_state.models:
        st.session_state.models = self._default_builtin_models()
    
    # Update parameter constraints
     st.session_state.parameter_constraints = []
     for constraint in config.parameter_constraints:
        constraint_dict = {
            'name': constraint.name,
            'type': constraint.type,
            'expression': constraint.expression,
            'description': constraint.description
        }
        st.session_state.parameter_constraints.append(constraint_dict)
    
    # Update optimization settings in session state
     optimization_mappings = {
        'enable_tuning_checkbox': config.enable_hyperparameter_tuning,
        'tuning_trials_slider': config.n_tuning_trials,
        'batch_iterations_input': config.batch_iterations,
        'batch_size_input': config.batch_size,
        'random_seed_input': config.random_seed,
        'generation_strategy_select': config.generation_strategy.value,
        'acquisition_function_select': (
            config.acquisition_function.value
            if hasattr(config.acquisition_function, 'value')
            else config.acquisition_function
        ),
        'initialization_strategy_select': (
            config.initialization_strategy.value
            if hasattr(config.initialization_strategy, 'value')
            else config.initialization_strategy
        ),
        'initialization_trials_input': getattr(config, 'initialization_trials', config.sobol_points),
        'use_sobol_checkbox': config.use_sobol,
        'sobol_points_input': config.sobol_points,
        'adaptive_search_checkbox': config.use_adaptive_search,
        'evolving_constraints_checkbox': config.use_evolving_constraints
    }
     st.session_state.adaptive_search_config = (
        config.adaptive_search_config if isinstance(config.adaptive_search_config, dict)
        else st.session_state.get('adaptive_search_config', {})
     )
     st.session_state.evolving_constraints_config = (
        config.evolving_constraints_config if isinstance(config.evolving_constraints_config, dict)
        else st.session_state.get('evolving_constraints_config', {})
     )
     st.session_state.uncertainty_config = (
        config.uncertainty_config if isinstance(getattr(config, "uncertainty_config", None), dict)
        else self._default_uncertainty_config()
     )
     st.session_state.distance_normalization_config = (
        config.distance_normalization_config if isinstance(getattr(config, "distance_normalization_config", None), dict)
        else self._default_distance_normalization_config()
     )
     adaptive_cfg = st.session_state.adaptive_search_config
     evolving_cfg = st.session_state.evolving_constraints_config
     uncertainty_cfg = st.session_state.uncertainty_config
     distance_cfg = st.session_state.distance_normalization_config
     st.session_state.adaptive_warmup_batches_input = int(adaptive_cfg.get("warmup_batches", 2))
     st.session_state.adaptive_update_frequency_input = int(adaptive_cfg.get("update_frequency", 1))
     st.session_state.adaptive_top_fraction_input = float(adaptive_cfg.get("top_fraction", 0.3))
     st.session_state.adaptive_min_candidates_input = int(adaptive_cfg.get("min_candidates", 5))
     st.session_state.adaptive_margin_fraction_input = float(adaptive_cfg.get("margin_fraction", 0.2))
     st.session_state.adaptive_min_relative_span_input = float(adaptive_cfg.get("min_relative_span", 0.15))
     st.session_state.adaptive_include_experimental_checkbox = bool(adaptive_cfg.get("include_experimental", True))
     st.session_state.static_r2_lower_input = float(evolving_cfg.get("static_r2_lower", 0.2))
     st.session_state.static_r2_upper_input = float(evolving_cfg.get("static_r2_upper", 0.965))
     st.session_state.static_nrmse_threshold_input = float(evolving_cfg.get("static_nrmse_threshold", 0.5))
     st.session_state.constraint_bonus_input = float(evolving_cfg.get("constraint_bonus", 0.5))
     st.session_state.evolving_schedule_type_select = str(evolving_cfg.get("schedule_type", "linear")).lower()
     st.session_state.evolving_progress_power_input = float(evolving_cfg.get("progress_power", 1.0))
     st.session_state.evolving_r2_lower_start_input = float(evolving_cfg.get("r2_lower_start", 0.2))
     st.session_state.evolving_r2_lower_end_input = float(evolving_cfg.get("r2_lower_end", 0.7))
     st.session_state.evolving_r2_upper_start_input = float(evolving_cfg.get("r2_upper_start", 0.965))
     st.session_state.evolving_r2_upper_end_input = float(evolving_cfg.get("r2_upper_end", 0.965))
     st.session_state.evolving_nrmse_start_input = float(evolving_cfg.get("nrmse_start", 0.5))
     st.session_state.evolving_nrmse_end_input = float(evolving_cfg.get("nrmse_end", 0.1))
     st.session_state.uncertainty_enabled_checkbox = bool(uncertainty_cfg.get("enabled", True))
     st.session_state.uncertainty_fallback_sem_input = float(uncertainty_cfg.get("fallback_sem", 0.0))
     st.session_state.uncertainty_min_sem_input = float(uncertainty_cfg.get("min_sem", 0.0))
     st.session_state.uncertainty_std_mode_select = str(uncertainty_cfg.get("std_mode", "as_sem")).lower()
     st.session_state.uncertainty_default_replicates_input = int(uncertainty_cfg.get("default_replicates", 1))
     replicate_col = str(uncertainty_cfg.get("replicates_column", "")).strip()
     uploaded_df = st.session_state.get("uploaded_data")
     if uploaded_df is not None and replicate_col and replicate_col in uploaded_df.columns:
        st.session_state.uncertainty_replicates_column_select = replicate_col
     else:
        st.session_state.uncertainty_replicates_column_select = "(none)"
     st.session_state.uncertainty_sem_suffixes_input = ",".join(uncertainty_cfg.get("data_sem_suffixes", ["_sem", "_stderr", "_se", "_uncertainty"]))
     st.session_state.uncertainty_std_suffixes_input = ",".join(uncertainty_cfg.get("data_std_suffixes", ["_std", "_stdev", "_sigma"]))
     st.session_state.uncertainty_sdl_sem_keys_input = ",".join(uncertainty_cfg.get("sdl_sem_keys", ["sem", "stderr", "se", "uncertainty"]))
     st.session_state.uncertainty_sdl_std_keys_input = ",".join(uncertainty_cfg.get("sdl_std_keys", ["std", "stdev", "sigma"]))
     st.session_state.uncertainty_virtual_scale_input = float(uncertainty_cfg.get("virtual_sem_scale", 1.0))
     st.session_state.distance_norm_enabled_checkbox = bool(distance_cfg.get("enabled", True))
     st.session_state.distance_norm_method_select = str(distance_cfg.get("method", "quantile")).lower()
     st.session_state.distance_norm_q_low_input = float(distance_cfg.get("q_low", 0.05))
     st.session_state.distance_norm_q_high_input = float(distance_cfg.get("q_high", 0.95))
     st.session_state.distance_norm_min_scale_input = float(distance_cfg.get("min_scale", 1e-6))
     st.session_state.distance_norm_clip_input = float(distance_cfg.get("clip_component", 10.0))
     st.session_state.distance_norm_weight_checkbox = bool(distance_cfg.get("normalize_weight_norm", True))
     st.session_state.distance_norm_targets_checkbox = bool(distance_cfg.get("normalize_target_components", False))
     st.session_state.distance_norm_max_samples_input = int(distance_cfg.get("max_scale_samples", 2000))
     st.session_state.task_parameter_name = config.task_parameter_name
    
     for key, value in optimization_mappings.items():
        st.session_state[key] = value
    
     # Store the loaded config
     st.session_state.experiment_config = config
    
     # Show import summary
     st.info(f"""
     **Configuration Imported Successfully:**
     - **Experiment**: {config.experiment_name}
     - **Parameters**: {len(config.parameters)}
     - **Objectives**: {len(config.objectives)}
     - **Parameter Constraints**: {len(config.parameter_constraints)}
     - **Task Parameter**: {config.task_parameter_name if config.task_parameter_name else 'N/A'}
     - **Models**: {len([m for m in config.models if m.enabled])} enabled
     - **Strategy**: {config.generation_strategy.value}
     - **Initialization**: {(config.initialization_strategy.value if hasattr(config.initialization_strategy, 'value') else config.initialization_strategy)} ({getattr(config, 'initialization_trials', config.sobol_points)} trials)
     - **Acquisition**: {config.acquisition_function.value if hasattr(config.acquisition_function, 'value') else config.acquisition_function}
     """)

    def _import_configuration(self, uploaded_config):
     """Import configuration from YAML file and update UI state with robust error handling"""
     try:
        # Read the uploaded config file
        config_content = uploaded_config.getvalue().decode('utf-8')
        
        # Create a temporary file to load the config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_file.write(config_content)
            tmp_file_path = tmp_file.name
        
        try:
            # First, validate the YAML structure
            with open(tmp_file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Check for required sections
            required_sections = ['experiment_name', 'parameters', 'objectives']
            missing_sections = [section for section in required_sections if section not in config_dict]
            
            if missing_sections:
                st.error(f":material/error: Missing required sections in configuration: {', '.join(missing_sections)}")
                st.info("The configuration file must contain at least: experiment_name, parameters, and objectives")
                return
            
            # Check for models section and provide defaults if missing
            if 'models' not in config_dict:
                st.warning(":material/warning: No models section found in configuration. Using default models.")
                config_dict['models'] = self._default_builtin_models()
            
            # Check for optimization_settings and provide defaults if missing
            if 'optimization_settings' not in config_dict:
                st.warning(":material/warning: No optimization_settings found. Using default settings.")
                config_dict['optimization_settings'] = {
                    'enable_hyperparameter_tuning': True,
                    'n_tuning_trials': 20,
                    'batch_iterations': 10,
                    'batch_size': 5,
                    'max_iterations': 100,
                    'random_seed': 42,
                    'n_initial_points': 10,
                    'generation_strategy': 'default',
                    'acquisition_function': 'auto',
                    'initialization_strategy': 'sobol',
                    'initialization_trials': 10,
                    'use_sobol': True,
                    'sobol_points': 10,
                    'use_adaptive_search': True,
                    'adaptive_search_config': {
                        "warmup_batches": 2,
                        "update_frequency": 1,
                        "top_fraction": 0.3,
                        "min_candidates": 5,
                        "margin_fraction": 0.2,
                        "min_relative_span": 0.15,
                        "include_experimental": True,
                    },
                    'use_evolving_constraints': False,
                    'evolving_constraints_config': {
                        "schedule_type": "linear",
                        "progress_power": 1.0,
                        "r2_lower_start": 0.2,
                        "r2_lower_end": 0.7,
                        "r2_upper_start": 0.965,
                        "r2_upper_end": 0.965,
                        "nrmse_start": 0.5,
                        "nrmse_end": 0.1,
                        "static_r2_lower": 0.2,
                        "static_r2_upper": 0.965,
                        "static_nrmse_threshold": 0.5,
                        "constraint_bonus": 0.5,
                    },
                    'uncertainty_config': {
                        "enabled": True,
                        "fallback_sem": 0.0,
                        "min_sem": 0.0,
                        "std_mode": "as_sem",
                        "default_replicates": 1,
                        "replicates_column": "",
                        "data_sem_suffixes": ["_sem", "_stderr", "_se", "_uncertainty"],
                        "data_std_suffixes": ["_std", "_stdev", "_sigma"],
                        "sdl_sem_keys": ["sem", "stderr", "se", "uncertainty"],
                        "sdl_std_keys": ["std", "stdev", "sigma"],
                        "virtual_sem_scale": 1.0,
                    },
                    'distance_normalization_config': {
                        "enabled": True,
                        "method": "quantile",
                        "q_low": 0.05,
                        "q_high": 0.95,
                        "min_scale": 1e-6,
                        "clip_component": 10.0,
                        "normalize_weight_norm": True,
                        "normalize_target_components": False,
                        "max_scale_samples": 2000,
                    },
                }
            
            # Check for parameter_constraints
            if 'parameter_constraints' not in config_dict:
                config_dict['parameter_constraints'] = []
            
            # Write the updated config back to temporary file
            with open(tmp_file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            # Now load the configuration using ConfigManager
            config = self.config_manager.load_config(tmp_file_path)
            
            # Update session state with loaded configuration
            self._update_ui_from_config(config)
            
            st.success(f":material/check_circle: Imported configuration: {config.experiment_name}")
            st.rerun()
            
        except Exception as e:
            st.error(f":material/error: Error processing configuration: {e}")
            st.info("Please check that the YAML file is properly formatted.")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
     except Exception as e:
        st.error(f":material/error: Error importing configuration: {e}")
        st.error("Please ensure the YAML file is properly formatted and contains valid configuration.")

    def render_experiment_setup(self, show_header: bool = True):
     """Render experiment configuration section - FIXED DUPLICATE UPLOAD ISSUES"""
     if show_header:
        st.header("Experiment Setup")
    
     df = st.session_state.get('uploaded_data')
     columns = df.columns.tolist() if df is not None else []
     if df is None:
        st.info("No dataset uploaded yet. Configure parameters/objectives manually or upload data later (SDL mode works without historical data).")

     enabled_models = [m for m in st.session_state.models if m.get("enabled", True)]
     summary_cols = st.columns(4)
     summary_cols[0].metric("Parameters", len(st.session_state.parameters))
     summary_cols[1].metric("Objectives", len(st.session_state.objectives))
     summary_cols[2].metric("Enabled Models", len(enabled_models))
     summary_cols[3].metric("Data Rows", 0 if df is None else len(df))
    
    # NEW: Import Configuration Section
     st.subheader(":material/upload_file: Import Configuration")
     col1, col2 = st.columns([3, 1])
    
     with col1:
        uploaded_config = st.file_uploader(
            "Upload YAML Configuration",
            type=['yaml', 'yml'],
            help="Upload a previously exported YAML configuration file",
            key="import_config_setup_uploader"
        )
    
     with col2:
        if st.button(":material/sync: Load Configuration", use_container_width=True, key="load_config_setup_button"):
            if uploaded_config is not None:
                self._import_configuration(uploaded_config)
            else:
                st.warning("Please upload a YAML configuration file first")
    
    # Experiment basic info
     st.subheader("Experiment Information")
     exp_name = st.text_input("Experiment Name", value="My_BO_Experiment", key="experiment_name_input")
    
    # Parameter configuration
     st.subheader(":material/tune: Parameter Configuration")
     st.markdown("Define your input parameters (features)")
     if columns:
        st.caption("Tip: Select parameter columns directly from your uploaded dataset.")
    
    # Add parameter button
     col1, col2 = st.columns([3, 1])
     with col2:
        if st.button(":material/add_circle: Add Parameter", key="add_param_button"):
            default_param_name = f'param_{len(st.session_state.parameters) + 1}'
            if columns:
                used_param_names = {p.get('name') for p in st.session_state.parameters}
                available_param_columns = [c for c in columns if c not in used_param_names]
                if available_param_columns:
                    default_param_name = available_param_columns[0]
            st.session_state.parameters.append({
                'name': default_param_name,
                'type': 'continuous',
                'bounds': [0.0, 1.0]
            })
    
    # Display parameters
     parameters_to_remove = []
     for i, param in enumerate(st.session_state.parameters):
        with st.expander(f"Parameter: {param['name']}", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                current_param_name = param.get('name', f'param_{i + 1}')
                if columns:
                    param_column_options = ["<custom>"] + columns
                    selected_param_option = st.selectbox(
                        "Parameter Column",
                        options=param_column_options,
                        index=param_column_options.index(current_param_name) if current_param_name in param_column_options else 0,
                        key=f"param_col_select_{i}",
                        help="Pick a dataset column or choose <custom> to type your own name.",
                    )
                    if selected_param_option == "<custom>":
                        manual_default = current_param_name if current_param_name not in columns else f'param_{i + 1}'
                        param['name'] = st.text_input(
                            "Custom Parameter Name",
                            value=manual_default,
                            key=f"param_name_custom_{i}"
                        )
                    else:
                        param['name'] = selected_param_option
                else:
                    param['name'] = st.text_input(
                        "Parameter Name",
                        value=current_param_name,
                        key=f"param_name_{i}"
                    )
            
            with col2:
                param['type'] = st.selectbox(
                    "Type",
                    options=[t.value for t in ParameterType],
                    index=[t.value for t in ParameterType].index(param['type']),
                    key=f"param_type_{i}"
                )
            
            with col3:
                if st.button(":material/delete:", key=f"remove_param_{i}"):
                    parameters_to_remove.append(i)
            
            # Type-specific configuration
            if param['type'] == 'continuous':
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.number_input("Min", value=float(param.get('bounds', [0, 1])[0]), key=f"param_min_{i}")
                with col2:
                    max_val = st.number_input("Max", value=float(param.get('bounds', [0, 1])[1]), key=f"param_max_{i}")
                param['bounds'] = [min_val, max_val]
            
            elif param['type'] == 'categorical':
                if df is not None and param['name'] in df.columns:
                    unique_vals = df[param['name']].unique()
                    param['categories'] = st.multiselect(
                        "Categories",
                        options=unique_vals,
                        default=unique_vals[:min(5, len(unique_vals))],
                        key=f"param_categories_{i}"
                    )
                else:
                    category_default = ",".join(param.get('categories', [])) if param.get('categories') else ""
                    cat_text = st.text_input(
                        "Categories (comma separated)",
                        value=category_default,
                        help="Provide categories manually when no dataset column is available",
                        key=f"param_categories_text_{i}"
                    )
                    param['categories'] = [c.strip() for c in cat_text.split(",") if c.strip()]
                    if not param['categories']:
                        st.warning("Enter at least one category for this parameter")
            
            elif param['type'] == 'discrete':
                col1, col2, col3 = st.columns(3)
                with col1:
                    min_val = st.number_input("Min", value=float(param.get('bounds', [0, 10])[0]), key=f"disc_min_{i}")
                with col2:
                    max_val = st.number_input("Max", value=float(param.get('bounds', [0, 10])[1]), key=f"disc_max_{i}")
                with col3:
                    step = st.number_input("Step", value=float(param.get('step', 1)), key=f"disc_step_{i}")
                param['bounds'] = [min_val, max_val]
                param['step'] = step
    
    # Remove marked parameters
     for i in sorted(parameters_to_remove, reverse=True):
        st.session_state.parameters.pop(i)
    
    # NEW: Parameter Constraints Section - UPDATED VERSION
     st.subheader(":material/device_hub: Parameter Constraints")
     st.markdown("Define relationships between parameters (sum, order, linear, composition)")
    
    # Add constraint button
     col1, col2 = st.columns([3, 1])
     with col2:
        if st.button(":material/add_circle: Add Constraint", key="add_constraint"):
            if 'parameter_constraints' not in st.session_state:
                st.session_state.parameter_constraints = []
            st.session_state.parameter_constraints.append({
                'name': f'constraint_{len(st.session_state.parameter_constraints) + 1}',
                'type': 'sum',
                'expression': '',
                'description': ''
            })
    
    # Display constraints
     constraints_to_remove = []
     if 'parameter_constraints' in st.session_state:
        for i, constraint in enumerate(st.session_state.parameter_constraints):
            with st.expander(f"Constraint: {constraint['name']}", expanded=True):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    constraint['name'] = st.text_input(
                        "Constraint Name",
                        value=constraint['name'],
                        key=f"constraint_name_{i}"
                    )
                
                with col2:
                    old_type = constraint['type']
                    constraint['type'] = st.selectbox(
                        "Constraint Type",
                        options=['sum', 'order', 'linear', 'composition'],
                        index=['sum', 'order', 'linear', 'composition'].index(constraint['type']),
                        key=f"constraint_type_{i}"
                    )
                    # Force expression update when type changes
                    if constraint['type'] != old_type:
                        constraint['expression'] = ""
                
                with col3:
                    if st.button(":material/delete:", key=f"remove_constraint_{i}"):
                        constraints_to_remove.append(i)
                
                # Constraint expression based on type - DYNAMIC UPDATES
                param_names = [p['name'] for p in st.session_state.parameters]
                
                if constraint['type'] == 'sum':
                    st.info("Sum Constraint: Parameters must sum to a value")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # Use session state to track current selections
                        param1_key = f"sum_{i}_param1"
                        if param1_key not in st.session_state:
                            st.session_state[param1_key] = param_names[0] if param_names else ""
                        param1 = st.selectbox("Parameter 1", options=param_names, 
                                             index=param_names.index(st.session_state[param1_key]) if st.session_state[param1_key] in param_names else 0,
                                             key=f"sum_{i}_param1_widget")
                        st.session_state[param1_key] = param1
                    
                    with col2:
                        op_key = f"sum_{i}_op"
                        if op_key not in st.session_state:
                            st.session_state[op_key] = '+'
                        operator = st.selectbox("Operator", options=['+', '-'], 
                                              index=['+', '-'].index(st.session_state[op_key]),
                                              key=f"sum_{i}_op_widget")
                        st.session_state[op_key] = operator
                    
                    with col3:
                        param2_key = f"sum_{i}_param2"
                        if param2_key not in st.session_state:
                            st.session_state[param2_key] = param_names[1] if len(param_names) > 1 else param_names[0] if param_names else ""
                        param2 = st.selectbox("Parameter 2", options=param_names, 
                                             index=param_names.index(st.session_state[param2_key]) if st.session_state[param2_key] in param_names else 0,
                                             key=f"sum_{i}_param2_widget")
                        st.session_state[param2_key] = param2
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        rel_key = f"sum_{i}_rel"
                        if rel_key not in st.session_state:
                            st.session_state[rel_key] = '<='
                        relation = st.selectbox("Relation", options=['<=', '>='],  # Ax only supports <= and >=
                                              index=['<=', '>='].index(st.session_state[rel_key]),
                                              key=f"sum_{i}_rel_widget")
                        st.session_state[rel_key] = relation
                    
                    with col5:
                        val_key = f"sum_{i}_val"
                        if val_key not in st.session_state:
                            st.session_state[val_key] = 1.0
                        value = st.number_input("Value", value=st.session_state[val_key], key=f"sum_{i}_val_widget")
                        st.session_state[val_key] = value
                    
                    # DYNAMIC UPDATE: Generate expression based on current selections
                    # Ax format: "x1 + x2 <= 1.0" or "x1 - x2 >= 0.5"
                    if st.session_state[op_key] == '+':
                        current_expression = f"{st.session_state[param1_key]} + {st.session_state[param2_key]} {st.session_state[rel_key]} {st.session_state[val_key]}"
                    else:  # '-'
                        current_expression = f"{st.session_state[param1_key]} - {st.session_state[param2_key]} {st.session_state[rel_key]} {st.session_state[val_key]}"
                    constraint['expression'] = current_expression
                    
                elif constraint['type'] == 'order':
                    st.info("Order Constraint: Parameter ordering (Ax only supports >= for order constraints)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        param1_key = f"order_{i}_param1"
                        if param1_key not in st.session_state:
                            st.session_state[param1_key] = param_names[0] if param_names else ""
                        param1 = st.selectbox("Parameter 1", options=param_names,
                                             index=param_names.index(st.session_state[param1_key]) if st.session_state[param1_key] in param_names else 0,
                                             key=f"order_{i}_param1_widget")
                        st.session_state[param1_key] = param1
                    
                    with col2:
                        # Ax only supports >= for order constraints
                        st.session_state[f"order_{i}_rel"] = '>='
                        st.write("**Relation**")
                        st.write(">= (Ax requirement)")
                    
                    with col3:
                        param2_key = f"order_{i}_param2"
                        if param2_key not in st.session_state:
                            st.session_state[param2_key] = param_names[1] if len(param_names) > 1 else param_names[0] if param_names else ""
                        param2 = st.selectbox("Parameter 2", options=param_names,
                                             index=param_names.index(st.session_state[param2_key]) if st.session_state[param2_key] in param_names else 0,
                                             key=f"order_{i}_param2_widget")
                        st.session_state[param2_key] = param2
                    
                    # DYNAMIC UPDATE: Generate expression based on current selections
                    # Ax format for order: "x1 >= x2"
                    current_expression = f"{st.session_state[param1_key]} >= {st.session_state[param2_key]}"
                    constraint['expression'] = current_expression
                    
                elif constraint['type'] == 'linear':
                    st.info("Linear Constraint: Linear combination of parameters")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        coeff1_key = f"linear_{i}_coeff1"
                        if coeff1_key not in st.session_state:
                            st.session_state[coeff1_key] = 1.0
                        coeff1 = st.number_input("Coefficient 1", value=st.session_state[coeff1_key],
                                               key=f"linear_{i}_coeff1_widget")
                        st.session_state[coeff1_key] = coeff1
                    
                    with col2:
                        param1_key = f"linear_{i}_param1"
                        if param1_key not in st.session_state:
                            st.session_state[param1_key] = param_names[0] if param_names else ""
                        param1 = st.selectbox("Parameter 1", options=param_names,
                                             index=param_names.index(st.session_state[param1_key]) if st.session_state[param1_key] in param_names else 0,
                                             key=f"linear_{i}_param1_widget")
                        st.session_state[param1_key] = param1
                    
                    with col3:
                        op_key = f"linear_{i}_op"
                        if op_key not in st.session_state:
                            st.session_state[op_key] = '+'
                        operator = st.selectbox("Operator", options=['+', '-'],
                                              index=['+', '-'].index(st.session_state[op_key]),
                                              key=f"linear_{i}_op_widget")
                        st.session_state[op_key] = operator
                    
                    with col4:
                        coeff2_key = f"linear_{i}_coeff2"
                        if coeff2_key not in st.session_state:
                            st.session_state[coeff2_key] = 1.0
                        coeff2 = st.number_input("Coefficient 2", value=st.session_state[coeff2_key],
                                               key=f"linear_{i}_coeff2_widget")
                        st.session_state[coeff2_key] = coeff2
                    
                    col5, col6, col7 = st.columns(3)
                    with col5:
                        param2_key = f"linear_{i}_param2"
                        if param2_key not in st.session_state:
                            st.session_state[param2_key] = param_names[1] if len(param_names) > 1 else param_names[0] if param_names else ""
                        param2 = st.selectbox("Parameter 2", options=param_names,
                                             index=param_names.index(st.session_state[param2_key]) if st.session_state[param2_key] in param_names else 0,
                                             key=f"linear_{i}_param2_widget")
                        st.session_state[param2_key] = param2
                    
                    with col6:
                        rel_key = f"linear_{i}_rel"
                        if rel_key not in st.session_state:
                            st.session_state[rel_key] = '<='
                        relation = st.selectbox("Relation", options=['<=', '>='],  # Ax only supports <= and >=
                                              index=['<=', '>='].index(st.session_state[rel_key]),
                                              key=f"linear_{i}_rel_widget")
                        st.session_state[rel_key] = relation
                    
                    with col7:
                        val_key = f"linear_{i}_val"
                        if val_key not in st.session_state:
                            st.session_state[val_key] = 1.0
                        value = st.number_input("Value", value=st.session_state[val_key], key=f"linear_{i}_val_widget")
                        st.session_state[val_key] = value
                    
                    # DYNAMIC UPDATE: Generate expression based on current selections
                    # Ax format: "2*x1 + 3*x2 <= 1.0" or "2*x1 - 3*x2 >= 0.5"
                    if st.session_state[op_key] == '+':
                        current_expression = f"{st.session_state[coeff1_key]}*{st.session_state[param1_key]} + {st.session_state[coeff2_key]}*{st.session_state[param2_key]} {st.session_state[rel_key]} {st.session_state[val_key]}"
                    else:  # '-'
                        current_expression = f"{st.session_state[coeff1_key]}*{st.session_state[param1_key]} - {st.session_state[coeff2_key]}*{st.session_state[param2_key]} {st.session_state[rel_key]} {st.session_state[val_key]}"
                    constraint['expression'] = current_expression
                    
                elif constraint['type'] == 'composition':
                    st.info("Composition Constraint: Parameters sum to a fixed total")
                    
                    params_key = f"composition_{i}_params"
                    if params_key not in st.session_state:
                        st.session_state[params_key] = param_names[:min(2, len(param_names))]
                    total_params = st.multiselect("Parameters in composition",
                                                options=param_names,
                                                default=st.session_state[params_key],
                                                key=f"composition_{i}_params_widget")
                    st.session_state[params_key] = total_params
                    
                    total_key = f"composition_{i}_total"
                    if total_key not in st.session_state:
                        st.session_state[total_key] = 1.0
                    total_value = st.number_input("Total Value", value=st.session_state[total_key],
                                                key=f"composition_{i}_total_widget")
                    st.session_state[total_key] = total_value
                    
                    # DYNAMIC UPDATE: Generate expression based on current selections
                    # For composition, we need to use two constraints: sum <= total and sum >= total
                    if len(st.session_state[params_key]) >= 2:
                        param_sum = " + ".join(st.session_state[params_key])
                        # Ax doesn't support ==, so we use two constraints: <= total and >= total
                        current_expression = f"{param_sum} <= {st.session_state[total_key]}"
                        constraint['expression'] = current_expression
                        
                        # Show info about composition constraint
                        st.info(f"â„¹ï¸ Composition constraint will be enforced as: {param_sum} <= {st.session_state[total_key]} AND {param_sum} >= {st.session_state[total_key]}")
                    else:
                        st.warning("Select at least 2 parameters for composition constraint")
                        constraint['expression'] = ""
                
                # Show the generated expression - NOW IT WILL UPDATE DYNAMICALLY
                st.text_input("Constraint Expression", 
                             value=constraint['expression'],
                             key=f"constraint_expr_{i}", 
                             disabled=True)
                
                # Show Ax compatibility info
                if constraint['expression']:
                    st.caption(":material/check_circle: This constraint format is compatible with Ax")
                else:
                    st.caption(":material/warning: Configure the constraint above")
                
                constraint['description'] = st.text_area("Description (optional)",
                                                       value=constraint.get('description', ''),
                                                       key=f"constraint_desc_{i}")

    # Remove marked constraints
     for i in sorted(constraints_to_remove, reverse=True):
        if 'parameter_constraints' in st.session_state:
            st.session_state.parameter_constraints.pop(i)
    
    # Objective configuration
     st.subheader(":material/flag: Objective Configuration")
     st.markdown("Define your optimization objectives")
     if columns:
        st.caption("Tip: Select objective columns directly from your uploaded dataset.")
    
    # Add objective button
     col1, col2 = st.columns([3, 1])
     with col2:
        if st.button(":material/add_circle: Add Objective", key="add_obj"):
            default_objective_name = f'objective_{len(st.session_state.objectives) + 1}'
            if columns:
                used_objective_names = {o.get('name') for o in st.session_state.objectives}
                available_objective_columns = [c for c in columns if c not in used_objective_names]
                if available_objective_columns:
                    default_objective_name = available_objective_columns[0]
            st.session_state.objectives.append({
                'name': default_objective_name,
                'type': 'minimize',
                'weight': 1.0
            })
    
    # Display objectives
     objectives_to_remove = []
     for i, obj in enumerate(st.session_state.objectives):
        with st.expander(f"Objective: {obj['name']}", expanded=True):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                current_objective_name = obj.get('name', f'objective_{i + 1}')
                if columns:
                    objective_column_options = ["<custom>"] + columns
                    selected_objective_option = st.selectbox(
                        "Objective Column",
                        options=objective_column_options,
                        index=objective_column_options.index(current_objective_name) if current_objective_name in objective_column_options else 0,
                        key=f"obj_col_select_{i}",
                        help="Pick a dataset column or choose <custom> to type your own name.",
                    )
                    if selected_objective_option == "<custom>":
                        manual_default = current_objective_name if current_objective_name not in columns else f'objective_{i + 1}'
                        obj['name'] = st.text_input(
                            "Custom Objective Name",
                            value=manual_default,
                            key=f"obj_name_custom_{i}"
                        )
                    else:
                        obj['name'] = selected_objective_option
                else:
                    obj['name'] = st.text_input(
                        "Objective Name",
                        value=current_objective_name,
                        key=f"obj_name_{i}"
                    )
            
            with col2:
                objective_type_options = [t.value for t in ObjectiveType]
                current_obj_type = str(obj.get('type', ObjectiveType.MINIMIZE.value)).lower()
                if current_obj_type not in objective_type_options:
                    current_obj_type = ObjectiveType.MINIMIZE.value
                obj['type'] = st.selectbox(
                    "Type",
                    options=objective_type_options,
                    index=objective_type_options.index(current_obj_type),
                    key=f"obj_type_{i}"
                )
            
            with col3:
                if st.button(":material/delete:", key=f"remove_obj_{i}"):
                    objectives_to_remove.append(i)
            
            # Type-specific configuration
            obj_type = str(obj.get('type', ObjectiveType.MINIMIZE.value)).lower()
            if obj_type == ObjectiveType.TARGET_RANGE.value:
                col1, col2 = st.columns(2)
                range_defaults = obj.get('target_range', [0.0, 1.0])
                if not isinstance(range_defaults, (list, tuple)) or len(range_defaults) != 2:
                    range_defaults = [0.0, 1.0]
                with col1:
                    target_min = st.number_input(
                        "Target Min",
                        value=float(range_defaults[0]),
                        key=f"target_min_{i}",
                    )
                with col2:
                    target_max = st.number_input(
                        "Target Max",
                        value=float(range_defaults[1]),
                        key=f"target_max_{i}",
                    )
                if target_min >= target_max:
                    st.warning("`Target Min` must be smaller than `Target Max`.")
                obj['target_range'] = [target_min, target_max]
                obj.pop('target_value', None)
                obj.pop('tolerance', None)
            elif obj_type == ObjectiveType.TARGET_VALUE.value:
                obj['target_value'] = st.number_input(
                    "Target Value",
                    value=float(obj.get('target_value', 0.0)),
                    key=f"target_value_{i}",
                )
                obj['tolerance'] = st.number_input(
                    "Tolerance",
                    min_value=0.0,
                    value=float(obj.get('tolerance', 0.0)),
                    key=f"tolerance_{i}",
                )
                obj.pop('target_range', None)
            else:
                obj.pop('target_range', None)
                obj.pop('target_value', None)
                obj.pop('tolerance', None)
                st.caption("No target settings required for minimize/maximize objectives.")
            
            # Weight
            obj['weight'] = st.slider("Weight", 0.1, 2.0, value=float(obj.get('weight', 1.0)), key=f"weight_{i}")
    
    # Remove marked objectives
     for i in sorted(objectives_to_remove, reverse=True):
        st.session_state.objectives.pop(i)
    
        # Evaluator selection
     st.subheader("Evaluators")
     evaluator_option = st.radio(
        "Choose how candidates are evaluated",
        options=[
            "Virtual Evaluators (AI models)",
            "Self-driving labs",
            "Third-party simulators (coming soon)"
        ],
        index=[
            "Virtual Evaluators (AI models)",
            "Self-driving labs",
            "Third-party simulators (coming soon)"
        ].index(st.session_state.get('evaluator_category', "Virtual Evaluators (AI models)")),
        key="evaluator_category_selector"
    )
     st.session_state.evaluator_category = evaluator_option
     if evaluator_option == "Virtual Evaluators (AI models)":
        st.session_state.evaluator_type = EvaluatorType.VIRTUAL.value
        is_virtual_evaluator = True
     elif evaluator_option == "Self-driving labs":
        st.session_state.evaluator_type = EvaluatorType.SELF_DRIVING_LAB.value
        is_virtual_evaluator = False
     else:
        st.session_state.evaluator_type = EvaluatorType.THIRD_PARTY_SIMULATOR.value
        is_virtual_evaluator = False

     if evaluator_option == "Self-driving labs":
        st.info("Configure connection to your lab hardware. Model configuration is disabled in SDL mode.")
        sdl_settings = st.session_state.get('sdl_settings', SDLSettings().__dict__).copy()
        protocol_choices = [
            ("MQTT", "mqtt"),
            ("HTTP", "http"),
            ("TCP", "tcp"),
            ("Serial", "serial"),
            ("Embedded (RYB SDL in CEID)", "embedded"),
        ]
        protocol_labels = [label for label, _ in protocol_choices]
        protocol_map = {label: value for label, value in protocol_choices}
        active_protocol = str(sdl_settings.get('protocol', 'http')).lower()
        if active_protocol not in protocol_map.values():
            active_protocol = "http"
        protocol = st.selectbox(
            "Communication protocol",
            options=protocol_labels,
            index=next(i for i, (_, value) in enumerate(protocol_choices) if value == active_protocol),
            key="sdl_protocol_select"
        )
        sdl_settings['protocol'] = protocol_map[protocol]
        sdl_settings['digital_twin_control'] = st.checkbox(
            "Digital twin controls experiment execution (Start/Stop/Continue from Unity)",
            value=bool(sdl_settings.get('digital_twin_control', False)),
            key="sdl_digital_twin_control_toggle",
            help="When enabled, SDL waits for Unity Start/Continue commands before running each automatic candidate.",
        )
        sdl_settings['require_continue_each_trial'] = st.checkbox(
            "Require Continue for every automatic trial",
            value=bool(sdl_settings.get('require_continue_each_trial', True)),
            key="sdl_require_continue_toggle",
            disabled=not sdl_settings['digital_twin_control'],
            help="When enabled, SDL pauses after each automatic trial until Unity sends Continue.",
        )

        if sdl_settings['protocol'] == "mqtt":
            col1, col2 = st.columns(2)
            with col1:
                sdl_settings['mqtt_host'] = st.text_input("MQTT broker host", value=sdl_settings.get('mqtt_host', 'localhost'), key="sdl_mqtt_host")
                sdl_settings['mqtt_publish_topic'] = st.text_input("Publish topic", value=sdl_settings.get('mqtt_publish_topic', 'bo/commands'), key="sdl_mqtt_pub")
            with col2:
                sdl_settings['mqtt_port'] = st.number_input("Port", value=int(sdl_settings.get('mqtt_port', 1883)), step=1, key="sdl_mqtt_port")
                sdl_settings['mqtt_response_topic'] = st.text_input("Response topic", value=sdl_settings.get('mqtt_response_topic', 'bo/results'), key="sdl_mqtt_resp")
            sdl_settings['mqtt_username'] = st.text_input("MQTT username (optional)", value=sdl_settings.get('mqtt_username', ''), key="sdl_mqtt_user")
            sdl_settings['mqtt_password'] = st.text_input("MQTT password (optional)", value=sdl_settings.get('mqtt_password', ''), type="password", key="sdl_mqtt_pass")
            sdl_settings['mqtt_client_id'] = st.text_input("Client ID", value=sdl_settings.get('mqtt_client_id', 'bo-platform'), key="sdl_mqtt_client")

        elif sdl_settings['protocol'] == "http":
            sdl_settings['http_endpoint'] = st.text_input("HTTP endpoint (POST)", value=sdl_settings.get('http_endpoint', 'http://localhost:8000/bo'), key="sdl_http_endpoint")
            sdl_settings['http_method'] = st.selectbox("HTTP method", options=["POST", "GET"], index=0 if sdl_settings.get('http_method', 'POST').upper() == 'POST' else 1, key="sdl_http_method")
            headers_text = st.text_area("HTTP headers (JSON)", value=json.dumps(sdl_settings.get('http_headers', {}), indent=2), key="sdl_http_headers")
            try:
                sdl_settings['http_headers'] = json.loads(headers_text) if headers_text.strip() else {}
            except Exception:
                st.warning("Invalid headers JSON; using defaults.")
                sdl_settings['http_headers'] = {}

        elif sdl_settings['protocol'] == "tcp":
            col1, col2 = st.columns(2)
            with col1:
                sdl_settings['tcp_host'] = st.text_input("TCP host", value=sdl_settings.get('tcp_host', 'localhost'), key="sdl_tcp_host")
            with col2:
                sdl_settings['tcp_port'] = st.number_input("TCP port", value=int(sdl_settings.get('tcp_port', 7000)), step=1, key="sdl_tcp_port")

        elif sdl_settings['protocol'] == "serial":
            col1, col2 = st.columns(2)
            with col1:
                sdl_settings['serial_port'] = st.text_input("Serial port", value=sdl_settings.get('serial_port', 'COM3'), key="sdl_serial_port")
            with col2:
                sdl_settings['serial_baud'] = st.number_input("Baud rate", value=int(sdl_settings.get('serial_baud', 115200)), step=100, key="sdl_serial_baud")
            sdl_settings['serial_timeout'] = st.number_input("Serial timeout (s)", value=float(sdl_settings.get('serial_timeout', 2.0)), min_value=0.5, step=0.5, key="sdl_serial_timeout")
        else:
            st.caption("Embedded mode starts RYB_SDL inside CEID; no separate adapter process is required.")
            embedded_col1, embedded_col2 = st.columns(2)
            with embedded_col1:
                sdl_settings['embedded_control_mode'] = st.selectbox(
                    "Initial SDL mode",
                    options=["sdl", "manual"],
                    index=0 if str(sdl_settings.get('embedded_control_mode', 'sdl')).lower() == 'sdl' else 1,
                    key="sdl_embedded_control_mode",
                )
                sdl_settings['embedded_arduino_port'] = st.text_input(
                    "Arduino port",
                    value=sdl_settings.get('embedded_arduino_port', 'COM7'),
                    key="sdl_embedded_arduino_port",
                )
                sdl_settings['embedded_arduino_baud'] = st.number_input(
                    "Arduino baud",
                    value=int(sdl_settings.get('embedded_arduino_baud', 9600)),
                    step=100,
                    key="sdl_embedded_arduino_baud",
                )
                sdl_settings['embedded_manual_aspiration_volume_ml'] = st.number_input(
                    "Manual aspiration volume (mL)",
                    value=float(sdl_settings.get('embedded_manual_aspiration_volume_ml', 1.0)),
                    min_value=0.01,
                    step=0.01,
                    key="sdl_embedded_manual_aspiration_volume_ml",
                )
            with embedded_col2:
                sdl_settings['embedded_unity_enable'] = st.checkbox(
                    "Enable Unity transport",
                    value=bool(sdl_settings.get('embedded_unity_enable', True)),
                    key="sdl_embedded_unity_enable",
                )
                sdl_settings['embedded_unity_transport'] = st.selectbox(
                    "Unity transport",
                    options=["livekit", "tcp", "none"],
                    index=["livekit", "tcp", "none"].index(str(sdl_settings.get('embedded_unity_transport', 'livekit')).lower())
                    if str(sdl_settings.get('embedded_unity_transport', 'livekit')).lower() in ["livekit", "tcp", "none"]
                    else 0,
                    key="sdl_embedded_unity_transport",
                )
                sdl_settings['embedded_sensor_timeout'] = st.number_input(
                    "Sensor timeout (s)",
                    value=float(sdl_settings.get('embedded_sensor_timeout', 30.0)),
                    min_value=0.5,
                    step=0.5,
                    key="sdl_embedded_sensor_timeout",
                )
                sdl_settings['embedded_manual_timeout'] = st.number_input(
                    "Manual mode timeout (s)",
                    value=float(sdl_settings.get('embedded_manual_timeout', 1800.0)),
                    min_value=1.0,
                    step=1.0,
                    key="sdl_embedded_manual_timeout",
                )

            unity_transport = str(sdl_settings.get('embedded_unity_transport', 'livekit')).lower()
            if unity_transport == "livekit":
                lk_col1, lk_col2 = st.columns(2)
                with lk_col1:
                    sdl_settings['embedded_livekit_url'] = st.text_input(
                        "LiveKit URL",
                        value=sdl_settings.get('embedded_livekit_url', 'wss://digital-twin-e1hn80jk.livekit.cloud'),
                        key="sdl_embedded_livekit_url",
                    )
                    sdl_settings['embedded_livekit_room'] = st.text_input(
                        "LiveKit room",
                        value=sdl_settings.get('embedded_livekit_room', 'dt'),
                        key="sdl_embedded_livekit_room",
                    )
                    sdl_settings['embedded_livekit_topic'] = st.text_input(
                        "LiveKit topic",
                        value=sdl_settings.get('embedded_livekit_topic', 'twin'),
                        key="sdl_embedded_livekit_topic",
                    )
                with lk_col2:
                    sdl_settings['embedded_sdl_livekit_token'] = st.text_input(
                        "SDL LiveKit token",
                        value=sdl_settings.get('embedded_sdl_livekit_token', ''),
                        type="password",
                        key="sdl_embedded_livekit_token",
                        help="If empty, CEID will use SDL_LIVEKIT_TOKEN from environment.",
                    )
                    sdl_settings['embedded_unity_dest_identity'] = st.text_input(
                        "Unity destination identity",
                        value=sdl_settings.get('embedded_unity_dest_identity', 'unity'),
                        key="sdl_embedded_unity_dest_identity",
                    )
            elif unity_transport == "tcp":
                tcp_col1, tcp_col2 = st.columns(2)
                with tcp_col1:
                    sdl_settings['embedded_unity_host'] = st.text_input(
                        "Unity TCP host",
                        value=sdl_settings.get('embedded_unity_host', '0.0.0.0'),
                        key="sdl_embedded_unity_host",
                    )
                with tcp_col2:
                    sdl_settings['embedded_unity_port'] = st.number_input(
                        "Unity TCP port",
                        value=int(sdl_settings.get('embedded_unity_port', 7100)),
                        step=1,
                        key="sdl_embedded_unity_port",
                    )

        # Clamp legacy/low timeouts to a sensible default (5s) but allow user to lower if needed
        current_timeout = float(sdl_settings.get('response_timeout', 20.0))
        if current_timeout < 0.5:
            current_timeout = 5.0
        sdl_settings['response_timeout'] = st.number_input(
            "Response timeout (s)", 
            value=max(current_timeout, 0.5),
            min_value=0.5,
            step=0.5, 
            key="sdl_response_timeout"
        )
        st.session_state.sdl_settings = sdl_settings

        if st.button("Connect & test SDL", use_container_width=True, key="sdl_connect_test"):
            try:
                connector = SDLConnector(SDLSettings(**sdl_settings))
                ok, msg = connector.test_connection()
                st.session_state.sdl_connector_ready = ok
                st.session_state.sdl_connector = connector if ok else None
                if ok:
                    st.success(f"SDL connection ready: {msg}")
                else:
                    st.error(f"SDL connection failed: {msg}")
            except Exception as e:
                st.error(f"SDL connection error: {e}")
                st.session_state.sdl_connector_ready = False
                st.session_state.sdl_connector = None

        if st.session_state.get('sdl_connector_ready'):
            st.success("Self-driving lab connection is ready. Candidates will be sent automatically during optimization.")
        else:
            st.warning("SDL connection not ready yet. Configure and test to enable run.")

     if evaluator_option == "Third-party simulators (coming soon)":
        st.info("Third-party simulators are still in preparation. Use Virtual evaluators or SDL for now.")

     # Defaults consumed later in optimization settings
     has_custom_models = any(m.get('type') == 'custom_model' and m.get('enabled', True) for m in st.session_state.models)
     enable_hyperparameter_tuning = False
     n_tuning_trials = 0

     if not is_virtual_evaluator:
        st.info("Model configuration is disabled for SDL/Simulator evaluators.")
     else:
        st.subheader("Model Configuration")
        st.markdown("Select and configure ML models for optimization")

        available_models = self.model_registry.get_available_models()
        supported_formats = self.model_factory.get_supported_formats()
        available_model_set = set(available_models)

        # Remove unsupported built-in model types from stale sessions/configs.
        before_models = list(st.session_state.models)
        st.session_state.models = [
            m for m in st.session_state.models
            if m.get("type") == "custom_model" or m.get("type") in available_model_set
        ]
        removed_models = [
            m.get("name", m.get("type", "unknown"))
            for m in before_models
            if m.get("type") != "custom_model" and m.get("type") not in available_model_set
        ]
        if removed_models:
            st.warning(
                f"Removed unavailable model types from configuration: {', '.join(removed_models)}."
            )

        # Keep at least one model available.
        if not st.session_state.models:
            st.session_state.models = self._default_builtin_models()

        # Toggle built-in models
        base_models = [m for m in st.session_state.models if m.get('type') != 'custom_model']
        updated_models = []
        for i, model in enumerate(base_models):
            col1, col2 = st.columns([3, 1])
            with col1:
                model['enabled'] = st.checkbox(model['name'], value=model.get('enabled', True), key=f"model_enabled_{i}")
            with col2:
                st.caption(f"type: {model.get('type')}")
            updated_models.append(model)
        # keep custom models untouched for now
        custom_models = [m for m in st.session_state.models if m.get('type') == 'custom_model']
        st.session_state.models = updated_models + custom_models

        # Custom Model Upload (simplified, keeps previous logic)
        st.subheader("Custom Model Upload")
        st.markdown("Upload your own pre-trained model in various formats")
        col1, col2 = st.columns(2)
        with col1:
            model_format = st.selectbox(
                "Model Format",
                options=supported_formats,
                index=0,
                help="Select the format of your custom model",
                key="custom_model_format_selector"
            )
        with col2:
            format_info = {
                'pickle': "Standard Python pickle format",
                'skops': "Secure model serialization format",
                'onnx': "Open Neural Network Exchange format", 
                'json': "Simple linear model in JSON format",
                'joblib': "Efficient scikit-learn serialization",
                'auto': "Auto-detect from file extension"
            }
            st.info(format_info.get(model_format, "Unknown format"))
        file_extensions = {
            'pickle': ['pkl', 'pickle'],
            'skops': ['skops'],
            'onnx': ['onnx'],
            'json': ['json'],
            'joblib': ['joblib'],
            'auto': ['pkl', 'pickle', 'skops', 'onnx', 'json', 'joblib']
        }
        uploaded_custom_model = st.file_uploader(
            f"Upload Custom Model ({', '.join(file_extensions[model_format])})", 
            type=file_extensions[model_format],
            help=f"Upload a pre-trained model in {model_format} format",
            key="custom_model_file_uploader"
        )
        if uploaded_custom_model is not None:
            current_upload_id = f"{uploaded_custom_model.name}_{uploaded_custom_model.size}_{uploaded_custom_model.type}"
            if current_upload_id not in st.session_state.processed_uploads:
                try:
                    file_extension = file_extensions[model_format][0] if model_format != 'auto' else 'pkl'
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                        tmp_file.write(uploaded_custom_model.getvalue())
                        model_path = tmp_file.name
                    st.session_state.custom_model_counter += 1
                    model_name = f"Custom_{model_format}_{st.session_state.custom_model_counter}"
                    st.session_state.custom_models[model_name] = {
                        'path': model_path,
                        'format': model_format,
                        'original_name': uploaded_custom_model.name,
                        'upload_id': current_upload_id,
                        'timestamp': time.time()
                    }
                    st.session_state.processed_uploads.add(current_upload_id)
                    st.session_state.models.append({
                        'name': model_name,
                        'type': 'custom_model',
                        'enabled': True,
                        'hyperparameters': {
                            'model_path': model_path,
                            'model_name': model_name,
                            'model_format': model_format
                        }
                    })
                    st.success(f"Custom {model_format} model uploaded: {uploaded_custom_model.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error uploading custom model: {e}")
                    if current_upload_id in st.session_state.processed_uploads:
                        st.session_state.processed_uploads.remove(current_upload_id)
            else:
                st.info("This model has already been uploaded and processed.")

        # Show uploaded custom models
        if st.session_state.custom_models:
            st.subheader("Uploaded Custom Models")
            for model_name, model_info in list(st.session_state.custom_models.items()):
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"**{model_name}**")
                    st.write(f"Format: {model_info['format']}")
                    st.write(f"File: {model_info['original_name']}")
                with col2:
                    if any(m['name'] == model_name and m['type'] == 'custom_model' for m in st.session_state.models):
                        st.success("Added to configuration")
                    else:
                        st.warning("Not in configuration")
                with col3:
                    remove_key = f"remove_custom_{model_name}"
                    if st.button("Remove", key=remove_key):
                        try:
                            if model_name in st.session_state.custom_models:
                                if os.path.exists(model_info['path']):
                                    os.unlink(model_info['path'])
                                if 'upload_id' in model_info and model_info['upload_id'] in st.session_state.processed_uploads:
                                    st.session_state.processed_uploads.remove(model_info['upload_id'])
                                del st.session_state.custom_models[model_name]
                            st.session_state.models = [m for m in st.session_state.models if m['name'] != model_name]
                            st.success(f"Removed custom model: {model_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing model: {e}")
            if st.button("Clear All Custom Models", type="secondary", key="clear_all_custom_models"):
                try:
                    for model_name, model_info in st.session_state.custom_models.items():
                        if os.path.exists(model_info['path']):
                            try:
                                os.unlink(model_info['path'])
                            except:
                                pass
                    st.session_state.custom_models = {}
                    st.session_state.processed_uploads = set()
                    st.session_state.custom_model_counter = 0
                    st.session_state.models = [m for m in st.session_state.models if m['type'] != 'custom_model']
                    st.success("All custom models cleared.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing models: {e}")

        has_custom_models = any(m.get('type') == 'custom_model' and m.get('enabled', True) for m in st.session_state.models)

        # Hyperparameter tuning controls
        col1, col2 = st.columns(2)
        tuning_disabled = has_custom_models or (not OPTUNA_AVAILABLE)
        with col1:
            enable_hyperparameter_tuning = st.checkbox(
                "Enable hyperparameter tuning",
                value=True,
                help="Tune model hyperparameters using Optuna",
                disabled=tuning_disabled,
                key="enable_tuning_checkbox"
            )
        with col2:
            tuning_strategy = st.selectbox(
                "Tuning Strategy",
                options=["Fast", "Balanced", "Comprehensive"],
                index=1,
                help="More trials = longer but potentially better",
                disabled=not enable_hyperparameter_tuning or tuning_disabled,
                key="tuning_strategy_select"
            )
        strategy_trials = {"Fast": 10, "Balanced": 20, "Comprehensive": 50}
        if enable_hyperparameter_tuning and not tuning_disabled:
            n_tuning_trials = strategy_trials[tuning_strategy]
            st.info(f"{tuning_strategy} tuning: {n_tuning_trials} trials per model")
        elif has_custom_models:
            st.info("Custom models detected: hyperparameter tuning disabled.")
        elif not OPTUNA_AVAILABLE:
            st.info("Optuna is not installed in this environment, so tuning is disabled.")
# Optimization settings - UPDATED WITH FULLYBAYESIAN
     st.subheader(":material/bolt: Optimization Settings")
    
     col1, col2 = st.columns(2)
     with col1:
        optimization_mode_label = st.radio(
            "Execution mode",
            options=["Batch", "Sequential"],
            index=0 if st.session_state.get('optimization_mode', OptimizationMode.BATCH.value) == OptimizationMode.BATCH.value else 1,
            key="optimization_mode_radio"
        )
        is_sequential_mode = optimization_mode_label == "Sequential"
        st.session_state.optimization_mode = OptimizationMode.SEQUENTIAL.value if is_sequential_mode else OptimizationMode.BATCH.value

        batch_iterations = st.number_input(
            "Iterations" if is_sequential_mode else "Batch Iterations", 
            min_value=1, 
            max_value=100, 
            value=10,
            help="Number of iterations to run" if is_sequential_mode else "Number of batch iterations to run",
            key="batch_iterations_input"
        )
        if is_sequential_mode:
            batch_size = 1
            st.caption("Batch size fixed to 1 in sequential mode.")
        else:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=20, value=3, key="batch_size_input")
        
        # Keep initialization-only strategy choices out of the main BO selector.
        hidden_main_strategies = {
            BOGenerationStrategy.UNIFORM.value,
        }
        strategy_options = [
            s.value for s in BOGenerationStrategy if s.value not in hidden_main_strategies
        ]
        current_strategy = st.session_state.get("generation_strategy_select", "default")
        if current_strategy in hidden_main_strategies:
            st.session_state.generation_strategy_select = BOGenerationStrategy.DEFAULT.value
            current_strategy = BOGenerationStrategy.DEFAULT.value
            st.info(
                "UNIFORM is now managed under Initialization Strategy. "
                "Main BO/Search strategy has been reset to default."
            )
        strategy_help = {
            "default": "Auto-select robust BO strategy from Ax.",
            "GPEI": "Legacy GP-EI alias (mapped to modern modular BO in newer Ax).",
            "SAASBO": "Sparse Axis-Aligned Subspace BO; best for higher-dimensional problems.",
            "FULLYBAYESIAN": "Legacy fully Bayesian alias (mapped to SAASBO in newer Ax).",
            "BOTORCH_MODULAR": "Modern default BoTorch modular Bayesian optimization.",
            "BO_MIXED": "Bayesian optimization for mixed spaces with categorical variables.",
            "ST_MTGP": "Single-type Multi-Task GP; requires a task parameter and multi-task data.",
            "SAAS_MTGP": "SAAS prior + MTGP; for high-dimensional multi-task BO.",
            "THOMPSON": "Discrete Thompson sampling; requires historical data.",
            "EMPIRICAL_BAYES_THOMPSON": "Discrete Thompson with empirical-Bayes shrinkage.",
            "EB_ASHR": "Empirical-Bayes ASHR + feasibility-aware discrete optimization.",
            "FACTORIAL": "Enumerate full discrete/categorical combinations.",
        }
        generation_strategy = st.selectbox(
            "Main BO/Search Strategy",
            options=strategy_options,
            index=strategy_options.index(current_strategy)
            if current_strategy in strategy_options else 0,
            help="Main optimization strategy used after the initialization strategy.",
            key="generation_strategy_select"
        )
        st.caption(strategy_help.get(generation_strategy, ""))
        strategy_conditions = {
            "default": ["No hard restriction; Ax-compatible fallback is selected automatically."],
            "GPEI": ["General BO; legacy alias in newer Ax."],
            "SAASBO": ["Recommended for higher-dimensional spaces (typically >= 4 parameters)."],
            "FULLYBAYESIAN": ["Legacy alias; maps to SAASBO in newer Ax."],
            "BOTORCH_MODULAR": ["General modern BO strategy; good default."],
            "BO_MIXED": ["Requires at least one categorical parameter.", "Avoid very large enumerated categorical combinations."],
            "ST_MTGP": ["Requires task parameter.", "Task parameter must be categorical/discrete.", "Requires historical data with at least 2 task values."],
            "SAAS_MTGP": ["Requires task parameter.", "Task parameter must be categorical/discrete.", "Requires historical data with at least 2 task values."],
            "THOMPSON": ["Requires discrete/categorical-only search space.", "Requires historical data (>=5 rows)."],
            "EMPIRICAL_BAYES_THOMPSON": ["Requires discrete/categorical-only search space.", "Requires historical data (>=5 rows)."],
            "EB_ASHR": ["Requires discrete/categorical-only search space.", "Requires historical data (>=5 rows)."],
            "FACTORIAL": ["Requires discrete/categorical-only search space.", "Total combinations should stay reasonably small."],
        }
        with st.expander("Strategy Conditions", expanded=False):
            for item in strategy_conditions.get(generation_strategy, []):
                st.write(f"- {item}")

        direct_objective_types = {ObjectiveType.MINIMIZE.value, ObjectiveType.MAXIMIZE.value}
        target_objective_types = {ObjectiveType.TARGET_RANGE.value, ObjectiveType.TARGET_VALUE.value}
        direct_count = sum(
            1 for obj in st.session_state.objectives
            if obj.get("type") in direct_objective_types
        )
        target_count = sum(
            1 for obj in st.session_state.objectives
            if obj.get("type") in target_objective_types
        )
        is_multi_objective = direct_count > 0 and (direct_count + target_count) > 1

        customizable_strategy_values = {s.value for s in ACQF_CUSTOMIZABLE_STRATEGIES}
        supports_custom_acq = generation_strategy in customizable_strategy_values
        problem_acqfs = MULTI_OBJECTIVE_ACQFS if is_multi_objective else SINGLE_OBJECTIVE_ACQFS
        acqf_options = [BOAcquisitionFunction.AUTO.value] + [a.value for a in problem_acqfs]
        acqf_help = {
            BOAcquisitionFunction.AUTO.value: "Let Ax/BoTorch choose the acquisition function automatically.",
            BOAcquisitionFunction.Q_LOG_NEI.value: "Default robust single-objective noisy EI variant.",
            BOAcquisitionFunction.Q_NEI.value: "Noisy expected improvement for single-objective BO.",
            BOAcquisitionFunction.Q_EI.value: "Expected improvement (best for low-noise settings).",
            BOAcquisitionFunction.Q_LOG_EI.value: "Log-space EI for improved numerical stability.",
            BOAcquisitionFunction.Q_KG.value: "Knowledge gradient; can be useful when sampling is expensive.",
            BOAcquisitionFunction.Q_SIMPLE_REGRET.value: "Simple-regret objective; exploratory candidate selection.",
            BOAcquisitionFunction.Q_UCB.value: "Upper confidence bound; explicit exploration/exploitation tradeoff.",
            BOAcquisitionFunction.Q_LOG_POF.value: "Probability-of-feasibility search under constraints/thresholds.",
            BOAcquisitionFunction.Q_LOG_NEHVI.value: "Default robust noisy multi-objective hypervolume improvement.",
            BOAcquisitionFunction.Q_NEHVI.value: "Noisy expected hypervolume improvement.",
            BOAcquisitionFunction.Q_EHVI.value: "Expected hypervolume improvement (best in low-noise settings).",
            BOAcquisitionFunction.Q_LOG_EHVI.value: "Log-space EHVI for numerical stability.",
            BOAcquisitionFunction.Q_LOG_NPAREGO.value: "ParEGO-style scalarization; scales better with many objectives.",
        }

        if supports_custom_acq:
            current_acqf = st.session_state.get(
                "acquisition_function_select",
                BOAcquisitionFunction.AUTO.value,
            )
            if current_acqf not in acqf_options:
                current_acqf = BOAcquisitionFunction.AUTO.value

            selected_acqf = st.selectbox(
                "Acquisition Function",
                options=acqf_options,
                index=acqf_options.index(current_acqf),
                key="acquisition_function_select",
                help="Choose the BoTorch acquisition function used by compatible strategies.",
            )
            st.caption(acqf_help.get(selected_acqf, ""))
            with st.expander("Acquisition Conditions", expanded=False):
                problem_label = "multi-objective" if is_multi_objective else "single-objective"
                st.write(f"- Current setup detected as **{problem_label}**.")
                st.write(f"- Allowed acquisitions: {', '.join(acqf_options)}")
                st.write("- `auto` is the safest option when unsure.")
        else:
            st.session_state.acquisition_function_select = BOAcquisitionFunction.AUTO.value
            st.info(
                "Selected strategy does not support custom acquisition selection. "
                "Using `auto`."
            )

        if generation_strategy in {"ST_MTGP", "SAAS_MTGP"}:
            task_candidates = [
                p.get("name")
                for p in st.session_state.parameters
                if p.get("type") in {"categorical", "discrete"} and p.get("name")
            ]
            if task_candidates:
                default_task = st.session_state.get("task_parameter_name")
                if default_task not in task_candidates:
                    default_task = task_candidates[0]
                st.session_state.task_parameter_name = st.selectbox(
                    "Task Parameter (required for MTGP)",
                    options=task_candidates,
                    index=task_candidates.index(default_task),
                    key="task_parameter_select",
                    help="Parameter that identifies task/domain for multi-task BO.",
                )
            else:
                st.session_state.task_parameter_name = None
                st.warning("MTGP strategies require at least one categorical or discrete parameter to mark as task.")
        else:
            st.session_state.task_parameter_name = None

     with col2:
        # Show calculated total iterations
        total_iterations = batch_iterations if is_sequential_mode else batch_iterations * batch_size
        st.metric("Total Iterations", total_iterations)
        
        # Show custom model warning if present
        if has_custom_models:
            st.warning(":material/warning: Custom models in use: using pre-trained models without retraining.")
        
        st.markdown("**Initialization Strategy**")
        init_strategy_options = [s.value for s in BOInitializationStrategy]
        init_strategy_help = {
            BOInitializationStrategy.NONE.value: "No dedicated initialization step before main BO strategy.",
            BOInitializationStrategy.SOBOL.value: "Quasi-random Sobol initialization (recommended default).",
            BOInitializationStrategy.UNIFORM.value: "Uniform random initialization.",
        }
        initialization_strategy = st.selectbox(
            "Initialization Strategy",
            options=init_strategy_options,
            index=init_strategy_options.index(
                st.session_state.get("initialization_strategy_select", BOInitializationStrategy.SOBOL.value)
            ) if st.session_state.get("initialization_strategy_select", BOInitializationStrategy.SOBOL.value) in init_strategy_options else 1,
            key="initialization_strategy_select",
            help="Initialization is separated from the main BO/search strategy.",
        )
        st.caption(init_strategy_help.get(initialization_strategy, ""))

        initialization_trials = st.number_input(
            "Initialization Trials",
            min_value=0,
            max_value=200,
            value=int(st.session_state.get("initialization_trials_input", 5)),
            key="initialization_trials_input",
            disabled=initialization_strategy == BOInitializationStrategy.NONE.value,
        )
        initialization_trials_effective = (
            0 if initialization_strategy == BOInitializationStrategy.NONE.value
            else int(initialization_trials)
        )

        # Backward-compatible fields for existing config serialization paths.
        use_sobol = initialization_strategy == BOInitializationStrategy.SOBOL.value
        sobol_points = int(initialization_trials_effective)
        
        random_seed = st.number_input("Random Seed", value=42, key="random_seed_input")
    # Advanced settings
     with st.expander(":material/tune: Advanced Settings"):
        is_virtual_mode = (
            st.session_state.get("evaluator_type", EvaluatorType.VIRTUAL.value)
            == EvaluatorType.VIRTUAL.value
        )
        if (not is_virtual_mode) and st.session_state.get("evolving_constraints_checkbox", False):
            # Streamlit keys must be adjusted before the widget is instantiated.
            st.session_state["evolving_constraints_checkbox"] = False
        col1, col2 = st.columns(2)
        with col1:
            use_adaptive_search = st.checkbox("Use Adaptive Search Space", value=True,
                                             help="Dynamically adjust search space based on best candidates",
                                             key="adaptive_search_checkbox")
        with col2:
            use_evolving_constraints = st.checkbox("Use Evolving Constraints", value=False,
                                                  help=(
                                                      "Gradually tighten model performance constraints during optimization."
                                                      if is_virtual_mode else
                                                      "Only available for Virtual evaluators."
                                                   ),
                                                  disabled=not is_virtual_mode,
                                                  key="evolving_constraints_checkbox")
            if not is_virtual_mode:
                use_evolving_constraints = False

        adaptive_defaults = st.session_state.get("adaptive_search_config", {})
        evolving_defaults = st.session_state.get("evolving_constraints_config", {})
        distance_defaults = self._resolve_distance_normalization_config(
            st.session_state.get("distance_normalization_config", {})
        )

        adaptive_search_config = dict(adaptive_defaults) if isinstance(adaptive_defaults, dict) else {}
        evolving_constraints_config = dict(evolving_defaults) if isinstance(evolving_defaults, dict) else {}
        distance_normalization_config = dict(distance_defaults)

        if use_adaptive_search:
            st.markdown("**Adaptive Search Space Configuration**")
            a_col1, a_col2, a_col3 = st.columns(3)
            with a_col1:
                adaptive_search_config["warmup_batches"] = int(st.number_input(
                    "Warmup Batches",
                    min_value=0,
                    max_value=200,
                    value=int(adaptive_search_config.get("warmup_batches", 2)),
                    help="Number of initial batches before adaptive shrinking starts.",
                    key="adaptive_warmup_batches_input",
                ))
                adaptive_search_config["top_fraction"] = float(st.number_input(
                    "Top Fraction",
                    min_value=0.05,
                    max_value=1.0,
                    value=float(adaptive_search_config.get("top_fraction", 0.3)),
                    step=0.05,
                    format="%.2f",
                    help="Top-performing fraction used to estimate the adaptive region.",
                    key="adaptive_top_fraction_input",
                ))
            with a_col2:
                adaptive_search_config["update_frequency"] = int(st.number_input(
                    "Update Frequency",
                    min_value=1,
                    max_value=50,
                    value=int(adaptive_search_config.get("update_frequency", 1)),
                    help="How often (in batches) adaptive bounds are updated.",
                    key="adaptive_update_frequency_input",
                ))
                adaptive_search_config["min_candidates"] = int(st.number_input(
                    "Minimum Candidates",
                    min_value=1,
                    max_value=500,
                    value=int(adaptive_search_config.get("min_candidates", 5)),
                    help="Minimum evaluated candidates required before shrinking.",
                    key="adaptive_min_candidates_input",
                ))
            with a_col3:
                adaptive_search_config["margin_fraction"] = float(st.number_input(
                    "Margin Fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(adaptive_search_config.get("margin_fraction", 0.2)),
                    step=0.05,
                    format="%.2f",
                    help="Extra margin around the top-candidate region.",
                    key="adaptive_margin_fraction_input",
                ))
                adaptive_search_config["min_relative_span"] = float(st.number_input(
                    "Min Relative Span",
                    min_value=0.01,
                    max_value=1.0,
                    value=float(adaptive_search_config.get("min_relative_span", 0.15)),
                    step=0.01,
                    format="%.2f",
                    help="Minimum width of adaptive bounds as fraction of original bounds.",
                    key="adaptive_min_relative_span_input",
                ))
            adaptive_search_config["include_experimental"] = bool(st.checkbox(
                "Use uploaded experimental points in adaptive-space estimation",
                value=bool(adaptive_search_config.get("include_experimental", True)),
                key="adaptive_include_experimental_checkbox",
            ))

        st.markdown("**Surrogate Selection Thresholds (R2 / NRMSE)**")
        if is_virtual_mode:
            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                evolving_constraints_config["static_r2_lower"] = float(st.number_input(
                    "Static R2 Lower",
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(evolving_constraints_config.get("static_r2_lower", 0.2)),
                    step=0.01,
                    format="%.3f",
                    help="Used when evolving constraints are disabled.",
                    key="static_r2_lower_input",
                ))
                evolving_constraints_config["static_r2_upper"] = float(st.number_input(
                    "Static R2 Upper",
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(evolving_constraints_config.get("static_r2_upper", 0.965)),
                    step=0.01,
                    format="%.3f",
                    help="Used when evolving constraints are disabled.",
                    key="static_r2_upper_input",
                ))
            with e_col2:
                evolving_constraints_config["static_nrmse_threshold"] = float(st.number_input(
                    "Static NRMSE Threshold",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(evolving_constraints_config.get("static_nrmse_threshold", 0.5)),
                    step=0.01,
                    format="%.3f",
                    help="Used when evolving constraints are disabled.",
                    key="static_nrmse_threshold_input",
                ))
                evolving_constraints_config["constraint_bonus"] = float(st.number_input(
                    "Constraint Bonus",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(evolving_constraints_config.get("constraint_bonus", 0.5)),
                    step=0.05,
                    format="%.3f",
                    help="Bonus added to model quality score when thresholds are met.",
                    key="constraint_bonus_input",
                ))
            with e_col3:
                evolving_constraints_config["schedule_type"] = st.selectbox(
                    "Schedule Type",
                    options=["linear", "power"],
                    index=["linear", "power"].index(
                        str(evolving_constraints_config.get("schedule_type", "linear")).lower()
                    ) if str(evolving_constraints_config.get("schedule_type", "linear")).lower() in ["linear", "power"] else 0,
                    help="How thresholds evolve over optimization batches.",
                    key="evolving_schedule_type_select",
                )
                evolving_constraints_config["progress_power"] = float(st.number_input(
                    "Progress Power",
                    min_value=0.1,
                    max_value=10.0,
                    value=float(evolving_constraints_config.get("progress_power", 1.0)),
                    step=0.1,
                    format="%.2f",
                    help="Only used for power schedule: higher values delay tightening.",
                    key="evolving_progress_power_input",
                ))

            if use_evolving_constraints:
                st.info(":material/monitoring: Thresholds evolve from configured start values to end values during optimization.")
                es_col1, es_col2, es_col3 = st.columns(3)
                with es_col1:
                    evolving_constraints_config["r2_lower_start"] = float(st.number_input(
                        "R2 Lower Start",
                        min_value=-1.0,
                        max_value=1.0,
                        value=float(evolving_constraints_config.get("r2_lower_start", 0.2)),
                        step=0.01,
                        format="%.3f",
                        key="evolving_r2_lower_start_input",
                    ))
                    evolving_constraints_config["r2_lower_end"] = float(st.number_input(
                        "R2 Lower End",
                        min_value=-1.0,
                        max_value=1.0,
                        value=float(evolving_constraints_config.get("r2_lower_end", 0.7)),
                        step=0.01,
                        format="%.3f",
                        key="evolving_r2_lower_end_input",
                    ))
                with es_col2:
                    evolving_constraints_config["r2_upper_start"] = float(st.number_input(
                        "R2 Upper Start",
                        min_value=-1.0,
                        max_value=1.0,
                        value=float(evolving_constraints_config.get("r2_upper_start", 0.965)),
                        step=0.01,
                        format="%.3f",
                        key="evolving_r2_upper_start_input",
                    ))
                    evolving_constraints_config["r2_upper_end"] = float(st.number_input(
                        "R2 Upper End",
                        min_value=-1.0,
                        max_value=1.0,
                        value=float(evolving_constraints_config.get("r2_upper_end", 0.965)),
                        step=0.01,
                        format="%.3f",
                        key="evolving_r2_upper_end_input",
                    ))
                with es_col3:
                    evolving_constraints_config["nrmse_start"] = float(st.number_input(
                        "NRMSE Start",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(evolving_constraints_config.get("nrmse_start", 0.5)),
                        step=0.01,
                        format="%.3f",
                        key="evolving_nrmse_start_input",
                    ))
                    evolving_constraints_config["nrmse_end"] = float(st.number_input(
                        "NRMSE End",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(evolving_constraints_config.get("nrmse_end", 0.1)),
                        step=0.01,
                        format="%.3f",
                        key="evolving_nrmse_end_input",
                    ))
        else:
            st.info("Surrogate-selection thresholds are only used with Virtual evaluators.")

        st.markdown("**Observation Uncertainty (SEM/STD) Configuration**")
        uncertainty_defaults = self._resolve_uncertainty_config(
            st.session_state.get("uncertainty_config", {})
        )
        uncertainty_config = dict(uncertainty_defaults)

        u_col1, u_col2, u_col3 = st.columns(3)
        with u_col1:
            uncertainty_config["enabled"] = bool(st.checkbox(
                "Enable Uncertainty Feedback",
                value=bool(uncertainty_config.get("enabled", True)),
                key="uncertainty_enabled_checkbox",
                help="Use uncertainty values (SEM) in Ax trial completion data.",
            ))
            uncertainty_config["std_mode"] = st.selectbox(
                "STD Interpretation",
                options=["as_sem", "std_to_sem"],
                index=["as_sem", "std_to_sem"].index(
                    str(uncertainty_config.get("std_mode", "as_sem")).lower()
                ) if str(uncertainty_config.get("std_mode", "as_sem")).lower() in ["as_sem", "std_to_sem"] else 0,
                key="uncertainty_std_mode_select",
                help="`as_sem`: treat STD as SEM. `std_to_sem`: divide STD by sqrt(replicates).",
            )
            uncertainty_config["default_replicates"] = int(st.number_input(
                "Default Replicates",
                min_value=1,
                max_value=1000,
                value=int(uncertainty_config.get("default_replicates", 1)),
                key="uncertainty_default_replicates_input",
            ))
        with u_col2:
            uncertainty_config["fallback_sem"] = float(st.number_input(
                "Fallback SEM",
                min_value=0.0,
                max_value=1000.0,
                value=float(uncertainty_config.get("fallback_sem", 0.0)),
                step=0.001,
                format="%.6f",
                key="uncertainty_fallback_sem_input",
            ))
            uncertainty_config["min_sem"] = float(st.number_input(
                "Minimum SEM",
                min_value=0.0,
                max_value=1000.0,
                value=float(uncertainty_config.get("min_sem", 0.0)),
                step=0.001,
                format="%.6f",
                key="uncertainty_min_sem_input",
            ))
            uncertainty_config["virtual_sem_scale"] = float(st.number_input(
                "Virtual SEM Scale",
                min_value=0.001,
                max_value=1000.0,
                value=float(uncertainty_config.get("virtual_sem_scale", 1.0)),
                step=0.1,
                format="%.3f",
                key="uncertainty_virtual_scale_input",
                help="Scale factor applied to virtual evaluator uncertainty before sending to Ax.",
            ))
        with u_col3:
            available_columns = []
            if st.session_state.get("uploaded_data") is not None:
                available_columns = st.session_state.uploaded_data.columns.tolist()
            replicates_column_options = ["(none)"] + available_columns
            current_repl_col = str(uncertainty_config.get("replicates_column", "") or "")
            repl_index = (
                replicates_column_options.index(current_repl_col)
                if current_repl_col in replicates_column_options
                else 0
            )
            selected_repl_col = st.selectbox(
                "Replicates Column",
                options=replicates_column_options,
                index=repl_index,
                key="uncertainty_replicates_column_select",
                help="Optional column with per-row replicate counts for STD to SEM conversion.",
            )
            uncertainty_config["replicates_column"] = "" if selected_repl_col == "(none)" else selected_repl_col

        uncertainty_config["data_sem_suffixes"] = self._split_csv_items(
            st.text_input(
                "Data SEM Suffixes (comma-separated)",
                value=",".join(uncertainty_config.get("data_sem_suffixes", ["_sem", "_stderr", "_se", "_uncertainty"])),
                key="uncertainty_sem_suffixes_input",
            ),
            ["_sem", "_stderr", "_se"],
        )
        uncertainty_config["data_std_suffixes"] = self._split_csv_items(
            st.text_input(
                "Data STD Suffixes (comma-separated)",
                value=",".join(uncertainty_config.get("data_std_suffixes", ["_std", "_stdev", "_sigma"])),
                key="uncertainty_std_suffixes_input",
            ),
            ["_std", "_stdev", "_sigma"],
        )
        uncertainty_config["sdl_sem_keys"] = self._split_csv_items(
            st.text_input(
                "SDL SEM Keys (comma-separated)",
                value=",".join(uncertainty_config.get("sdl_sem_keys", ["sem", "stderr", "se", "uncertainty"])),
                key="uncertainty_sdl_sem_keys_input",
            ),
            ["sem", "stderr", "se", "uncertainty"],
        )
        uncertainty_config["sdl_std_keys"] = self._split_csv_items(
            st.text_input(
                "SDL STD Keys (comma-separated)",
                value=",".join(uncertainty_config.get("sdl_std_keys", ["std", "stdev", "sigma"])),
                key="uncertainty_sdl_std_keys_input",
            ),
            ["std", "stdev", "sigma"],
        )

        if target_count > 0:
            st.markdown("**Distance Normalization Configuration**")
            dn_col1, dn_col2, dn_col3 = st.columns(3)
            with dn_col1:
                distance_normalization_config["enabled"] = bool(st.checkbox(
                    "Enable Distance Normalization",
                    value=bool(distance_normalization_config.get("enabled", True)),
                    key="distance_norm_enabled_checkbox",
                    help="Normalize objective components before weighted Euclidean aggregation.",
                ))
                distance_method_options = ["quantile", "range", "std"]
                current_distance_method = str(distance_normalization_config.get("method", "quantile")).lower()
                if current_distance_method not in distance_method_options:
                    current_distance_method = "quantile"
                distance_normalization_config["method"] = st.selectbox(
                    "Scale Method",
                    options=distance_method_options,
                    index=distance_method_options.index(current_distance_method),
                    disabled=not distance_normalization_config["enabled"],
                    key="distance_norm_method_select",
                )
                distance_normalization_config["normalize_weight_norm"] = bool(st.checkbox(
                    "Normalize by Weight Norm",
                    value=bool(distance_normalization_config.get("normalize_weight_norm", True)),
                    disabled=not distance_normalization_config["enabled"],
                    key="distance_norm_weight_checkbox",
                    help="Divide final distance by sqrt(sum(weights^2)).",
                ))
            with dn_col2:
                active_method = str(distance_normalization_config.get("method", "quantile")).lower()
                if active_method == "quantile":
                    distance_normalization_config["q_low"] = float(st.number_input(
                        "Lower Quantile",
                        min_value=0.0,
                        max_value=0.99,
                        value=float(distance_normalization_config.get("q_low", 0.05)),
                        step=0.01,
                        format="%.2f",
                        disabled=not distance_normalization_config["enabled"],
                        key="distance_norm_q_low_input",
                    ))
                    distance_normalization_config["q_high"] = float(st.number_input(
                        "Upper Quantile",
                        min_value=0.01,
                        max_value=1.0,
                        value=float(distance_normalization_config.get("q_high", 0.95)),
                        step=0.01,
                        format="%.2f",
                        disabled=not distance_normalization_config["enabled"],
                        key="distance_norm_q_high_input",
                    ))
                else:
                    st.caption("`Lower/Upper Quantile` are used only when `Scale Method = quantile`.")
                distance_normalization_config["min_scale"] = float(st.number_input(
                    "Minimum Scale",
                    min_value=1e-12,
                    max_value=1e6,
                    value=float(distance_normalization_config.get("min_scale", 1e-6)),
                    step=1e-4,
                    format="%.8f",
                    disabled=not distance_normalization_config["enabled"],
                    key="distance_norm_min_scale_input",
                ))
            with dn_col3:
                distance_normalization_config["clip_component"] = float(st.number_input(
                    "Clip Component",
                    min_value=0.0,
                    max_value=1e6,
                    value=float(distance_normalization_config.get("clip_component", 10.0)),
                    step=0.5,
                    format="%.3f",
                    disabled=not distance_normalization_config["enabled"],
                    key="distance_norm_clip_input",
                    help="0 disables clipping.",
                ))
                distance_normalization_config["max_scale_samples"] = int(st.number_input(
                    "Max Scale Samples",
                    min_value=10,
                    max_value=200000,
                    value=int(distance_normalization_config.get("max_scale_samples", 2000)),
                    step=10,
                    disabled=not distance_normalization_config["enabled"],
                    key="distance_norm_max_samples_input",
                ))
                distance_normalization_config["normalize_target_components"] = bool(st.checkbox(
                    "Normalize Target Components",
                    value=bool(distance_normalization_config.get("normalize_target_components", False)),
                    disabled=not distance_normalization_config["enabled"],
                    key="distance_norm_targets_checkbox",
                    help="Also normalize target_range / target_value components in total-distance calculation.",
                ))
                if direct_count > 0:
                    st.caption(
                        "In mixed/direct-objective mode, target-distance metrics sent to Ax "
                        "also use the selected scale method."
                    )
                elif not distance_normalization_config["normalize_target_components"]:
                    st.caption(
                        "With only target objectives, keep `Normalize Target Components` enabled "
                        "if you want `Scale Method` to change total distance behavior."
                    )
        else:
            distance_normalization_config["normalize_target_components"] = False
            st.info(
                "Distance Normalization settings are hidden because no target objectives "
                "(`target_value` / `target_range`) are configured."
            )

        distance_normalization_config = self._resolve_distance_normalization_config(
            distance_normalization_config
        )

        st.session_state.adaptive_search_config = adaptive_search_config
        st.session_state.evolving_constraints_config = evolving_constraints_config
        st.session_state.uncertainty_config = uncertainty_config
        st.session_state.distance_normalization_config = distance_normalization_config

    # Save configuration
     if st.button(":material/save: Save Experiment Configuration", use_container_width=True, type="primary", key="save_config_button"):
        try:
            if not st.session_state.parameters:
                st.error("Add at least one parameter before saving.")
                return
            if not st.session_state.objectives:
                st.error("Add at least one objective before saving.")
                return

            param_names = [p.get("name", "").strip() for p in st.session_state.parameters]
            objective_names = [o.get("name", "").strip() for o in st.session_state.objectives]
            if any(not n for n in param_names):
                st.error("All parameters must have a non-empty name.")
                return
            if any(not n for n in objective_names):
                st.error("All objectives must have a non-empty name.")
                return
            if len(set(param_names)) != len(param_names):
                st.error("Parameter names must be unique.")
                return
            if len(set(objective_names)) != len(objective_names):
                st.error("Objective names must be unique.")
                return

            # FIX: Ensure at least one model is enabled before saving
            enabled_models = [m for m in st.session_state.models if m.get('enabled', False)]
            if st.session_state.get('evaluator_type', EvaluatorType.VIRTUAL.value) == EvaluatorType.VIRTUAL.value:
                if not enabled_models:
                    if st.session_state.models:
                        st.session_state.models[0]['enabled'] = True
                        st.warning(":material/warning: No models were enabled. Enabled the first model automatically.")
                        enabled_models = [st.session_state.models[0]]
                    else:
                        st.error(":material/error: No models configured. Please add at least one model.")
                        return
            
            config = self.config_manager.create_config_from_ui({
                'experiment_name': exp_name,
                'parameters': st.session_state.parameters,
                'objectives': st.session_state.objectives,
                'models': st.session_state.models,
                'evaluator_type': st.session_state.get('evaluator_type', EvaluatorType.VIRTUAL.value),
                'optimization_mode': st.session_state.get('optimization_mode', OptimizationMode.BATCH.value),
                'sdl_settings': st.session_state.get('sdl_settings', {}),
                'task_parameter_name': st.session_state.get('task_parameter_name'),
                'parameter_constraints': st.session_state.get('parameter_constraints', []),  # NEW
                'enable_hyperparameter_tuning': enable_hyperparameter_tuning and not has_custom_models and OPTUNA_AVAILABLE,
                'n_tuning_trials': n_tuning_trials if enable_hyperparameter_tuning and not has_custom_models and OPTUNA_AVAILABLE else 0,
                'batch_iterations': batch_iterations,
                'batch_size': batch_size,
                'max_iterations': total_iterations,
                'random_seed': random_seed,
                'n_initial_points': 10,
                'generation_strategy': generation_strategy,
                'acquisition_function': st.session_state.get('acquisition_function_select', BOAcquisitionFunction.AUTO.value),
                'initialization_strategy': st.session_state.get('initialization_strategy_select', BOInitializationStrategy.SOBOL.value),
                'initialization_trials': int(initialization_trials_effective),
                'use_sobol': use_sobol,
                'sobol_points': sobol_points,
                'use_adaptive_search': use_adaptive_search,
                'adaptive_search_config': adaptive_search_config,
                'use_evolving_constraints': use_evolving_constraints,
                'evolving_constraints_config': evolving_constraints_config,
                'uncertainty_config': uncertainty_config,
                'distance_normalization_config': st.session_state.get('distance_normalization_config'),
            })
            
            st.session_state.experiment_config = config
            st.success(":material/check_circle: Experiment configuration saved.")
            
            # Show configuration summary
            st.subheader(":material/summarize: Configuration Summary")
            summary_data = {
                "Experiment Name": config.experiment_name,
                "Parameters": len(config.parameters),
                "Objectives": len(config.objectives),
                "Parameter Constraints": len(config.parameter_constraints),  # NEW
                "Evaluator": config.evaluator_type.value if hasattr(config.evaluator_type, 'value') else config.evaluator_type,
                "Mode": config.optimization_mode.value if hasattr(config.optimization_mode, 'value') else config.optimization_mode,
                "Enabled Models": len([m for m in config.models if m.enabled]),
                "Model Types": ", ".join(list(set([m.type for m in config.models if m.enabled]))),
                "Hyperparameter Tuning": "Enabled" if (enable_hyperparameter_tuning and not has_custom_models and OPTUNA_AVAILABLE) else "Disabled",
                "Tuning Trials": n_tuning_trials if (enable_hyperparameter_tuning and not has_custom_models and OPTUNA_AVAILABLE) else 0,
                "Batch Iterations": config.batch_iterations,
                "Batch Size": config.batch_size,
                "Total Iterations": config.max_iterations,
                "Generation Strategy": config.generation_strategy.value,
                "Acquisition Function": (
                    config.acquisition_function.value
                    if hasattr(config.acquisition_function, 'value')
                    else config.acquisition_function
                ),
                "Initialization Strategy": (
                    config.initialization_strategy.value
                    if hasattr(config.initialization_strategy, 'value')
                    else config.initialization_strategy
                ),
                "Initialization Trials": getattr(config, 'initialization_trials', config.sobol_points),
                "Task Parameter": config.task_parameter_name or "N/A",
                "Adaptive Search": "Enabled" if config.use_adaptive_search else "Disabled",
                "Evolving Constraints": "Enabled" if config.use_evolving_constraints else "Disabled",
                "Uncertainty Feedback": "Enabled" if (getattr(config, "uncertainty_config", {}) or {}).get("enabled", True) else "Disabled",
                "Distance Normalization": "Enabled" if (getattr(config, "distance_normalization_config", {}) or {}).get("enabled", True) else "Disabled",
            }
            
            # Display summary as metrics
            cols = st.columns(3)
            for idx, (key, value) in enumerate(summary_data.items()):
                cols[idx % 3].metric(key, value)
            
        except Exception as e:
            st.error(f":material/error: Error saving configuration: {e}")
            st.error("Please check that all parameters and objectives are properly configured.")
    def render_optimization(self, show_header: bool = True):
        """Render optimization execution section - UPDATED WITH TUNING INFO"""
        if show_header:
            st.header("Optimization")
        config = st.session_state.experiment_config
        if config is None:
            st.warning("Please configure your experiment first in the 'Experiment Setup' section.")
            return
        evaluator_type = getattr(config, 'evaluator_type', EvaluatorType.VIRTUAL)
        is_virtual_evaluator = evaluator_type == EvaluatorType.VIRTUAL

        df = st.session_state.get('uploaded_data')
        if df is None and is_virtual_evaluator:
            st.info(
                "No dataset uploaded. Virtual mode can still run using pre-trained custom models "
                "or a synthetic cold-start objective. Upload data for realistic model-based optimization."
            )

        sdl_connector = st.session_state.get('sdl_connector')
        if evaluator_type == EvaluatorType.SELF_DRIVING_LAB:
            if not st.session_state.get('sdl_connector_ready'):
                try:
                    connector = SDLConnector(SDLSettings(**st.session_state.get('sdl_settings', {})))
                    ok, msg = connector.connect()
                    st.session_state.sdl_connector_ready = ok
                    st.session_state.sdl_connector = connector if ok else None
                    sdl_connector = connector if ok else None
                    if ok:
                        st.success(f"SDL connected: {msg}")
                    else:
                        st.error(f"SDL connection failed: {msg}")
                        return
                except Exception as e:
                    st.error(f"SDL connection error: {e}")
                    return

        st.subheader("Experiment Ready")
        
        # Show batch/sequence configuration with tuning info
        tuning_status = "Enabled" if config.enable_hyperparameter_tuning else "Disabled"
        tuning_trials = config.n_tuning_trials if config.enable_hyperparameter_tuning else 0
        has_custom_models = any(m.type == 'custom_model' and m.enabled for m in config.models)
        
        st.info(f"""
        **Experiment**: {config.experiment_name}
        **Parameters**: {len(config.parameters)} | **Objectives**: {len(config.objectives)}
        **Models**: {len([m for m in config.models if m.enabled])} 
        **Hyperparameter Tuning**: {tuning_status} | **Trials**: {tuning_trials}
        **Batch Iterations**: {config.batch_iterations} | **Batch Size**: {config.batch_size}
        **Total Candidates**: {config.max_iterations}
        **Strategy**: {config.generation_strategy.value}
        **Initialization**: {(config.initialization_strategy.value if hasattr(config.initialization_strategy, 'value') else config.initialization_strategy)} ({getattr(config, 'initialization_trials', config.sobol_points)} trials)
        **Acquisition**: {config.acquisition_function.value if hasattr(config.acquisition_function, 'value') else config.acquisition_function}
        **Task Parameter**: {config.task_parameter_name if config.task_parameter_name else 'N/A'}
        **Initial Data Points**: {len(df) if df is not None else 0} experimental
        **Custom Models**: {'Yes' if has_custom_models else 'No'}
        """)
        
        if evaluator_type == EvaluatorType.SELF_DRIVING_LAB:
            st.info("Self-driving lab mode is enabled. The platform accepts objective values with optional uncertainty (SEM/STD).")
        elif not is_virtual_evaluator:
            st.warning("Third-party simulator integrations are still under development.")
        
        # Prepare data
        try:
            feature_columns = [p.name for p in config.parameters]
            target_columns = [o.name for o in config.objectives]
            uncertainty_config = self._resolve_uncertainty_config(
                getattr(config, "uncertainty_config", None) or st.session_state.get("uncertainty_config", {})
            )
            if df is not None:
                missing_features = [col for col in feature_columns if col not in df.columns]
                missing_targets = [col for col in target_columns if col not in df.columns]
                if missing_features:
                    st.error(f"Missing feature columns: {missing_features}")
                    return
                if missing_targets:
                    st.error(f"Missing target columns: {missing_targets}")
                    return
                X = df[feature_columns].values
                Y = df[target_columns].values
                Y_sem = self._extract_objective_sem_matrix(df, target_columns, uncertainty_config)
                st.success(f"? Data prepared: {X.shape[0]} samples, {X.shape[1]} features, {Y.shape[1]} objectives")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Features Shape", f"{X.shape}")
                with col2:
                    st.metric("Targets Shape", f"{Y.shape}")
                finite_sem = np.isfinite(Y_sem).sum()
                if uncertainty_config.get("enabled", True) and finite_sem > 0:
                    st.caption(f"Uncertainty detected for {finite_sem} objective observations (interpreted as SEM).")
            else:
                X = np.zeros((0, len(feature_columns)))
                Y = np.zeros((0, len(target_columns)))
                Y_sem = np.zeros((0, len(target_columns)))
                st.info("Starting without historical data. Initialization candidates and evaluator feedback will populate the experiment.")
        except Exception as e:
            st.error(f"Error preparing data: {e}")
            return

        st.session_state.optimization_data = {"X": X, "Y": Y, "Y_sem": Y_sem}

        direct_objectives_present = any(
            (hasattr(obj.type, 'value') and obj.type in [ObjectiveType.MINIMIZE, ObjectiveType.MAXIMIZE]) or
            (not hasattr(obj.type, 'value') and obj.type in ['minimize', 'maximize'])
            for obj in config.objectives
        )

        effective_total_batches = config.batch_iterations if config.optimization_mode == OptimizationMode.BATCH else max(config.max_iterations, config.batch_iterations)


        st.subheader("Optimization Controls")

        control_cols = st.columns([1.3, 1.2, 1.1, 1.1, 1])
        start_clicked = control_cols[0].button("Start Fresh", use_container_width=True, type="primary", key="start_fresh_btn", disabled=False)
        resume_clicked = control_cols[1].button("Resume / Next Batch", use_container_width=True, key="resume_btn", disabled=False)
        pause_clicked = control_cols[2].button("Pause & Save", use_container_width=True, key="pause_btn", disabled=('optimizer' not in st.session_state))
        stop_clicked = control_cols[3].button("Stop & Save", use_container_width=True, key="stop_btn", disabled=('optimizer' not in st.session_state))
        save_clicked = control_cols[4].button("Save", use_container_width=True, key="save_btn", disabled=('optimizer' not in st.session_state))

        if evaluator_type == EvaluatorType.SELF_DRIVING_LAB and not st.session_state.get('sdl_connector_ready'):
            st.error("Connect to the self-driving lab before starting optimization.")
            return

        uploaded_checkpoint = st.file_uploader("Load checkpoint (.json)", type=["json"], key="checkpoint_loader")
        checkpoint_data = None
        if uploaded_checkpoint is not None:
            try:
                checkpoint_data = self._load_checkpoint(uploaded_checkpoint)
                st.success("Checkpoint loaded. Use Resume to continue.")
            except Exception as e:
                st.error(f"Failed to load checkpoint: {e}")
        elif st.session_state.get('loaded_checkpoint'):
            checkpoint_data = st.session_state.loaded_checkpoint

        use_loaded_checkpoint = False
        if checkpoint_data:
            use_loaded_checkpoint = st.checkbox("Use loaded checkpoint", value=True, key="use_loaded_checkpoint")

        resume_state = None
        if use_loaded_checkpoint and checkpoint_data:
            try:
                config = self._config_from_dict(checkpoint_data['config'])
                X = np.array(checkpoint_data.get('X', X))
                Y = np.array(checkpoint_data.get('Y', Y))
                Y_sem = np.array(checkpoint_data.get('Y_sem', Y_sem))
                resume_state = checkpoint_data.get('optimizer_state')
                if Y_sem.shape != Y.shape:
                    Y_sem = np.full(Y.shape, np.nan, dtype=np.float64)
                direct_objectives_present = any(
                    (hasattr(obj.type, 'value') and obj.type in [ObjectiveType.MINIMIZE, ObjectiveType.MAXIMIZE]) or
                    (not hasattr(obj.type, 'value') and obj.type in ['minimize', 'maximize'])
                    for obj in config.objectives
                )
                st.info("Using configuration and data from the loaded checkpoint.")
            except Exception as e:
                st.error(f"Failed to apply checkpoint config/data: {e}")
                resume_state = None

        st.session_state.optimization_data = {"X": X, "Y": Y, "Y_sem": Y_sem}

        def _reset_progress_state(start_batch=0, best_distance=float('inf'), keep_history=False):
            total_batches = max(effective_total_batches, start_batch)
            existing = st.session_state.get('optimization_progress', {})
            st.session_state.optimization_progress = {
                'current_batch': start_batch,
                'total_batches': total_batches,
                'current_trial': None,
                'status': 'ready',
                'candidates_completed': existing.get('candidates_completed', 0) if keep_history else 0,
                'best_distance': best_distance if not keep_history else existing.get('best_distance', best_distance),
                'start_time': time.time(),
                'current_parameters': {},
                'current_uncertainties': {},
                'current_objectives': {},
                'uses_direct_objectives': direct_objectives_present,
                'distance_history': existing.get('distance_history', []) if keep_history else [],
                'batch_history': existing.get('batch_history', []) if keep_history else [],
                'uncertainty_history': existing.get('uncertainty_history', []) if keep_history else [],
                'objective_history': existing.get('objective_history', {}) if keep_history else {}
            }

        if start_clicked:
            for key in ['optimization_result', 'evaluator', 'optimizer']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.loaded_checkpoint = None
            st.session_state.checkpoint_info = None
            _reset_progress_state(0, float('inf'), keep_history=False)
            st.session_state.run_status = 'running'
            self._run_optimization(config, X, Y, Y_sem, resume_state=None, single_batch=True, sdl_connector=sdl_connector, reset_progress=True)

        if resume_clicked:
            st.session_state.run_status = 'running'
            start_from = 0
            best_distance = float('inf')
            if resume_state:
                start_from = resume_state.get('completed_batches', 0)
                best_distance = resume_state.get('best_overall_distance', float('inf'))
            elif st.session_state.get('optimizer') is not None:
                start_from = getattr(st.session_state.optimizer, 'completed_batches', 0)
                best_distance = getattr(st.session_state.optimizer, 'best_overall_distance', float('inf'))
            if resume_state is None and st.session_state.get('optimizer') is not None:
                st.session_state.optimizer.config = config
            # keep prior histories when resuming
            _reset_progress_state(start_from, best_distance, keep_history=True)
            self._run_optimization(config, X, Y, Y_sem, resume_state=resume_state, single_batch=True, sdl_connector=sdl_connector, reset_progress=False)

        if save_clicked or pause_clicked or stop_clicked:
            if 'optimizer' in st.session_state:
                ckpt_path = self._save_checkpoint(config, X, Y, Y_sem, optimizer=st.session_state.get('optimizer'), progress=st.session_state.get('optimization_progress'))
                st.success(f"Checkpoint saved to {ckpt_path}")
                try:
                    ckpt_bytes = open(ckpt_path, 'rb').read()
                    st.download_button('Download checkpoint', ckpt_bytes, file_name=Path(ckpt_path).name, use_container_width=True)
                except Exception as e:
                    st.warning(f'Could not prepare download: {e}')
                if stop_clicked:
                    st.session_state.run_status = 'stopped'
                    if 'optimization_progress' in st.session_state:
                        st.session_state.optimization_progress['status'] = 'stopped'
                elif pause_clicked:
                    st.session_state.run_status = 'paused'
                    if 'optimization_progress' in st.session_state:
                        st.session_state.optimization_progress['status'] = 'paused'
        if st.session_state.get('run_status') == "paused":
            st.info("Optimization paused. Click Resume to run the next batch.")
        if st.session_state.get('run_status') == "stopped":
            st.info("Optimization stopped. You can Resume from the saved checkpoint later.")

        # Show optimization results if available
        if hasattr(self, "optimization_result") and self.optimization_result:
            self._display_optimization_status()




    def _run_optimization(self, config, X, Y, Y_sem=None, resume_state=None, single_batch=True, sdl_connector=None, reset_progress=True):
        """Run optimization with checkpoint-aware updates (one batch at a time)."""
        try:
            progress_container = st.empty()
            direct_objectives_present = any(
                (hasattr(obj.type, 'value') and obj.type in [ObjectiveType.MINIMIZE, ObjectiveType.MAXIMIZE]) or
                (not hasattr(obj.type, 'value') and obj.type in ['minimize', 'maximize'])
                for obj in config.objectives
            )
            effective_total_batches = config.batch_iterations if config.optimization_mode == OptimizationMode.BATCH else max(config.max_iterations, config.batch_iterations)

            def _reset_progress(start_batch=0, best_distance=float('inf')):
                total_batches = max(effective_total_batches, start_batch)
                st.session_state.optimization_progress = {
                    'current_batch': start_batch,
                    'total_batches': total_batches,
                    'current_trial': None,
                    'status': 'running',
                    'candidates_completed': 0,
                    'best_distance': best_distance,
                    'start_time': time.time(),
                    'current_parameters': {},
                    'current_uncertainties': {},
                    'current_objectives': {},
                    'uses_direct_objectives': direct_objectives_present,
                    # keep history only if resetting is requested
                    'distance_history': [] if reset_progress else st.session_state.optimization_progress.get('distance_history', []),
                    'batch_history': [] if reset_progress else st.session_state.optimization_progress.get('batch_history', []),
                    'uncertainty_history': [] if reset_progress else st.session_state.optimization_progress.get('uncertainty_history', []),
                    'objective_history': {} if reset_progress else st.session_state.optimization_progress.get('objective_history', {})
                }

            start_batch = 0
            best_distance = float('inf')
            if resume_state:
                start_batch = resume_state.get('completed_batches', 0)
                best_distance = resume_state.get('best_overall_distance', float('inf'))
            _reset_progress(start_batch, best_distance)

            reuse_optimizer = st.session_state.get('optimizer') if resume_state is None else None
            if reuse_optimizer is None:
                if config.evaluator_type == EvaluatorType.VIRTUAL:
                    enabled_models = [m for m in config.models if m.enabled]
                    n_features = len(config.parameters) if (X is None or len(X) == 0) else X.shape[1]
                    self.model_factory.create_model_instances(
                        [{'name': m.name, 'type': m.type, 'hyperparameters': m.hyperparameters}
                         for m in enabled_models],
                        n_features
                    )
                    self.evaluator = GeneralizedEvaluator(
                        self.model_factory,
                        config.parameters,
                        config.objectives,
                        distance_normalization_config=getattr(config, "distance_normalization_config", None),
                    )
                    if X is None or len(X) == 0:
                        custom_models = [m for m in enabled_models if m.type == 'custom_model']
                        if custom_models:
                            # Pre-trained virtual mode: skip fitting and use custom model predictions directly.
                            self.evaluator.models = {}
                            self.evaluator.metrics = {}
                            for model_cfg in custom_models:
                                model_instance = self.model_factory.get_model(model_cfg.name)
                                if model_instance is None:
                                    continue
                                self.evaluator.metrics[model_cfg.name] = {}
                                for obj in config.objectives:
                                    model_key = f"{model_cfg.name}_{obj.name}"
                                    self.evaluator.models[model_key] = model_instance
                                    self.evaluator.metrics[model_cfg.name][obj.name] = EvaluationMetrics(
                                        train_rmse=0.0,
                                        test_rmse=0.0,
                                        train_r2=1.0,
                                        test_r2=1.0,
                                        train_mae=0.0,
                                        test_mae=0.0,
                                        normalized_train_rmse=0.0,
                                        normalized_test_rmse=0.0,
                                        quality_score=1.0,
                                        meets_constraints=True,
                                    )
                            self.evaluator.fitted = True
                            st.info("Running virtual evaluator in pre-trained custom-model mode (no uploaded data).")
                        else:
                            # Cold-start virtual mode (no data, no pre-trained models): deterministic synthetic evaluator.
                            # This keeps workflow operational while user prepares data/models.
                            def _cold_start_predict(X_input, model_name=None, objective_name=None):
                                X_arr = np.asarray(X_input, dtype=np.float64)
                                if X_arr.ndim == 1:
                                    X_arr = X_arr.reshape(1, -1)
                                if X_arr.size == 0:
                                    return np.zeros((0, len(config.objectives)))

                                base_signal = np.sum(X_arr, axis=1)
                                outputs = []
                                for j, _obj in enumerate(config.objectives):
                                    scale = float(j + 1)
                                    objective_signal = np.sin(base_signal * scale) + 0.25 * np.cos(base_signal / scale)
                                    outputs.append(objective_signal)
                                return np.column_stack(outputs)

                            self.evaluator.predict = _cold_start_predict
                            self.evaluator.models = {}
                            self.evaluator.metrics = {}
                            self.evaluator.fitted = True
                            st.warning(
                                "Running virtual evaluator in cold-start mode (no uploaded data/models). "
                                "Objective values are synthetic; upload data or custom models for realistic optimization."
                            )
                    else:
                        self.evaluator.fit(
                            X,
                            Y,
                            tune_hyperparams=config.enable_hyperparameter_tuning,
                            n_trials=config.n_tuning_trials if config.enable_hyperparameter_tuning else 0,
                            current_batch=0,
                            total_batches=max(1, config.batch_iterations),
                            use_evolving_constraints=config.use_evolving_constraints,
                            constraints_config=config.evolving_constraints_config,
                        )
                else:
                    # SDL/simulator: evaluator is used only for distance calculations
                    self.evaluator = GeneralizedEvaluator(
                        self.model_factory,
                        config.parameters,
                        config.objectives,
                        distance_normalization_config=getattr(config, "distance_normalization_config", None),
                    )

                self.optimizer = BayesianOptimizer(
                    config,
                    self.evaluator,
                    X,
                    Y,
                    Y_sem_data=Y_sem,
                    resume_state=resume_state,
                    sdl_client=sdl_connector,
                )
                st.session_state.evaluator = self.evaluator
                st.session_state.optimizer = self.optimizer
            else:
                self.optimizer = reuse_optimizer
                self.optimizer.config = config
                self.optimizer.X_data = np.array(X) if X is not None else np.zeros((0, len(config.parameters)))
                self.optimizer.Y_data = np.array(Y) if Y is not None else np.zeros((0, len(config.objectives)))
                if Y_sem is not None:
                    y_sem_arr = np.array(Y_sem, dtype=np.float64)
                    self.optimizer.Y_sem_data = y_sem_arr if y_sem_arr.shape == self.optimizer.Y_data.shape else np.full(self.optimizer.Y_data.shape, np.nan, dtype=np.float64)
                else:
                    self.optimizer.Y_sem_data = np.full(self.optimizer.Y_data.shape, np.nan, dtype=np.float64)
                self.optimizer.sdl_client = sdl_connector or getattr(self.optimizer, 'sdl_client', None)
                self.optimizer.direct_objectives = [
                    obj for obj in config.objectives
                    if (hasattr(obj.type, 'value') and obj.type in [ObjectiveType.MINIMIZE, ObjectiveType.MAXIMIZE])
                    or (not hasattr(obj.type, 'value') and obj.type in ['minimize', 'maximize'])
                ]
                self.optimizer.target_objectives = [
                    obj for obj in config.objectives
                    if (hasattr(obj.type, 'value') and obj.type in [ObjectiveType.TARGET_RANGE, ObjectiveType.TARGET_VALUE])
                    or (not hasattr(obj.type, 'value') and obj.type in ['target_range', 'target_value'])
                ]
                self.optimizer.uses_direct_objectives = len(self.optimizer.direct_objectives) > 0
                self.evaluator = st.session_state.get('evaluator', self.evaluator)
                if self.evaluator is not None and hasattr(self.evaluator, "set_distance_normalization_config"):
                    self.evaluator.set_distance_normalization_config(
                        getattr(config, "distance_normalization_config", None)
                    )

            st.session_state.run_status = 'running'
            st.session_state.optimization_progress['status'] = 'running'

            def update_progress(progress_info):
                status = progress_info.get('status', '')
                current_progress = st.session_state.optimization_progress
                uses_direct = current_progress.get('uses_direct_objectives', False)
                if status == 'starting_batch':
                    current_progress.update({
                        'current_batch': progress_info['batch'],
                        'total_batches': progress_info['total_batches'],
                        'status': 'running',
                        'current_trial': None
                    })
                elif status == 'completed_candidate':
                    current_progress['candidates_completed'] += 1
                    new_distance = progress_info.get('distance', current_progress.get('best_distance', float('inf')))
                    if not uses_direct and new_distance < current_progress.get('best_distance', float('inf')):
                        current_progress['best_distance'] = new_distance
                    current_progress.update({
                        'current_trial': {
                            'parameters': progress_info.get('parameters', {}),
                            'distance': progress_info.get('distance', 0),
                            'uncertainties': progress_info.get('uncertainties', {}),
                            'objective_values': progress_info.get('objective_values', {}),
                            'measured_objectives': progress_info.get('measured_objectives', {}),
                            'measurement_metadata': progress_info.get('measurement_metadata', {}),
                            'trial_index': progress_info.get('trial_index'),
                            'status': 'completed'
                        },
                        'status': 'running',
                        'current_parameters': progress_info.get('parameters', {}),
                        'current_uncertainties': progress_info.get('uncertainties', {}),
                        'current_objectives': progress_info.get('objective_values', {}),
                        'current_measured_objectives': progress_info.get('measured_objectives', {}),
                    })
                    if uses_direct and progress_info.get('objective_values'):
                        obj_hist = current_progress.setdefault('objective_history', {})
                        for obj_name, obj_val in progress_info.get('objective_values', {}).items():
                            values = obj_hist.setdefault(obj_name, [])
                            try:
                                values.append(float(obj_val))
                            except Exception:
                                values.append(obj_val)
                    if not uses_direct:
                        current_progress.setdefault('distance_history', []).append(new_distance)
                        current_progress.setdefault('batch_history', []).append(current_progress['current_batch'])
                    if 'uncertainties' in progress_info:
                        avg_uncertainty = np.mean(list(progress_info['uncertainties'].values()))
                        current_progress.setdefault('uncertainty_history', []).append(avg_uncertainty)
                try:
                    self._update_progress_display(progress_container)
                except Exception as e:
                    print(f"UI Update warning: {e}")

            result = self.optimizer.run_next_batch(progress_callback=update_progress) if single_batch else self.optimizer.run_optimization(progress_callback=update_progress)
            if result is not None:
                self.optimization_result = result
                st.session_state.optimization_result = result
                st.session_state.run_status = 'completed'
                st.session_state.optimization_progress['status'] = 'completed'
                st.session_state.optimization_progress['completed_at'] = time.time()
            else:
                st.session_state.run_status = 'paused'
                st.session_state.optimization_progress['status'] = 'awaiting_resume'
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            if 'optimization_progress' in st.session_state:
                st.session_state.optimization_progress['status'] = 'error'
    def _serialize_config(self, config: OptimizationConfig) -> Dict[str, Any]:
        return {
            'experiment_name': config.experiment_name,
            'evaluator_type': config.evaluator_type.value if hasattr(config.evaluator_type, 'value') else config.evaluator_type,
            'optimization_mode': config.optimization_mode.value if hasattr(config.optimization_mode, 'value') else config.optimization_mode,
            'sdl_settings': config.sdl_settings,
            'task_parameter_name': config.task_parameter_name,
            'parameters': [
                {
                    'name': p.name,
                    'type': p.type.value if hasattr(p.type, 'value') else p.type,
                    'bounds': p.bounds,
                    'categories': p.categories,
                    'step': p.step
                } for p in config.parameters
            ],
            'objectives': [
                {
                    'name': o.name,
                    'type': o.type.value if hasattr(o.type, 'value') else o.type,
                    'target_range': o.target_range,
                    'target_value': o.target_value,
                    'tolerance': o.tolerance,
                    'weight': o.weight
                } for o in config.objectives
            ],
            'models': [
                {
                    'name': m.name,
                    'type': m.type,
                    'hyperparameters': m.hyperparameters,
                    'enabled': m.enabled
                } for m in config.models
            ],
            'parameter_constraints': [
                {
                    'name': c.name,
                    'type': c.type,
                    'expression': c.expression,
                    'description': c.description
                } for c in config.parameter_constraints
            ],
            'optimization_settings': {
                'enable_hyperparameter_tuning': config.enable_hyperparameter_tuning,
                'n_tuning_trials': config.n_tuning_trials,
                'batch_iterations': config.batch_iterations,
                'batch_size': config.batch_size,
                'max_iterations': config.max_iterations,
                'optimization_mode': config.optimization_mode.value if hasattr(config.optimization_mode, 'value') else config.optimization_mode,
                'random_seed': config.random_seed,
                'n_initial_points': config.n_initial_points,
                'generation_strategy': config.generation_strategy.value if hasattr(config.generation_strategy, 'value') else config.generation_strategy,
                'acquisition_function': config.acquisition_function.value if hasattr(config.acquisition_function, 'value') else config.acquisition_function,
                'initialization_strategy': config.initialization_strategy.value if hasattr(config.initialization_strategy, 'value') else config.initialization_strategy,
                'initialization_trials': getattr(config, 'initialization_trials', config.sobol_points),
                'use_sobol': config.use_sobol,
                'sobol_points': config.sobol_points,
                'use_adaptive_search': config.use_adaptive_search,
                'adaptive_search_config': config.adaptive_search_config,
                'use_evolving_constraints': config.use_evolving_constraints,
                'evolving_constraints_config': config.evolving_constraints_config,
                'uncertainty_config': config.uncertainty_config,
                'distance_normalization_config': config.distance_normalization_config,
            }
        }

    def _config_from_dict(self, cfg_dict: Dict[str, Any]) -> OptimizationConfig:
        params = [ParameterConfig(**p) for p in cfg_dict.get('parameters', [])]
        objectives = [ObjectiveConfig(**o) for o in cfg_dict.get('objectives', [])]
        models = [ModelConfig(**m) for m in cfg_dict.get('models', [])]
        constraints = [ParameterConstraint(**c) for c in cfg_dict.get('parameter_constraints', [])]
        settings = cfg_dict.get('optimization_settings', {})
        default_init_strategy = settings.get(
            'initialization_strategy',
            cfg_dict.get(
                'initialization_strategy',
                'sobol' if settings.get('use_sobol', True) else 'none'
            )
        )
        default_init_trials = settings.get('initialization_trials')
        if default_init_trials is None:
            default_init_trials = cfg_dict.get('initialization_trials')
        if default_init_trials is None:
            default_init_trials = 0 if default_init_strategy == 'none' else settings.get('sobol_points', 10)
        return OptimizationConfig(
            experiment_name=cfg_dict.get('experiment_name', 'Loaded_Experiment'),
            parameters=params,
            objectives=objectives,
            models=models,
            evaluator_type=cfg_dict.get('evaluator_type', EvaluatorType.VIRTUAL),
            optimization_mode=cfg_dict.get('optimization_mode', settings.get('optimization_mode', OptimizationMode.BATCH)),
            sdl_settings=cfg_dict.get('sdl_settings', {}),
            task_parameter_name=cfg_dict.get('task_parameter_name'),
            parameter_constraints=constraints,
            enable_hyperparameter_tuning=settings.get('enable_hyperparameter_tuning', True),
            n_tuning_trials=settings.get('n_tuning_trials', 20),
            batch_iterations=settings.get('batch_iterations', 10),
            batch_size=settings.get('batch_size', 5),
            max_iterations=settings.get('max_iterations', 100),
            random_seed=settings.get('random_seed', 42),
            n_initial_points=settings.get('n_initial_points', 10),
            generation_strategy=settings.get('generation_strategy', 'default'),
            acquisition_function=settings.get(
                'acquisition_function',
                cfg_dict.get('acquisition_function', 'auto')
            ),
            initialization_strategy=default_init_strategy,
            initialization_trials=default_init_trials,
            use_sobol=settings.get('use_sobol', True),
            sobol_points=settings.get('sobol_points', 10),
            use_adaptive_search=settings.get('use_adaptive_search', True),
            adaptive_search_config=settings.get('adaptive_search_config'),
            use_evolving_constraints=settings.get('use_evolving_constraints', False),
            evolving_constraints_config=settings.get('evolving_constraints_config'),
            uncertainty_config=settings.get('uncertainty_config'),
            distance_normalization_config=settings.get('distance_normalization_config'),
        )

    def _save_checkpoint(self, config: OptimizationConfig, X: np.ndarray, Y: np.ndarray, Y_sem: Optional[np.ndarray] = None, optimizer: Optional[BayesianOptimizer] = None, progress: Optional[Dict[str, Any]] = None, filename: Optional[str] = None) -> str:
        checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        stamp = int(time.time())
        fname = filename or f"bo_checkpoint_{stamp}.json"
        path = os.path.join(checkpoint_dir, fname)
        payload = {
            'config': self._serialize_config(config),
            'X': X.tolist() if hasattr(X, 'tolist') else [],
            'Y': Y.tolist() if hasattr(Y, 'tolist') else [],
            'Y_sem': Y_sem.tolist() if hasattr(Y_sem, 'tolist') else [],
            'timestamp': stamp
        }
        if optimizer is not None:
            try:
                payload['optimizer_state'] = optimizer.export_state()
            except Exception as e:
                print(f"Warning: could not export optimizer state: {e}")
        if progress is not None:
            payload['progress'] = progress
        save_json(path, payload)
        st.session_state.checkpoint_info = {'path': path, 'timestamp': stamp}
        return path

    def _load_checkpoint(self, file_obj) -> Dict[str, Any]:
        data = json.load(file_obj)
        st.session_state.loaded_checkpoint = data
        return data

    def _render_progress_header(self, message: str):
        st.info(message)
        st.write("Please wait while the optimization runs...")

    def _update_progress_display(self, progress_container):
        progress = st.session_state.optimization_progress
        uses_direct = progress.get('uses_direct_objectives', False)
        status = progress.get('status', 'idle')
        optimizer = st.session_state.get('optimizer')
        with progress_container.container():
            st.subheader("Optimization Progress")
            col1, col2, col3, col4 = st.columns(4)
            total_batches = progress.get('total_batches', 0)
            current_batch = progress.get('current_batch', 0)
            col1.metric("Current Batch", f"{current_batch}/{total_batches}")
            col2.metric("Candidates Completed", f"{progress.get('candidates_completed', 0)}")
            if uses_direct:
                col3.metric("Mode", "Direct (Pareto)")
                col4.metric("Status", status.title())
            else:
                best_dist = progress.get('best_distance', float('inf'))
                col3.metric("Best Distance", "inf" if np.isinf(best_dist) else f"{best_dist:.6f}")
                col4.metric("Status", status.title())

            if status == 'completed':
                st.success("All batches completed.")
            elif status in ('paused', 'awaiting_resume'):
                st.info("Batch finished. Click Resume / Next Batch to continue.")
            elif status == 'stopped':
                st.warning("Optimization stopped. Load the saved checkpoint to continue later.")

            if uses_direct:
                obj_hist = progress.get('objective_history', {})
                names = list(obj_hist.keys())
                if len(names) >= 2:
                    df = pd.DataFrame({
                        names[0]: obj_hist.get(names[0], []),
                        names[1]: obj_hist.get(names[1], []),
                        'idx': list(range(1, len(obj_hist.get(names[0], [])) + 1))
                    })
                    fig = px.scatter(
                        df, x=names[0], y=names[1], color='idx',
                        title="Pareto exploration (latest candidates)",
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    fig.update_traces(marker=dict(size=9, line=dict(width=1, color='rgba(0,0,0,0.3)')))
                    st.plotly_chart(fig, use_container_width=True)
                elif len(names) == 1:
                    series = obj_hist[names[0]]
                    if series:
                        fig = px.line(
                            x=list(range(1, len(series) + 1)), y=series,
                            labels={'x': 'Candidate', 'y': names[0]},
                            title=f"{names[0]} trajectory", markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Waiting for objective values to plot Pareto progress.")
            else:
                dist_hist = progress.get('distance_history', [])
                if dist_hist:
                    batches = list(range(1, len(dist_hist) + 1))
                    fig = px.line(
                        x=batches, y=dist_hist, markers=True,
                        labels={'x': 'Candidate', 'y': 'Best distance'},
                        title="Distance convergence"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                unc_hist = progress.get('uncertainty_history', [])
                if unc_hist:
                    fig = px.line(
                        x=list(range(1, len(unc_hist) + 1)), y=unc_hist, markers=True,
                        labels={'x': 'Candidate', 'y': 'Avg uncertainty'},
                        title="Uncertainty trend"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Latest candidate snapshot (parameters & objectives)
            current_trial = progress.get('current_trial')
            if current_trial:
                st.subheader("Latest Candidate")
                c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.0, 0.8])
                with c1:
                    st.markdown("**Parameters**")
                    st.json(current_trial.get('parameters', {}))
                with c2:
                    st.markdown("**Objective values (evaluator)**")
                    st.json(current_trial.get('objective_values', {}))
                with c3:
                    st.markdown("**Measured objective means**")
                    measured_obj = current_trial.get('measured_objectives', {})
                    st.json(measured_obj if isinstance(measured_obj, dict) else {})
                with c4:
                    st.markdown("**Uncertainty (exact)**")
                    unc_json = current_trial.get('uncertainties', {})
                    st.json(unc_json if isinstance(unc_json, dict) else {})
                    if uses_direct:
                        st.metric("Pareto point #", progress.get('candidates_completed', 0))
                    else:
                        st.metric("Distance", f"{current_trial.get('distance', float('inf')):.6f}")
                    measurement_meta = current_trial.get('measurement_metadata', {})
                    if isinstance(measurement_meta, dict) and measurement_meta:
                        st.caption("Measurement metadata")
                        st.json(measurement_meta)

            # Rolling table of last 10 candidates pulled from optimizer (if available)
            if optimizer is not None and hasattr(optimizer, 'all_candidates') and optimizer.all_candidates:
                tail = optimizer.all_candidates[-10:]
                rows = []
                for i, cand in enumerate(tail, 1):
                    row = {
                        "Seq": len(optimizer.all_candidates) - len(tail) + i,
                        "Distance": cand.get("distance"),
                        "Type": "Experimental" if cand.get("is_experimental") else "Suggested",
                    }
                    # Merge parameters and objectives into the row
                    for k, v in cand.get("parameters", {}).items():
                        row[f"param::{k}"] = v
                    for k, v in (cand.get("objective_values") or {}).items():
                        row[f"obj::{k}"] = v
                    rows.append(row)
                st.subheader("Recent candidates (live)")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Live parallel coordinates for latest candidates
            if optimizer is not None and hasattr(optimizer, 'all_candidates'):
                params_cfg = st.session_state.experiment_config.parameters if st.session_state.get('experiment_config') else []
                objs_cfg = st.session_state.experiment_config.objectives if st.session_state.get('experiment_config') else []
                try:
                    df_live = self.viz_engine._candidate_dataframe(
                        type('tmp', (), {'all_candidates': optimizer.all_candidates})(),
                        params_cfg,
                        objs_cfg
                    )
                    if not df_live.empty and len(df_live) > 1:
                        # Limit to recent 50 for readability
                        df_live = df_live.tail(50)
                        param_cols = [p.name for p in params_cfg if p.name in df_live.columns]
                        obj_cols = [o.name for o in objs_cfg if o.name in df_live.columns]
                        use_cols = param_cols + obj_cols
                        if len(use_cols) >= 2:
                            fig_pc = px.parallel_coordinates(
                                df_live,
                                dimensions=use_cols,
                                color='index',
                                color_continuous_scale=px.colors.sequential.Plasma,
                                labels={c: c for c in use_cols}
                            )
                            st.subheader("Parallel coordinates (live)")
                            st.plotly_chart(fig_pc, use_container_width=True)
                except Exception as e:
                    st.debug(f"Parallel coord live plot error: {e}")

    def _display_optimization_status(self):
        if not hasattr(self, 'optimization_result'):
            return
        result = self.optimization_result
        uses_direct = getattr(result, 'uses_direct_objectives', False)
        st.subheader("Current Optimization Status")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mode", "Direct (Pareto)" if uses_direct else "Distance")
        col2.metric("Batches Completed", len(result.history))
        col3.metric("Candidates", len(result.all_candidates))
        col4.metric("Status", "Completed" if result.best_parameters else "In Progress")

    def _display_optimization_completion(self, total_time):
        st.success("Optimization Completed Successfully!")
        self._display_optimization_status()

    def render_results(self, show_header: bool = True):
        """Render results visualization section"""
        if show_header:
            st.header("Results")
        if (st.session_state.get('evaluator') is None or
            st.session_state.get('optimization_result') is None):
            st.warning("Please run optimization first in the 'Optimization' section.")
            return
        evaluator = st.session_state.evaluator
        result = st.session_state.optimization_result
        config = st.session_state.experiment_config
        data = st.session_state.get('optimization_data', {})
        X = data.get('X')
        Y = data.get('Y')
        self.viz_engine.create_optimization_dashboard(
            result, evaluator, config.parameters, config.objectives, X, Y
        )

    def render_export(self, show_header: bool = True):
        """Render export, report generation, and database management."""
        if show_header:
            st.header("Export & Import")

        config = st.session_state.get('experiment_config')
        result = st.session_state.get('optimization_result')

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Export Configuration")
            if config is None:
                st.warning("No experiment configuration to export.")
            else:
                config_yaml = yaml.dump(self._serialize_config(config), default_flow_style=False)
                st.download_button(
                    label="Download Configuration (YAML)",
                    data=config_yaml,
                    file_name=f"{config.experiment_name}_config.yaml",
                    mime="text/yaml",
                    use_container_width=True
                )
        with col2:
            st.subheader("Import Configuration")
            uploaded_config = st.file_uploader(
                "Upload YAML Configuration",
                type=['yaml', 'yml'],
                key="import_config_export_uploader"
            )
            if uploaded_config is not None:
                try:
                    cfg = yaml.safe_load(uploaded_config)
                    st.session_state.experiment_config = self._config_from_dict(cfg)
                    st.success("Configuration loaded.")
                except Exception as e:
                    st.error(f"Failed to import configuration: {e}")

        st.divider()
        st.subheader("Run Report, Tables, Charts")
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            export_root = st.text_input(
                "Export root directory",
                value=st.session_state.get('export_root_dir', os.path.join(os.getcwd(), "exports")),
                key="export_root_dir_input",
                help="Artifacts are written into a timestamped folder under this directory.",
            )
            st.session_state.export_root_dir = export_root
        with export_col2:
            db_path = st.text_input(
                "SQLite database path",
                value=st.session_state.get('export_db_path', os.path.join(os.getcwd(), "exports", "bo_platform.db")),
                key="export_db_path_input",
                help="Used when saving run metadata and tables into SQLite.",
            )
            st.session_state.export_db_path = db_path

        save_to_database = st.checkbox(
            "Save generated run into SQLite database",
            value=True,
            key="save_run_to_database_checkbox",
        )

        if st.button("Generate Full Report + Save Tables + PNG Charts", use_container_width=True, type="primary", key="generate_full_export_btn"):
            if config is None:
                st.error("No experiment configuration available.")
            elif result is None:
                st.error("No optimization result available. Run optimization first.")
            else:
                try:
                    artifacts = export_run_artifacts(
                        config=config,
                        result=result,
                        export_root=export_root,
                        config_payload=self._serialize_config(config),
                    )

                    db_exp_id = None
                    if save_to_database:
                        db_manager = ExperimentDatabaseManager(db_path=db_path)
                        db_manager.initialize()
                        db_exp_id = db_manager.save_experiment_run(
                            config=config,
                            result=result,
                            artifacts=artifacts,
                        )
                        artifacts["database_experiment_id"] = db_exp_id

                    st.session_state.last_export_artifacts = artifacts
                    success_msg = f"Export generated at: {artifacts['export_dir']}"
                    if db_exp_id is not None:
                        success_msg += f" | Database experiment_id: {db_exp_id}"
                    st.success(success_msg)
                except Exception as e:
                    st.error(f"Failed to generate export artifacts: {e}")

        artifacts = st.session_state.get('last_export_artifacts')
        if artifacts:
            st.markdown("**Latest export artifacts**")
            st.write(f"- Export directory: `{artifacts.get('export_dir')}`")
            st.write(f"- Report HTML: `{artifacts.get('report_html')}`")
            st.write(f"- Bundle ZIP: `{artifacts.get('bundle_zip')}`")
            if artifacts.get("database_experiment_id") is not None:
                st.write(f"- Database experiment ID: `{artifacts.get('database_experiment_id')}`")

            if artifacts.get("report_html") and os.path.exists(artifacts["report_html"]):
                with open(artifacts["report_html"], "rb") as f:
                    st.download_button(
                        "Download HTML Report",
                        data=f.read(),
                        file_name=Path(artifacts["report_html"]).name,
                        mime="text/html",
                        key="download_html_report_btn",
                    )
            if artifacts.get("bundle_zip") and os.path.exists(artifacts["bundle_zip"]):
                with open(artifacts["bundle_zip"], "rb") as f:
                    st.download_button(
                        "Download Full Bundle (ZIP)",
                        data=f.read(),
                        file_name=Path(artifacts["bundle_zip"]).name,
                        mime="application/zip",
                        key="download_bundle_zip_btn",
                    )
            if artifacts.get("summary_json") and os.path.exists(artifacts["summary_json"]):
                with open(artifacts["summary_json"], "rb") as f:
                    st.download_button(
                        "Download Summary JSON",
                        data=f.read(),
                        file_name=Path(artifacts["summary_json"]).name,
                        mime="application/json",
                        key="download_summary_json_btn",
                    )

        st.divider()
        st.subheader("Database Management")
        db_manager = ExperimentDatabaseManager(db_path=st.session_state.export_db_path)

        db_col1, db_col2 = st.columns(2)
        with db_col1:
            if st.button("Initialize / Repair Database", key="init_db_btn", use_container_width=True):
                try:
                    db_manager.initialize()
                    st.success(f"Database ready: {st.session_state.export_db_path}")
                except Exception as e:
                    st.error(f"Failed to initialize database: {e}")
        with db_col2:
            refresh_db = st.button("Refresh Database View", key="refresh_db_view_btn", use_container_width=True)

        should_show_db = refresh_db or Path(st.session_state.export_db_path).exists()
        if should_show_db:
            try:
                db_manager.initialize()
                experiments_df = db_manager.list_experiments(limit=200)
                if experiments_df.empty:
                    st.info("No runs stored in database yet.")
                else:
                    st.markdown("**Stored Experiments**")
                    st.dataframe(experiments_df, use_container_width=True)

                    selected_id = st.selectbox(
                        "Select experiment ID for details",
                        options=experiments_df["id"].tolist(),
                        key="db_selected_experiment_id",
                    )
                    details_col1, details_col2 = st.columns(2)
                    with details_col1:
                        candidates_df = db_manager.get_candidates(int(selected_id))
                        st.markdown("**Candidates (from DB)**")
                        st.dataframe(candidates_df, use_container_width=True)
                    with details_col2:
                        batch_df = db_manager.get_batch_history(int(selected_id))
                        st.markdown("**Batch History (from DB)**")
                        st.dataframe(batch_df, use_container_width=True)
            except Exception as e:
                st.error(f"Database view failed: {e}")


def main():
    st.set_page_config(
        page_title="Bayesian Optimization Platform",
        page_icon="logo.svg",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bo-bg-a: #0b131a;
        --bo-bg-b: #11212b;
        --bo-bg-c: #17323f;
        --bo-card: #111f29;
        --bo-card-2: #162936;
        --bo-border: #274654;
        --bo-accent: #19a974;
        --bo-accent-strong: #0f8c5f;
        --bo-sidebar-a: #081018;
        --bo-sidebar-b: #10202b;
        --bo-text: #e7f2f6;
        --bo-muted: #a9c0c9;
    }
    .stApp {
        background:
            radial-gradient(980px 500px at 4% 0%, var(--bo-bg-c), transparent 65%),
            radial-gradient(920px 440px at 100% 0%, var(--bo-bg-b), transparent 66%),
            linear-gradient(180deg, var(--bo-bg-a) 0%, #0a1118 100%);
        color: var(--bo-text);
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bo-sidebar-a) 0%, var(--bo-sidebar-b) 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f2faf8 !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        padding: 6px 8px;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] .stRadio > div:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    .stTextArea textarea {
        background: var(--bo-card-2) !important;
        color: var(--bo-text) !important;
        border: 1px solid var(--bo-border) !important;
    }
    [data-testid="stDataFrame"],
    [data-testid="stExpander"] {
        border: 1px solid var(--bo-border);
        border-radius: 12px;
        background: rgba(17, 31, 41, 0.72);
    }
    [data-testid="stMetric"] {
        border: 1px solid var(--bo-border);
        border-radius: 12px;
        padding: 10px 12px;
        background: var(--bo-card);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.28);
    }
    .stAlert {
        border-radius: 12px;
        border: 1px solid var(--bo-border);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.24);
        background: rgba(17, 31, 41, 0.8);
    }
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid var(--bo-accent);
        background: linear-gradient(180deg, var(--bo-accent) 0%, var(--bo-accent-strong) 100%);
        color: #ffffff;
        font-weight: 600;
    }
    div.stButton > button:hover {
        border-color: var(--bo-accent-strong);
        filter: brightness(1.05);
    }
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid var(--bo-border);
        background: var(--bo-card-2);
        color: var(--bo-text);
    }
    .stTabs [role="tab"] {
        background: var(--bo-card-2);
        border: 1px solid var(--bo-border);
        border-radius: 8px;
        color: var(--bo-text);
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, #1f8f6a 0%, #157a59 100%);
        color: #ffffff;
        border-color: #1f8f6a;
    }
    [data-testid="stSidebar"] h3 {
        margin-top: 14px;
        margin-bottom: 8px;
        font-size: 0.95rem;
        letter-spacing: 0.2px;
        color: #d8edf4 !important;
    }
    p, li, label, span {
        color: var(--bo-text);
    }
    .stCaption, small {
        color: var(--bo-muted) !important;
    }
    h1, h2, h3 {
        letter-spacing: 0.15px;
        color: #f0fbff;
    }
    .block-container {
        padding-top: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    platform = BOPlatform()
    platform.render_sidebar()

if __name__ == "__main__":
    main()
