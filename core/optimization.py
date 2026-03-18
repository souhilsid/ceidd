# core/optimization.py - UPDATED WITH FULLYBAYESIAN AND BETTER UNCERTAINTY

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import importlib
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy as AxGenerationStrategy
from ax.modelbridge.registry import Models
from ax.core.observation import ObservationFeatures
from ax.core.data import Data
from ax.core.types import ComparisonOp
from utils.state_manager import save_json, load_json, dump_ax_to_dict, load_ax_from_dict
import warnings

from .config import (
    OptimizationConfig,
    ParameterConfig,
    ObjectiveConfig,
    ParameterType,
    ObjectiveType,
    BOGenerationStrategy,
    BOAcquisitionFunction,
    BOInitializationStrategy,
    ACQF_CUSTOMIZABLE_STRATEGIES,
    SINGLE_OBJECTIVE_ACQFS,
    MULTI_OBJECTIVE_ACQFS,
    EvaluatorType,
    OptimizationMode,
)
from .evaluators import GeneralizedEvaluator
from .sdl import SDLConnector

warnings.filterwarnings('ignore')

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    best_parameters: Dict[str, float] = field(default_factory=dict)
    best_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    best_distance: float = float('inf')
    history: List[Dict[str, Any]] = field(default_factory=list)
    all_candidates: List[Dict[str, Any]] = field(default_factory=list)
    model_performance: Dict[str, Any] = field(default_factory=dict)
    pareto_front: List[Dict[str, Any]] = field(default_factory=list)
    uses_direct_objectives: bool = False
    
    def __post_init__(self):
        # Ensure best_predictions is always a numpy array
        if self.best_predictions is None:
            self.best_predictions = np.array([])
        elif not isinstance(self.best_predictions, np.ndarray):
            self.best_predictions = np.array(self.best_predictions)
        
        # Ensure best_parameters is always a dict
        if self.best_parameters is None:
            self.best_parameters = {}
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        return pd.DataFrame(self.history)

class BayesianOptimizer:
    """Main Bayesian Optimization engine"""
    
    def __init__(
        self,
        config: OptimizationConfig,
        evaluator: Optional[GeneralizedEvaluator],
        X_data: Optional[np.ndarray],
        Y_data: Optional[np.ndarray],
        Y_sem_data: Optional[np.ndarray] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        sdl_client: Optional[SDLConnector] = None,
    ):
        self.config = config
        self.evaluator = evaluator
        self.X_data = np.array(X_data) if X_data is not None else np.zeros((0, len(config.parameters)))
        self.Y_data = np.array(Y_data) if Y_data is not None else np.zeros((0, len(config.objectives)))
        if Y_sem_data is not None:
            y_sem_arr = np.array(Y_sem_data, dtype=np.float64)
            if y_sem_arr.ndim == 1:
                y_sem_arr = y_sem_arr.reshape(-1, 1)
            if y_sem_arr.shape == self.Y_data.shape:
                self.Y_sem_data = y_sem_arr
            else:
                self.Y_sem_data = np.full(self.Y_data.shape, np.nan, dtype=np.float64)
        else:
            self.Y_sem_data = np.full(self.Y_data.shape, np.nan, dtype=np.float64)
        self.evaluator_type = config.evaluator_type
        self.optimization_mode = config.optimization_mode
        self.sdl_client = sdl_client
        if self.evaluator_type == EvaluatorType.SELF_DRIVING_LAB and self.sdl_client is None:
            raise ValueError("SDL connector is required for self-driving lab evaluations")
        self.direct_objectives = [
            obj for obj in self.config.objectives
            if (hasattr(obj.type, "value") and obj.type in [ObjectiveType.MINIMIZE, ObjectiveType.MAXIMIZE])
            or (not hasattr(obj.type, "value") and obj.type in ["minimize", "maximize"])
        ]
        self.target_objectives = [
            obj for obj in self.config.objectives
            if (hasattr(obj.type, "value") and obj.type in [ObjectiveType.TARGET_RANGE, ObjectiveType.TARGET_VALUE])
            or (not hasattr(obj.type, "value") and obj.type in ["target_range", "target_value"])
        ]
        self.uses_direct_objectives = len(self.direct_objectives) > 0

        # Track optimization state
        self.best_overall_distance = float('inf')
        self.best_overall_parameters = {}
        self.best_overall_predictions = np.array([])
        self.history = []
        self.all_candidates = []
        self.completed_batches = 0
        self.current_adaptive_bounds: Dict[str, Tuple[float, float]] = {}
        self.adaptive_bounds_history: List[Dict[str, Any]] = []
        self._resume_state = resume_state

        # Store original data for reference
        self.original_X = self.X_data.copy()
        self.original_Y = self.Y_data.copy()
        self.original_Y_sem = self.Y_sem_data.copy()

        # Validate strategy compatibility early so UI can surface actionable errors.
        self._validate_strategy_conditions()

        if resume_state is not None:
            self._restore_from_state(resume_state)
            return

        # Print parameter configuration for debugging
        self._print_parameter_configuration()
        
        # Initialize Ax client with experimental data
        self.ax_client = self._initialize_ax_client()
        self._initialize_with_experimental_data()

    def _strategy_value(self) -> str:
        gs = self.config.generation_strategy
        return gs.value if hasattr(gs, "value") else str(gs)

    def _is_mtgp_strategy(self) -> bool:
        return self.config.generation_strategy in {
            BOGenerationStrategy.ST_MTGP,
            BOGenerationStrategy.SAAS_MTGP,
        }

    def _get_task_parameter_name(self) -> Optional[str]:
        configured = getattr(self.config, "task_parameter_name", None)
        if isinstance(configured, str):
            configured = configured.strip()
        if configured:
            return configured
        # Backward-compatible fallback for old checkpoints/configs.
        for default_name in ("task", "task_id"):
            if any(p.name == default_name for p in self.config.parameters):
                return default_name
        return None

    def _selected_acquisition_function(self) -> BOAcquisitionFunction:
        configured = getattr(self.config, "acquisition_function", BOAcquisitionFunction.AUTO)
        if isinstance(configured, str):
            try:
                return BOAcquisitionFunction(configured)
            except Exception:
                return BOAcquisitionFunction.AUTO
        return configured

    def _supports_acquisition_customization(self) -> bool:
        return self.config.generation_strategy in ACQF_CUSTOMIZABLE_STRATEGIES

    def _initialization_strategy_value(self) -> str:
        strategy = getattr(self.config, "initialization_strategy", BOInitializationStrategy.SOBOL)
        if isinstance(strategy, str):
            try:
                strategy = BOInitializationStrategy(strategy)
            except Exception:
                strategy = BOInitializationStrategy.SOBOL if getattr(self.config, "use_sobol", True) else BOInitializationStrategy.NONE
        return strategy.value

    def _initialization_trials_value(self) -> int:
        trials = getattr(self.config, "initialization_trials", None)
        if trials is None:
            trials = getattr(self.config, "sobol_points", 0)
        try:
            trials = int(trials)
        except Exception:
            trials = 0
        return max(0, trials)

    def _is_multi_objective_setup(self) -> bool:
        if not self.uses_direct_objectives:
            return False
        return (len(self.direct_objectives) + len(self.target_objectives)) > 1

    def _resolve_botorch_acquisition_class(self, acqf: BOAcquisitionFunction):
        """Resolve configured acquisition function enum to BoTorch class."""
        class_paths = {
            BOAcquisitionFunction.Q_LOG_NEI: ("botorch.acquisition.logei", "qLogNoisyExpectedImprovement"),
            BOAcquisitionFunction.Q_NEI: ("botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement"),
            BOAcquisitionFunction.Q_EI: ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
            BOAcquisitionFunction.Q_LOG_EI: ("botorch.acquisition.logei", "qLogExpectedImprovement"),
            BOAcquisitionFunction.Q_KG: ("botorch.acquisition.knowledge_gradient", "qKnowledgeGradient"),
            BOAcquisitionFunction.Q_SIMPLE_REGRET: ("botorch.acquisition.monte_carlo", "qSimpleRegret"),
            BOAcquisitionFunction.Q_UCB: ("botorch.acquisition.monte_carlo", "qUpperConfidenceBound"),
            BOAcquisitionFunction.Q_LOG_POF: ("botorch.acquisition.logei", "qLogProbabilityOfFeasibility"),
            BOAcquisitionFunction.Q_LOG_NEHVI: ("botorch.acquisition.multi_objective.logei", "qLogNoisyExpectedHypervolumeImprovement"),
            BOAcquisitionFunction.Q_NEHVI: ("botorch.acquisition.multi_objective.monte_carlo", "qNoisyExpectedHypervolumeImprovement"),
            BOAcquisitionFunction.Q_EHVI: ("botorch.acquisition.multi_objective.monte_carlo", "qExpectedHypervolumeImprovement"),
            BOAcquisitionFunction.Q_LOG_EHVI: ("botorch.acquisition.multi_objective.logei", "qLogExpectedHypervolumeImprovement"),
            BOAcquisitionFunction.Q_LOG_NPAREGO: ("botorch.acquisition.multi_objective.parego", "qLogNParEGO"),
        }
        if acqf not in class_paths:
            raise ValueError(f"Unsupported acquisition function: {acqf.value}")

        module_name, class_name = class_paths[acqf]
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ValueError(
                f"Could not import module '{module_name}' for acquisition "
                f"'{acqf.value}'."
            ) from exc

        if not hasattr(module, class_name):
            raise ValueError(
                f"Acquisition class '{class_name}' was not found in '{module_name}'."
            )
        return getattr(module, class_name)

    def _build_acquisition_model_kwargs(self, resolved_model_name: str) -> Dict[str, Any]:
        selected_acqf = self._selected_acquisition_function()
        if selected_acqf == BOAcquisitionFunction.AUTO:
            return {}

        supported_model_names = {
            "BOTORCH_MODULAR",
            "BO_MIXED",
            "SAASBO",
            "FULLYBAYESIAN",
            "ST_MTGP",
            "ST_MTGP_NEHVI",
            "ST_MTGP_LEGACY",
            "SAAS_MTGP",
            "FULLYBAYESIAN_MTGP",
        }
        if resolved_model_name not in supported_model_names:
            print(
                f"Warning: Custom acquisition '{selected_acqf.value}' is not applied "
                f"for Ax model '{resolved_model_name}'."
            )
            return {}

        acqf_class = self._resolve_botorch_acquisition_class(selected_acqf)
        print(f"Using custom acquisition function: {selected_acqf.value}")
        return {"botorch_acqf_class": acqf_class}

    def _estimate_search_space_cardinality(self) -> Optional[int]:
        """Return finite cardinality for enumerated spaces, else None for continuous."""
        total = 1
        for p in self.config.parameters:
            p_type = p.type.value if hasattr(p.type, "value") else str(p.type)
            if p_type == "continuous":
                return None
            if p_type == "categorical":
                n_vals = len(p.categories or [])
                if n_vals <= 0:
                    return 0
                total *= n_vals
            elif p_type == "discrete":
                if p.bounds is None:
                    return 0
                step = float(getattr(p, "step", 0) or 0)
                if step <= 0:
                    return 0
                low, high = p.bounds
                if high < low:
                    return 0
                count = int(np.floor(((high - low) / step) + 1e-9)) + 1
                if count <= 0:
                    return 0
                total *= count
            else:
                return 0
            if total > 10_000_000:
                # No need to keep multiplying huge combinatorial spaces.
                return total
        return int(total)

    def _search_space_profile(self) -> Dict[str, Any]:
        param_types = [
            (p.type.value if hasattr(p.type, "value") else str(p.type))
            for p in self.config.parameters
        ]
        has_cont = any(t == "continuous" for t in param_types)
        has_cat = any(t == "categorical" for t in param_types)
        card = self._estimate_search_space_cardinality()
        task_name = self._get_task_parameter_name()

        unique_tasks = None
        if task_name and self.X_data is not None and self.X_data.size > 0:
            try:
                idx = [p.name for p in self.config.parameters].index(task_name)
                unique_tasks = int(len(np.unique(self.X_data[:, idx])))
            except Exception:
                unique_tasks = None

        return {
            "n_parameters": len(self.config.parameters),
            "n_objectives": len(self.config.objectives),
            "n_observations": int(self.X_data.shape[0]) if self.X_data is not None else 0,
            "has_continuous": has_cont,
            "has_categorical": has_cat,
            "cardinality": card,
            "task_parameter_name": task_name,
            "unique_tasks": unique_tasks,
        }

    def _validate_strategy_conditions(self):
        """Validate selected Ax strategy against current search-space/data conditions."""
        strategy = self.config.generation_strategy
        profile = self._search_space_profile()
        errors: List[str] = []
        warnings_list: List[str] = []

        discrete_only = {
            BOGenerationStrategy.THOMPSON,
            BOGenerationStrategy.EMPIRICAL_BAYES_THOMPSON,
            BOGenerationStrategy.EB_ASHR,
            BOGenerationStrategy.FACTORIAL,
        }
        empirical_bandits = {
            BOGenerationStrategy.THOMPSON,
            BOGenerationStrategy.EMPIRICAL_BAYES_THOMPSON,
            BOGenerationStrategy.EB_ASHR,
        }

        if strategy in discrete_only and profile["has_continuous"]:
            errors.append(
                f"{self._strategy_value()} requires a fully enumerated search space "
                "(categorical/discrete only, no continuous parameters)."
            )

        if strategy in empirical_bandits and profile["n_observations"] < 5:
            errors.append(
                f"{self._strategy_value()} requires historical observations (>= 5 rows). "
                "Upload data first."
            )

        if strategy == BOGenerationStrategy.FACTORIAL:
            card = profile["cardinality"]
            if card is None:
                errors.append("FACTORIAL cannot run with continuous parameters.")
            elif card <= 0:
                errors.append("FACTORIAL requires valid finite choices for all parameters.")
            elif card > 500:
                errors.append(
                    f"FACTORIAL expands to {card} combinations; this platform caps it at 500."
                )

        if strategy == BOGenerationStrategy.BO_MIXED:
            if not profile["has_categorical"]:
                errors.append("BO_MIXED requires at least one categorical parameter.")
            card = profile["cardinality"]
            if card is not None and card > 10_000 and not profile["has_continuous"]:
                errors.append(
                    f"BO_MIXED would enumerate {card} discrete combinations (>10000). "
                    "Reduce categories or use BOTORCH_MODULAR/SAASBO."
                )

        if strategy in {BOGenerationStrategy.ST_MTGP, BOGenerationStrategy.SAAS_MTGP}:
            task_name = profile["task_parameter_name"]
            if self.evaluator_type != EvaluatorType.VIRTUAL:
                errors.append(
                    f"{self._strategy_value()} is only supported with virtual evaluators."
                )
            if not task_name:
                errors.append(
                    f"{self._strategy_value()} requires `task_parameter_name` to be set."
                )
            else:
                matching = [p for p in self.config.parameters if p.name == task_name]
                if not matching:
                    errors.append(f"Task parameter '{task_name}' is not in the parameter list.")
                else:
                    p_type = matching[0].type.value if hasattr(matching[0].type, "value") else str(matching[0].type)
                    if p_type not in {"categorical", "discrete"}:
                        errors.append(
                            f"Task parameter '{task_name}' must be categorical or discrete."
                        )
            if profile["n_observations"] <= 0:
                errors.append(
                    f"{self._strategy_value()} requires historical data with at least two tasks."
                )
            elif profile["unique_tasks"] is not None and profile["unique_tasks"] < 2:
                errors.append(
                    f"{self._strategy_value()} requires >= 2 unique task values in "
                    f"'{task_name}', found {profile['unique_tasks']}."
                )

        if strategy in {BOGenerationStrategy.SAASBO, BOGenerationStrategy.FULLYBAYESIAN}:
            if profile["n_parameters"] < 4:
                warnings_list.append(
                    f"{self._strategy_value()} is usually more useful in higher dimensions "
                    "(>= 4 parameters)."
                )

        init_strategy = self._initialization_strategy_value()
        init_trials = self._initialization_trials_value()
        if (
            strategy in {BOGenerationStrategy.UNIFORM, BOGenerationStrategy.FACTORIAL}
            and init_strategy != BOInitializationStrategy.NONE.value
            and init_trials > 0
        ):
            warnings_list.append(
                f"Initialization step '{init_strategy}' is ignored for {self._strategy_value()}."
            )

        selected_acqf = self._selected_acquisition_function()
        if selected_acqf != BOAcquisitionFunction.AUTO:
            if not self._supports_acquisition_customization():
                errors.append(
                    f"{self._strategy_value()} does not support custom acquisition "
                    "functions. Use acquisition_function='auto'."
                )
            else:
                is_moo = self._is_multi_objective_setup()
                allowed = MULTI_OBJECTIVE_ACQFS if is_moo else SINGLE_OBJECTIVE_ACQFS
                if selected_acqf not in allowed:
                    mode_label = "multi-objective" if is_moo else "single-objective"
                    errors.append(
                        f"Acquisition '{selected_acqf.value}' is not supported for "
                        f"{mode_label} setups. Allowed: {[a.value for a in allowed]}."
                    )

        for w in warnings_list:
            print(f"Warning: {w}")

        if errors:
            formatted = "\n - ".join(errors)
            raise ValueError(
                f"Generation strategy '{self._strategy_value()}' is incompatible with the current setup:\n - {formatted}"
            )

    def _resolve_ax_model(self, preferred_names: List[str], strategy_label: str):
        """Resolve a strategy name across Ax versions (legacy + modern aliases)."""
        for idx, model_name in enumerate(preferred_names):
            if hasattr(Models, model_name):
                if idx > 0:
                    print(
                        f"Warning: Ax model '{preferred_names[0]}' unavailable. "
                        f"Using '{model_name}' for strategy {strategy_label}."
                    )
                return getattr(Models, model_name), model_name

        available_models = [name for name in dir(Models) if name.isupper()]
        raise ValueError(
            f"Ax installation does not support strategy '{strategy_label}'. "
            f"Tried {preferred_names}. Available model symbols: {available_models}"
        )


    def _restore_from_state(self, state: Dict[str, Any]):
        """Restore optimizer and Ax client from a saved state"""
        try:
            if state.get('ax_state'):
                self.ax_client = load_ax_from_dict(AxClient, state['ax_state'])
            else:
                self.ax_client = self._initialize_ax_client()
                self._initialize_with_experimental_data()
        except Exception as e:
            print(f"Warning: failed to load Ax state ({e}), reinitializing")
            self.ax_client = self._initialize_ax_client()
            self._initialize_with_experimental_data()
    
        self.history = state.get('history', [])
        self.all_candidates = state.get('all_candidates', [])
        self.best_overall_distance = state.get('best_overall_distance', self.best_overall_distance)
        self.best_overall_parameters = state.get('best_overall_parameters', self.best_overall_parameters)
        self.best_overall_predictions = np.array(state.get('best_overall_predictions', self.best_overall_predictions))
        self.completed_batches = state.get('completed_batches', len(self.history))
        self.uses_direct_objectives = state.get('uses_direct_objectives', self.uses_direct_objectives)
        self.X_data = np.array(state.get('X_data', self.X_data))
        self.Y_data = np.array(state.get('Y_data', self.Y_data))
        self.Y_sem_data = np.array(state.get('Y_sem_data', self.Y_sem_data))
        if self.Y_sem_data.shape != self.Y_data.shape:
            self.Y_sem_data = np.full(self.Y_data.shape, np.nan, dtype=np.float64)
        self.current_adaptive_bounds = {
            k: (float(v[0]), float(v[1]))
            for k, v in (state.get('current_adaptive_bounds') or {}).items()
            if isinstance(v, (list, tuple)) and len(v) == 2
        }
        self.adaptive_bounds_history = state.get('adaptive_bounds_history', [])
        self.original_Y_sem = self.Y_sem_data.copy()
    
    def export_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable checkpoint for the current run"""
        state = {
            'history': self.history,
            'all_candidates': self.all_candidates,
            'best_overall_distance': self.best_overall_distance,
            'best_overall_parameters': self.best_overall_parameters,
            'best_overall_predictions': self.best_overall_predictions.tolist() if hasattr(self.best_overall_predictions, 'tolist') else [],
            'completed_batches': getattr(self, 'completed_batches', 0),
            'uses_direct_objectives': self.uses_direct_objectives,
            'X_data': self.X_data.tolist() if hasattr(self.X_data, 'tolist') else [],
            'Y_data': self.Y_data.tolist() if hasattr(self.Y_data, 'tolist') else [],
            'Y_sem_data': self.Y_sem_data.tolist() if hasattr(self.Y_sem_data, 'tolist') else [],
            'current_adaptive_bounds': {
                k: [float(v[0]), float(v[1])]
                for k, v in getattr(self, 'current_adaptive_bounds', {}).items()
            },
            'adaptive_bounds_history': getattr(self, 'adaptive_bounds_history', []),
        }
        try:
            state['ax_state'] = dump_ax_to_dict(self.ax_client)
        except Exception as e:
            print(f"Warning: failed to serialize Ax client: {e}")
        return state
    
    def save_state(self, path: str) -> str:
        """Persist checkpoint to disk"""
        return save_json(path, self.export_state())
    
    def _print_parameter_configuration(self):
        """Print parameter configuration for debugging - ENHANCED FOR STEPS"""
        print("ðŸ”§ PARAMETER CONFIGURATION:")
        for i, param_config in enumerate(self.config.parameters):
            param_type = param_config.type.value if hasattr(param_config.type, 'value') else param_config.type
            print(f"  {i+1}. {param_config.name}: {param_type}")
            if param_type == "discrete":
                step = getattr(param_config, 'step', 1)
                print(f"     Bounds: {param_config.bounds}, Step: {step}")
                # Show expected values
                min_val, max_val = param_config.bounds
                if step > 0:
                    possible_values = []
                    current = min_val
                    while current <= max_val:
                        possible_values.append(round(current, 3))
                        current += step
                        if len(possible_values) > 10:  # Limit output
                            possible_values.append("...")
                            break
                    print(f"     Possible values: {possible_values}")
            elif param_type == "categorical":
                print(f"     Categories: {getattr(param_config, 'categories', [])}")
            else:
                print(f"     Bounds: {param_config.bounds}")

    def _default_adaptive_search_config(self) -> Dict[str, Any]:
        return {
            "warmup_batches": 2,
            "update_frequency": 1,
            "top_fraction": 0.3,
            "min_candidates": 5,
            "margin_fraction": 0.2,
            "min_relative_span": 0.15,
            "include_experimental": True,
        }

    def _adaptive_search_settings(self) -> Dict[str, Any]:
        settings = self._default_adaptive_search_config()
        user_settings = getattr(self.config, "adaptive_search_config", None)
        if isinstance(user_settings, dict):
            settings.update(user_settings)
        return settings

    def _parameter_dicts_equal(self, first: Dict[str, Any], second: Dict[str, Any], tol: float = 1e-9) -> bool:
        keys = set(first.keys()) | set(second.keys())
        for key in keys:
            if key not in first or key not in second:
                return False
            left = first.get(key)
            right = second.get(key)
            if isinstance(left, (int, float, np.floating)) and isinstance(right, (int, float, np.floating)):
                if abs(float(left) - float(right)) > tol:
                    return False
            else:
                if left != right:
                    return False
        return True

    def _update_evaluator_selection_context(self, batch_idx: int, total_batches: int) -> None:
        if self.evaluator_type != EvaluatorType.VIRTUAL or self.evaluator is None:
            return
        if not hasattr(self.evaluator, "set_selection_context"):
            return

        self.evaluator.set_selection_context(
            current_batch=batch_idx,
            total_batches=total_batches,
            use_evolving_constraints=bool(getattr(self.config, "use_evolving_constraints", False)),
            constraints_config=getattr(self.config, "evolving_constraints_config", None),
        )

    def _update_adaptive_search_space(self, batch_idx: int) -> None:
        if not bool(getattr(self.config, "use_adaptive_search", False)):
            self.current_adaptive_bounds = {}
            return

        settings = self._adaptive_search_settings()
        warmup_batches = int(settings.get("warmup_batches", 2))
        update_frequency = max(1, int(settings.get("update_frequency", 1)))
        top_fraction = float(settings.get("top_fraction", 0.3))
        min_candidates = max(1, int(settings.get("min_candidates", 5)))
        margin_fraction = max(0.0, float(settings.get("margin_fraction", 0.2)))
        min_relative_span = max(1e-6, float(settings.get("min_relative_span", 0.15)))
        include_experimental = bool(settings.get("include_experimental", True))

        # Batch is zero-indexed internally; make warmup semantics one-indexed for users.
        batch_number = batch_idx + 1
        if batch_number <= warmup_batches:
            self.current_adaptive_bounds = {}
            return
        if ((batch_number - warmup_batches - 1) % update_frequency) != 0:
            return

        valid_candidates = []
        for candidate in self.all_candidates:
            if not include_experimental and candidate.get("is_experimental"):
                continue
            distance = candidate.get("distance")
            if distance is None:
                continue
            try:
                distance_value = float(distance)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(distance_value):
                continue
            params = candidate.get("parameters", {})
            if isinstance(params, dict) and params:
                valid_candidates.append((distance_value, params))

        if len(valid_candidates) < min_candidates:
            return

        valid_candidates.sort(key=lambda item: item[0])
        top_fraction = max(1e-6, min(top_fraction, 1.0))
        top_count = max(min_candidates, int(np.ceil(len(valid_candidates) * top_fraction)))
        top_candidates = [params for _, params in valid_candidates[:top_count]]
        new_bounds: Dict[str, Tuple[float, float]] = {}

        for param_cfg in self.config.parameters:
            param_type = param_cfg.type.value if hasattr(param_cfg.type, "value") else str(param_cfg.type)
            if param_type not in {"continuous", "discrete"} or param_cfg.bounds is None:
                continue

            original_low, original_high = float(param_cfg.bounds[0]), float(param_cfg.bounds[1])
            original_span = max(original_high - original_low, 1e-12)
            min_span = original_span * min_relative_span

            top_values = []
            for params in top_candidates:
                value = params.get(param_cfg.name)
                if isinstance(value, (int, float, np.floating)):
                    top_values.append(float(value))

            if not top_values:
                continue

            local_low = min(top_values)
            local_high = max(top_values)
            local_span = local_high - local_low
            margin = local_span * margin_fraction
            if local_span <= 1e-12:
                margin = max(margin, min_span / 2.0)

            candidate_low = max(original_low, local_low - margin)
            candidate_high = min(original_high, local_high + margin)

            if (candidate_high - candidate_low) < min_span:
                center = (candidate_low + candidate_high) / 2.0
                half_span = min_span / 2.0
                candidate_low = max(original_low, center - half_span)
                candidate_high = min(original_high, center + half_span)
                if (candidate_high - candidate_low) < min_span:
                    candidate_low = original_low
                    candidate_high = original_high

            new_bounds[param_cfg.name] = (candidate_low, candidate_high)

        if new_bounds:
            self.current_adaptive_bounds = new_bounds
            self.adaptive_bounds_history.append({
                "batch": batch_number,
                "bounds": {k: [float(v[0]), float(v[1])] for k, v in new_bounds.items()},
            })
            print(f"Adaptive search space updated at batch {batch_number}: {self.current_adaptive_bounds}")
        else:
            self.current_adaptive_bounds = {}

    def _apply_adaptive_search_space(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        if not self.current_adaptive_bounds:
            return parameters

        adjusted = dict(parameters)
        for param_cfg in self.config.parameters:
            bounds = self.current_adaptive_bounds.get(param_cfg.name)
            if bounds is None:
                continue
            value = adjusted.get(param_cfg.name)
            if not isinstance(value, (int, float, np.floating)):
                continue
            lower, upper = bounds
            adjusted[param_cfg.name] = min(max(float(value), lower), upper)
        return adjusted

    def _default_uncertainty_config(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "fallback_sem": 0.0,
            "min_sem": 0.0,
            "std_mode": "as_sem",  # "as_sem" | "std_to_sem"
            "default_replicates": 1,
            "replicates_column": "",
            "data_sem_suffixes": ["_sem", "_stderr", "_se"],
            "data_std_suffixes": ["_std", "_stdev", "_sigma"],
            "sdl_sem_keys": ["sem", "stderr", "se", "uncertainty"],
            "sdl_std_keys": ["std", "stdev", "sigma"],
            "virtual_sem_scale": 1.0,
        }

    def _uncertainty_settings(self) -> Dict[str, Any]:
        cfg = self._default_uncertainty_config()
        user_cfg = getattr(self.config, "uncertainty_config", None)
        if isinstance(user_cfg, dict):
            cfg.update(user_cfg)
        return cfg

    def _list_of_strings(self, value: Any, default: List[str]) -> List[str]:
        if isinstance(value, list):
            out = [str(v).strip() for v in value if str(v).strip()]
            return out or default
        if isinstance(value, str):
            out = [v.strip() for v in value.split(",") if v.strip()]
            return out or default
        return default

    def _sanitize_sem(self, sem_value: Optional[float]) -> float:
        settings = self._uncertainty_settings()
        if not bool(settings.get("enabled", True)):
            return 0.0
        fallback_sem = max(0.0, float(settings.get("fallback_sem", 0.0)))
        min_sem = max(0.0, float(settings.get("min_sem", 0.0)))
        if sem_value is None:
            return max(fallback_sem, min_sem)
        try:
            sem = float(sem_value)
        except (TypeError, ValueError):
            return max(fallback_sem, min_sem)
        if not np.isfinite(sem):
            return max(fallback_sem, min_sem)
        sem = abs(sem)
        return max(sem, min_sem)

    def _std_to_sem(self, std_value: Optional[float], replicates: Optional[float] = None) -> float:
        settings = self._uncertainty_settings()
        try:
            std = abs(float(std_value))
        except (TypeError, ValueError):
            return self._sanitize_sem(None)

        if not np.isfinite(std):
            return self._sanitize_sem(None)

        std_mode = str(settings.get("std_mode", "as_sem")).lower()
        if std_mode == "std_to_sem":
            default_reps = max(1, int(settings.get("default_replicates", 1)))
            if replicates is None:
                reps = default_reps
            else:
                try:
                    reps = int(max(1, round(float(replicates))))
                except (TypeError, ValueError):
                    reps = default_reps
            sem = std / np.sqrt(float(reps))
            return self._sanitize_sem(float(sem))

        return self._sanitize_sem(std)

    def _aggregate_objective_uncertainty(self, objective_uncertainties: Dict[str, float]) -> float:
        if not objective_uncertainties:
            return self._sanitize_sem(None)
        finite_vals = []
        for value in objective_uncertainties.values():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                finite_vals.append(abs(v))
        if not finite_vals:
            return self._sanitize_sem(None)
        return self._sanitize_sem(float(np.mean(finite_vals)))

    def _build_raw_data(
        self,
        distance: float,
        objective_outputs: Dict[str, Any],
        objective_uncertainties: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Tuple[float, float]]:
        uncertainties = objective_uncertainties or {}
        aggregate_sem = self._aggregate_objective_uncertainty(uncertainties)
        if self.uses_direct_objectives:
            raw_data: Dict[str, Tuple[float, float]] = {}
            for metric_name, metric_value in objective_outputs.items():
                try:
                    value = float(metric_value)
                except (TypeError, ValueError):
                    continue

                sem = uncertainties.get(metric_name)
                if sem is None and metric_name.endswith("_distance"):
                    base_name = metric_name[:-9]
                    sem = uncertainties.get(base_name)
                sem = aggregate_sem if sem is None else self._sanitize_sem(sem)
                raw_data[metric_name] = (value, sem)

            if raw_data:
                return raw_data

        return {"distance": (float(distance), aggregate_sem)}

    def _extract_uploaded_objective_uncertainties(self, row_idx: int) -> Dict[str, float]:
        if self.Y_sem_data is None or self.Y_sem_data.shape != self.Y_data.shape:
            return {}
        if row_idx < 0 or row_idx >= self.Y_sem_data.shape[0]:
            return {}

        out: Dict[str, float] = {}
        for j, obj in enumerate(self.config.objectives):
            value = self.Y_sem_data[row_idx, j]
            if np.isfinite(value):
                out[obj.name] = self._sanitize_sem(float(value))
        return out

    def _extract_sdl_measurements(
        self,
        response_payload: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
        if not isinstance(response_payload, dict):
            raise ValueError("SDL response is not a dictionary")

        measured_payload = response_payload.get("objectives", response_payload)
        if not isinstance(measured_payload, dict):
            raise ValueError("SDL response missing objective dictionary")

        uncertainty_payload = response_payload.get("objective_uncertainties")
        if not isinstance(uncertainty_payload, dict):
            uncertainty_payload = response_payload.get("uncertainties")
        if not isinstance(uncertainty_payload, dict):
            uncertainty_payload = {}

        settings = self._uncertainty_settings()
        sem_keys = [k.lower() for k in self._list_of_strings(settings.get("sdl_sem_keys"), ["sem", "stderr", "se", "uncertainty"])]
        std_keys = [k.lower() for k in self._list_of_strings(settings.get("sdl_std_keys"), ["std", "stdev", "sigma"])]
        uncertainty_type = str(response_payload.get("uncertainty_type", "sem")).lower()

        measured_values: Dict[str, float] = {}
        objective_uncertainties: Dict[str, float] = {}

        def parse_entry(value: Any) -> Tuple[Optional[float], Optional[float]]:
            if isinstance(value, (int, float, np.floating)):
                return float(value), None

            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return None, None
                first = value[0]
                second = value[1] if len(value) > 1 else None
                try:
                    mean_val = float(first)
                except (TypeError, ValueError):
                    mean_val = None
                if second is None:
                    return mean_val, None
                try:
                    spread = float(second)
                except (TypeError, ValueError):
                    spread = None
                if spread is None:
                    return mean_val, None
                if uncertainty_type in {"std", "stdev", "sigma"}:
                    return mean_val, self._std_to_sem(spread)
                return mean_val, self._sanitize_sem(spread)

            if isinstance(value, dict):
                mean_val = None
                for key in ("value", "mean", "mu", "prediction"):
                    if key in value:
                        try:
                            mean_val = float(value[key])
                            break
                        except (TypeError, ValueError):
                            continue

                sem_val = None
                lower_map = {str(k).lower(): v for k, v in value.items()}
                for key in sem_keys:
                    if key in lower_map:
                        try:
                            sem_val = self._sanitize_sem(float(lower_map[key]))
                            break
                        except (TypeError, ValueError):
                            continue
                if sem_val is None:
                    for key in std_keys:
                        if key in lower_map:
                            try:
                                sem_val = self._std_to_sem(float(lower_map[key]))
                                break
                            except (TypeError, ValueError):
                                continue
                if sem_val is None:
                    for key in ("variance", "var"):
                        if key in lower_map:
                            try:
                                variance = float(lower_map[key])
                                sem_val = self._sanitize_sem(np.sqrt(max(variance, 0.0)))
                                break
                            except (TypeError, ValueError):
                                continue
                return mean_val, sem_val

            return None, None

        for obj in self.config.objectives:
            obj_name = obj.name
            mean_val, sem_val = parse_entry(measured_payload.get(obj_name))
            if mean_val is not None:
                measured_values[obj_name] = mean_val
            if sem_val is not None:
                objective_uncertainties[obj_name] = sem_val

        if len(measured_values) < len(self.config.objectives):
            ordered_values = []
            ignored_keys = {
                "trial_index",
                "ts",
                "type",
                "status",
                "parameters",
                "control",
                "objective_uncertainties",
                "uncertainties",
                "candidate_modified",
                "observed_parameters",
            }
            for key in sorted(measured_payload.keys()):
                if key in ignored_keys:
                    continue
                mean_val, _ = parse_entry(measured_payload[key])
                if mean_val is not None:
                    ordered_values.append(mean_val)
            if len(ordered_values) == len(self.config.objectives):
                for idx, obj in enumerate(self.config.objectives):
                    measured_values[obj.name] = float(ordered_values[idx])

        for obj in self.config.objectives:
            obj_name = obj.name
            if obj_name in uncertainty_payload:
                raw_unc = uncertainty_payload[obj_name]
                sem_override = None
                if isinstance(raw_unc, (int, float, np.floating)):
                    if uncertainty_type in {"std", "stdev", "sigma"}:
                        sem_override = self._std_to_sem(float(raw_unc))
                    else:
                        sem_override = self._sanitize_sem(float(raw_unc))
                else:
                    parsed_mean, parsed_sem = parse_entry(raw_unc)
                    if parsed_sem is not None:
                        sem_override = parsed_sem
                    elif parsed_mean is not None:
                        if uncertainty_type in {"std", "stdev", "sigma"}:
                            sem_override = self._std_to_sem(parsed_mean)
                        else:
                            sem_override = self._sanitize_sem(parsed_mean)
                if sem_override is not None:
                    objective_uncertainties[obj_name] = sem_override

        measured_array = np.array([float(measured_values.get(obj.name, np.nan)) for obj in self.config.objectives])
        if np.isnan(measured_array).any():
            raise ValueError("SDL response missing objective values")

        objective_uncertainties = {
            obj.name: self._sanitize_sem(objective_uncertainties.get(obj.name))
            for obj in self.config.objectives
        }
        return measured_array, objective_uncertainties, measured_payload

    def _enforce_parameter_constraints(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce discrete and categorical constraints on generated parameters - FIXED FOR DECIMAL STEPS"""
        constrained_params = parameters.copy()
        
        for param_config in self.config.parameters:
            param_name = param_config.name
            if param_name in constrained_params:
                param_type = param_config.type.value if hasattr(param_config.type, 'value') else param_config.type
                
                if param_type == "discrete":
                    # Handle both integer and decimal steps
                    raw_value = constrained_params[param_name]
                    min_val, max_val = param_config.bounds
                    step = getattr(param_config, 'step', 1)
                    
                    # Calculate discrete value - works for both integer and decimal steps
                    if step > 0:
                        # Round to nearest step
                        steps_from_min = round((raw_value - min_val) / step)
                        discrete_value = min_val + steps_from_min * step
                        
                        # Clamp to bounds
                        discrete_value = max(min_val, min(max_val, discrete_value))
                        
                        # For display: keep as float for decimal steps, int for integer steps
                        if step == 1:
                            discrete_value = int(discrete_value)
                        else:
                            # Round to avoid floating point precision issues
                            decimal_places = len(str(step).split('.')[-1]) if '.' in str(step) else 0
                            discrete_value = round(discrete_value, decimal_places)
                        
                        constrained_params[param_name] = discrete_value
                        
                        # Debug output
                        print(f"  ðŸ”§ Discrete enforcement: {raw_value:.4f} -> {discrete_value} (step: {step})")
                    
                elif param_type == "categorical":
                    # Ensure categorical values are valid
                    raw_value = constrained_params[param_name]
                    if hasattr(param_config, 'categories') and param_config.categories:
                        # For continuous suggestions, map to nearest category
                        if isinstance(raw_value, (int, float)):
                            # Convert numerical suggestion to categorical
                            idx = int(round(raw_value * (len(param_config.categories) - 1)))
                            idx = max(0, min(len(param_config.categories) - 1, idx))
                            constrained_params[param_name] = param_config.categories[idx]
                        else:
                            # Ensure it's a valid category
                            if raw_value not in param_config.categories:
                                # Fallback to first category
                                constrained_params[param_name] = param_config.categories[0]
        
        return constrained_params

    def _initialize_with_experimental_data(self):
        """Initialize with any uploaded experimental data (if provided)."""
        print(f"?? Initializing with {len(self.X_data)} experimental data points...")

        if len(self.X_data) == 0 or len(self.Y_data) == 0:
            print("??  No uploaded experimental data to warm start with.")
            return

        successful_points = 0

        # Train evaluator only when we are using virtual evaluators
        if self.evaluator_type == EvaluatorType.VIRTUAL and self.evaluator is not None:
            print("?? Training evaluator on experimental data...")
            try:
                self.evaluator.fit(
                    self.X_data,
                    self.Y_data,
                    tune_hyperparams=self.config.enable_hyperparameter_tuning,
                    n_trials=self.config.n_tuning_trials if self.config.enable_hyperparameter_tuning else 0,
                    current_batch=0,
                    total_batches=self.config.batch_iterations,
                    use_evolving_constraints=self.config.use_evolving_constraints,
                    constraints_config=self.config.evolving_constraints_config,
                )
                print("? Evaluator trained successfully on experimental data")
            except Exception as e:
                print(f"??  Evaluator training failed: {e}")
        else:
            print("??  Skipping model training for SDL/simulator evaluator.")

        for i in range(len(self.X_data)):
            try:
                # Create parameters dictionary from experimental data - USE ORIGINAL VALUES
                params = {}
                for j, param_config in enumerate(self.config.parameters):
                    param_name = param_config.name
                    raw_value = float(self.X_data[i, j])
                    param_type = param_config.type.value if hasattr(param_config.type, 'value') else param_config.type

                    if param_type == "discrete":
                        min_bound, max_bound = param_config.bounds
                        step = getattr(param_config, 'step', 1)
                        if step > 0:
                            steps_from_min = round((raw_value - min_bound) / step)
                            discrete_value = min_bound + steps_from_min * step
                            discrete_value = max(min_bound, min(max_bound, discrete_value))
                            if step != 1:
                                decimal_places = len(str(step).split('.')[-1]) if '.' in str(step) else 0
                                discrete_value = round(discrete_value, decimal_places)
                            params[param_name] = discrete_value
                            print(f"  ?? Discrete experimental: {raw_value} -> {discrete_value}")
                    else:
                        params[param_name] = raw_value

                actual_objectives = self.Y_data[i]
                objective_outputs = {}
                try:
                    if self.evaluator is not None:
                        objective_outputs = self.evaluator.compute_objective_outputs(
                            actual_objectives,
                        )
                        if self.uses_direct_objectives:
                            distance = self.evaluator.calculate_objective_distance(
                                actual_objectives, include_direct_objectives=False
                            )
                        else:
                            distance = self.evaluator.calculate_objective_distance(actual_objectives)
                    else:
                        distance = float(np.linalg.norm(actual_objectives))
                    print(f"  Experimental point {i+1}: objectives = {actual_objectives}, distance = {distance:.6f}")
                except Exception as pred_error:
                    print(f"Warning: Distance calculation failed for point {i+1}: {pred_error}")
                    distance = 0.5
                    objective_outputs = {}

                if np.isinf(distance) or np.isnan(distance) or distance > 100:
                    distance = 1.0

                objective_uncertainties = self._extract_uploaded_objective_uncertainties(i)
                raw_data = self._build_raw_data(
                    distance=float(distance),
                    objective_outputs=objective_outputs,
                    objective_uncertainties=objective_uncertainties,
                )

                _, trial_index = self.ax_client.attach_trial(params)
                self.ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

                candidate_data = {
                    'parameters': params.copy(),
                    'predictions': actual_objectives.tolist() if hasattr(actual_objectives, 'tolist') else list(actual_objectives),
                    'distance': distance,
                    'objective_values': objective_outputs,
                    'uncertainties': objective_uncertainties,
                    'best_models_used': {} if self.evaluator is None else (self.evaluator.get_best_model_per_objective() if getattr(self.evaluator, "metrics", {}) else {}),
                    'is_experimental': True,
                    'trial_index': trial_index
                }
                self.all_candidates.append(candidate_data)

                if distance < self.best_overall_distance:
                    self.best_overall_distance = distance
                    self.best_overall_parameters = params.copy()
                    self.best_overall_predictions = np.array(actual_objectives)

                successful_points += 1
                print(f"  ? Added experimental point {i+1}: distance = {distance:.6f}")

            except Exception as e:
                print(f"??  Error adding experimental point {i+1}: {str(e)}")
                continue

        print(f"? Successfully initialized with {successful_points}/{len(self.X_data)} experimental data points")

        if successful_points > 0:
            exp_candidates = [c for c in self.all_candidates if c.get('is_experimental', False)]
            if exp_candidates:
                best_exp = min(exp_candidates, key=lambda x: x['distance'])
                self.best_overall_distance = best_exp['distance']
                self.best_overall_parameters = best_exp['parameters']
                self.best_overall_predictions = np.array(best_exp['predictions'])
                print(f"?? Best experimental distance: {self.best_overall_distance:.6f}")
        else:
            print("? No valid experimental data points could be added")
            self.best_overall_distance = float('inf')
    def _create_generation_strategy(self) -> AxGenerationStrategy:
        """Create generation strategy with cross-version Ax model resolution."""
        steps: List[GenerationStep] = []
        strategy = self.config.generation_strategy
        max_parallelism = (
            1 if self.config.optimization_mode == OptimizationMode.SEQUENTIAL
            else self.config.batch_size
        )

        no_initialization_warm_start = {
            BOGenerationStrategy.THOMPSON,
            BOGenerationStrategy.EMPIRICAL_BAYES_THOMPSON,
            BOGenerationStrategy.EB_ASHR,
            BOGenerationStrategy.FACTORIAL,
            BOGenerationStrategy.UNIFORM,
        }

        init_strategy = self._initialization_strategy_value()
        init_trials = self._initialization_trials_value()
        if init_strategy != BOInitializationStrategy.NONE.value and init_trials > 0:
            if strategy in no_initialization_warm_start:
                print(
                    f"Warning: Initialization step '{init_strategy}' ignored for "
                    f"{self._strategy_value()}."
                )
            else:
                if init_strategy == BOInitializationStrategy.SOBOL.value:
                    init_model, _ = self._resolve_ax_model(["SOBOL"], "SOBOL")
                    init_model_kwargs = {"seed": self.config.random_seed}
                elif init_strategy == BOInitializationStrategy.UNIFORM.value:
                    init_model, _ = self._resolve_ax_model(["UNIFORM"], "UNIFORM")
                    init_model_kwargs = {"seed": self.config.random_seed}
                else:
                    init_model, _ = self._resolve_ax_model(["SOBOL"], "SOBOL")
                    init_model_kwargs = {"seed": self.config.random_seed}

                steps.append(
                    GenerationStep(
                        model=init_model,
                        num_trials=init_trials,
                        min_trials_observed=1,
                        max_parallelism=max_parallelism,
                        model_kwargs=init_model_kwargs,
                        model_gen_kwargs={},
                    )
                )

        profile = self._search_space_profile()
        if strategy == BOGenerationStrategy.DEFAULT:
            if profile["has_categorical"] and not profile["has_continuous"]:
                model_candidates = ["BO_MIXED", "BOTORCH_MODULAR", "GPEI"]
            else:
                model_candidates = ["BOTORCH_MODULAR", "GPEI"]
        else:
            strategy_to_models = {
                BOGenerationStrategy.GPEI: ["GPEI", "BOTORCH_MODULAR"],
                BOGenerationStrategy.SAASBO: ["SAASBO", "FULLYBAYESIAN", "BOTORCH_MODULAR"],
                BOGenerationStrategy.FULLYBAYESIAN: ["FULLYBAYESIAN", "SAASBO", "BOTORCH_MODULAR"],
                BOGenerationStrategy.BOTORCH_MODULAR: ["BOTORCH_MODULAR", "GPEI"],
                BOGenerationStrategy.BO_MIXED: ["BO_MIXED", "BOTORCH_MODULAR", "GPEI"],
                BOGenerationStrategy.ST_MTGP: ["ST_MTGP", "ST_MTGP_NEHVI", "ST_MTGP_LEGACY"],
                BOGenerationStrategy.SAAS_MTGP: ["SAAS_MTGP", "FULLYBAYESIAN_MTGP", "ST_MTGP"],
                BOGenerationStrategy.THOMPSON: ["THOMPSON"],
                BOGenerationStrategy.EMPIRICAL_BAYES_THOMPSON: ["EMPIRICAL_BAYES_THOMPSON", "EB"],
                BOGenerationStrategy.EB_ASHR: ["EB_ASHR"],
                BOGenerationStrategy.FACTORIAL: ["FACTORIAL"],
                BOGenerationStrategy.UNIFORM: ["UNIFORM"],
            }
            model_candidates = strategy_to_models.get(strategy, ["BOTORCH_MODULAR", "GPEI"])

        main_model, main_model_name = self._resolve_ax_model(
            model_candidates,
            self._strategy_value(),
        )
        # Keep the strategy open-ended; total evaluations are controlled by the
        # run loop budget (batch_iterations * batch_size).
        main_num_trials = -1
        main_step_kwargs: Dict[str, Any] = {
            "model": main_model,
            "num_trials": main_num_trials,
            "max_parallelism": max_parallelism,
        }
        acqf_model_kwargs = self._build_acquisition_model_kwargs(main_model_name)
        if acqf_model_kwargs:
            main_step_kwargs["model_kwargs"] = acqf_model_kwargs
        steps.append(GenerationStep(**main_step_kwargs))

        print(
            f"Using generation strategy '{self._strategy_value()}' "
            f"with {len(steps)} step(s)."
        )
        return AxGenerationStrategy(steps=steps)
    def _initialize_ax_client(self) -> AxClient:
     """Initialize Ax client with parameter constraints."""

     generation_strategy = self._create_generation_strategy()

     ax_client = AxClient(
        generation_strategy=generation_strategy,
        random_seed=self.config.random_seed
    )

     task_parameter_name = self._get_task_parameter_name() if self._is_mtgp_strategy() else None

     parameters = []
     for param_config in self.config.parameters:
        param_type = param_config.type.value if hasattr(param_config.type, 'value') else param_config.type
        is_task = task_parameter_name is not None and param_config.name == task_parameter_name

        if param_type == "continuous":
            param_dict = {
                "name": param_config.name,
                "type": "range",
                "bounds": param_config.bounds,
                "value_type": "float"
            }
        elif param_type == "discrete":
            step = float(getattr(param_config, 'step', 1) or 1)
            low, high = param_config.bounds

            if is_task:
                values = []
                cursor = low
                while cursor <= high + 1e-9:
                    if abs(cursor - round(cursor)) < 1e-9:
                        values.append(int(round(cursor)))
                    else:
                        values.append(float(round(cursor, 8)))
                    cursor += step
                value_type = "int" if all(isinstance(v, int) for v in values) else "float"
                param_dict = {
                    "name": param_config.name,
                    "type": "choice",
                    "values": values,
                    "value_type": value_type,
                    "is_task": True,
                }
            elif step == 1:
                param_dict = {
                    "name": param_config.name,
                    "type": "range",
                    "bounds": [int(low), int(high)],
                    "value_type": "int",
                }
            else:
                param_dict = {
                    "name": param_config.name,
                    "type": "range",
                    "bounds": [float(low), float(high)],
                    "value_type": "float",
                }
        elif param_type == "categorical":
            param_dict = {
                "name": param_config.name,
                "type": "choice",
                "values": param_config.categories,
                "value_type": "str",
            }
            if is_task:
                param_dict["is_task"] = True
        else:
            raise ValueError(f"Unsupported parameter type for Ax: {param_type}")

        parameters.append(param_dict)

     # Convert parameter constraints to Ax format
     parameter_constraints = []
     for constraint in self.config.parameter_constraints:
        parameter_constraints.append(constraint.expression)
        print(f"Added parameter constraint: {constraint.expression}")

     if self.uses_direct_objectives:
        objectives = {}
        for obj in self.direct_objectives:
            obj_type = obj.type.value if hasattr(obj.type, 'value') else obj.type
            minimize_flag = (obj_type == 'minimize')
            objectives[obj.name] = ObjectiveProperties(minimize=minimize_flag)
        for obj in self.target_objectives:
            objectives[f"{obj.name}_distance"] = ObjectiveProperties(minimize=True)
     else:
        objectives = {"distance": ObjectiveProperties(minimize=True)}

     ax_client.create_experiment(
        name=self.config.experiment_name,
        parameters=parameters,
        objectives=objectives,
        parameter_constraints=parameter_constraints
    )

     return ax_client
    def _calculate_candidate_uncertainty(self, param_array: np.ndarray) -> Dict[str, float]:
        """Estimate per-objective uncertainty from objective-specific surrogate models."""
        try:
            if self.evaluator_type != EvaluatorType.VIRTUAL or self.evaluator is None or not getattr(self.evaluator, "models", {}):
                names = [obj.name if hasattr(obj, "name") else str(obj) for obj in self.config.objectives]
                return {n: self._sanitize_sem(None) for n in names}

            objective_names = list(getattr(self.evaluator, "objective_names", []))
            if not objective_names:
                objective_names = [obj.name for obj in self.config.objectives]
            model_map = getattr(self.evaluator, "models", {}) or {}
            scale = float(self._uncertainty_settings().get("virtual_sem_scale", 1.0))

            def _first_finite_scalar(value: Any) -> Optional[float]:
                try:
                    arr = np.asarray(value, dtype=np.float64).reshape(-1)
                except Exception:
                    return None
                for item in arr:
                    if np.isfinite(item):
                        return float(item)
                return None

            objective_uncertainties: Dict[str, float] = {}

            for obj_name in objective_names:
                suffix = f"_{obj_name}"
                objective_model_keys = [k for k in model_map.keys() if str(k).endswith(suffix)]
                if not objective_model_keys:
                    objective_uncertainties[obj_name] = self._sanitize_sem(None)
                    continue

                objective_predictions: List[float] = []
                weighted_uncertainty = 0.0
                total_weight = 0.0

                for model_key in objective_model_keys:
                    model = model_map.get(model_key)
                    if model is None:
                        continue

                    base_model_name = str(model_key)[:-len(suffix)]

                    try:
                        pred_raw = model.predict(param_array)
                        pred_value = _first_finite_scalar(pred_raw)
                        if pred_value is not None:
                            objective_predictions.append(pred_value)
                    except Exception as pred_err:
                        print(f"Warning: prediction failed for model {model_key}: {pred_err}")

                    try:
                        model_unc_raw = self.evaluator._get_single_model_uncertainty(model, param_array)
                    except Exception:
                        model_unc_raw = None
                    model_unc = self._sanitize_sem(_first_finite_scalar(model_unc_raw))

                    quality_score = 0.0
                    try:
                        if (
                            base_model_name in self.evaluator.metrics
                            and obj_name in self.evaluator.metrics[base_model_name]
                        ):
                            quality_score = float(
                                self.evaluator.metrics[base_model_name][obj_name].quality_score
                            )
                    except Exception:
                        quality_score = 0.0

                    weight = max(0.1, quality_score)
                    weighted_uncertainty += model_unc * weight
                    total_weight += weight

                avg_model_uncertainty = (
                    weighted_uncertainty / total_weight
                    if total_weight > 0.0
                    else self._sanitize_sem(None)
                )
                if len(objective_predictions) > 1:
                    ensemble_std = float(np.std(np.asarray(objective_predictions, dtype=np.float64)))
                    combined_uncertainty = 0.5 * ensemble_std + 0.5 * avg_model_uncertainty
                else:
                    combined_uncertainty = avg_model_uncertainty

                objective_uncertainties[obj_name] = self._sanitize_sem(
                    float(combined_uncertainty) * scale
                )

            for obj_name in objective_names:
                objective_uncertainties.setdefault(obj_name, self._sanitize_sem(None))
            return objective_uncertainties

        except Exception as e:
            print(f"Warning: uncertainty calculation failed: {e}")
            if self.evaluator is None:
                return {}
            return {obj_name: self._sanitize_sem(None) for obj_name in self.evaluator.objective_names}

    def _evaluate_candidate_with_uncertainty(
        self,
        parameters: Dict[str, Any],
        trial_index: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float], Dict[str, Any], Dict[str, float], np.ndarray, Dict[str, Any]]:
        """Evaluate a single candidate with proper parameter constraints and uncertainty."""
        try:
            proposed_parameters = dict(parameters)
            constrained_parameters = self._enforce_parameter_constraints(proposed_parameters)
            adaptive_parameters = self._apply_adaptive_search_space(constrained_parameters)
            effective_parameters = self._enforce_parameter_constraints(adaptive_parameters)
            param_array = np.array([[effective_parameters[p.name] for p in self.config.parameters]])

            constraints_adjusted = not self._parameter_dicts_equal(proposed_parameters, constrained_parameters)
            adaptive_adjusted = not self._parameter_dicts_equal(constrained_parameters, adaptive_parameters)

            print(
                "Parameter Enforcement: "
                f"{proposed_parameters} -> {constrained_parameters} -> {effective_parameters}"
            )

            # SDL path: send to hardware/remote and wait for measured objectives
            if self.evaluator_type == EvaluatorType.SELF_DRIVING_LAB and self.sdl_client is not None:
                response_payload = self.sdl_client.send_candidate_detailed(effective_parameters, trial_index=trial_index)
                if not isinstance(response_payload, dict):
                    raise ValueError("SDL response is not a dictionary")

                measured_array, uncertainties, measured_payload = self._extract_sdl_measurements(response_payload)

                observed_parameters = response_payload.get("observed_parameters")
                candidate_modified = bool(response_payload.get("candidate_modified")) and isinstance(observed_parameters, dict)

                if self.evaluator is not None:
                    objective_outputs = self.evaluator.compute_objective_outputs(
                        measured_array,
                    )
                    if self.uses_direct_objectives:
                        distance = self.evaluator.calculate_objective_distance(measured_array, include_direct_objectives=False)
                    else:
                        distance = self.evaluator.calculate_objective_distance(measured_array)
                else:
                    distance = float(np.linalg.norm(measured_array))
                    objective_outputs = {obj.name: measured_payload.get(obj.name) for obj in self.config.objectives}

                raw_data = self._build_raw_data(
                    distance=float(distance),
                    objective_outputs=objective_outputs,
                    objective_uncertainties=uncertainties,
                )
                metadata = {
                    "candidate_modified": candidate_modified,
                    "observed_parameters": observed_parameters if candidate_modified else None,
                    "response_payload": response_payload,
                    "measured_objectives": {
                        obj.name: float(measured_array[idx]) for idx, obj in enumerate(self.config.objectives)
                    },
                    "measurement_metadata": response_payload.get("measurement_metadata", {}),
                    "proposed_parameters": proposed_parameters,
                    "effective_parameters": effective_parameters,
                    "constraints_adjusted": constraints_adjusted,
                    "adaptive_adjusted": adaptive_adjusted,
                }
                return distance, uncertainties, raw_data, objective_outputs, measured_array, metadata

            # Virtual evaluator path
            predictions = self.evaluator.predict(param_array)
            prediction_for_distance = predictions[0] if predictions.ndim == 2 and predictions.shape[0] == 1 else predictions.flatten()
            objective_outputs = self.evaluator.compute_objective_outputs(
                prediction_for_distance,
            )
            uncertainties = self._calculate_candidate_uncertainty(param_array)

            if self.uses_direct_objectives:
                distance = self.evaluator.calculate_objective_distance(prediction_for_distance, include_direct_objectives=False)
            else:
                distance = self.evaluator.calculate_objective_distance(prediction_for_distance)
            if np.isinf(distance) or np.isnan(distance):
                print(f"Invalid distance calculated: {distance}")
                distance = float('inf')
            raw_data = self._build_raw_data(
                distance=float(distance),
                objective_outputs=objective_outputs,
                objective_uncertainties=uncertainties,
            )

            return distance, uncertainties, raw_data, objective_outputs, prediction_for_distance, {
                "candidate_modified": False,
                "observed_parameters": None,
                "measured_objectives": {
                    obj.name: float(prediction_for_distance[idx])
                    for idx, obj in enumerate(self.config.objectives)
                    if idx < len(prediction_for_distance)
                },
                "measurement_metadata": {},
                "proposed_parameters": proposed_parameters,
                "effective_parameters": effective_parameters,
                "constraints_adjusted": constraints_adjusted,
                "adaptive_adjusted": adaptive_adjusted,
            }

        except Exception as e:
            # Let caller decide how to mark trial; propagate for failure handling
            raise




    def run_optimization(self, progress_callback=None) -> OptimizationResult:
        """Run the main optimization loop with progress tracking (resumable)"""
        print(f"Starting Bayesian Optimization for {self.config.experiment_name}")
        print(f"Config: {len(self.config.parameters)} parameters, {len(self.config.objectives)} objectives")
        print(f"Settings: mode={self.config.optimization_mode.value}, batches={self.config.batch_iterations}, batch size {self.config.batch_size}")
        print(f"Generation Strategy: {self.config.generation_strategy.value}")
        print(
            f"Initialization: {self._initialization_strategy_value()} "
            f"({self._initialization_trials_value()} trials)"
        )
        print(f"Acquisition Function: {self._selected_acquisition_function().value}")
        print(f"Initial data points: {len(self.X_data)} experimental")
        print(f"Hyperparameter Tuning: {'Enabled' if self.config.enable_hyperparameter_tuning else 'Disabled'}")

        total_batches = self.config.batch_iterations if self.config.optimization_mode == OptimizationMode.BATCH else max(self.config.max_iterations, self.config.batch_iterations)
        start_batch = getattr(self, 'completed_batches', 0)
        if start_batch >= total_batches:
            print("No remaining batches to run; returning existing result")
            return self._create_final_result()

        print(f"Running {total_batches} {'batches' if self.config.optimization_mode == OptimizationMode.BATCH else 'sequential iterations'} (starting at batch {start_batch + 1})")

        for batch_idx in range(start_batch, total_batches):
            self._update_evaluator_selection_context(batch_idx, total_batches)
            self._update_adaptive_search_space(batch_idx)
            if progress_callback:
                progress_callback({
                    'batch': batch_idx + 1,
                    'total_batches': total_batches,
                    'status': 'starting_batch'
                })

            print("")
            print("=" * 50)
            print(f"BATCH {batch_idx + 1}/{total_batches}")
            print("=" * 50)

            batch_results = self._process_batch(batch_idx, progress_callback)
            self._store_batch_history(batch_idx, batch_results)
            self.completed_batches = batch_idx + 1

            print(f"Completed batch {batch_idx + 1}/{total_batches}")

        return self._create_final_result()



    def run_next_batch(self, progress_callback=None) -> Optional[OptimizationResult]:
        """Run exactly one batch and return a result if done (pause/resume support)."""
        total_batches = self.config.batch_iterations if self.config.optimization_mode == OptimizationMode.BATCH else max(self.config.max_iterations, self.config.batch_iterations)
        batch_idx = getattr(self, 'completed_batches', 0)

        if batch_idx >= total_batches:
            return self._create_final_result()

        if progress_callback:
            progress_callback({
                'batch': batch_idx + 1,
                'total_batches': total_batches,
                'status': 'starting_batch'
            })

        self._update_evaluator_selection_context(batch_idx, total_batches)
        self._update_adaptive_search_space(batch_idx)

        print("")
        print("=" * 50)
        print(f"BATCH {batch_idx + 1}/{total_batches}")
        print("=" * 50)

        batch_results = self._process_batch(batch_idx, progress_callback)
        self._store_batch_history(batch_idx, batch_results)
        self.completed_batches = batch_idx + 1

        print(f"Completed batch {batch_idx + 1}/{total_batches}")

        if self.completed_batches >= total_batches:
            return self._create_final_result()
        return None

    
    def _validate_prediction_consistency(self):
        """Validate that all predictions are in consistent format"""
        print("Validating prediction consistency...")
        if self.evaluator is None:
            print("No evaluator available for prediction consistency check.")
            return
        for i, candidate in enumerate(self.all_candidates):
            predictions = candidate['predictions']
            if isinstance(predictions, np.ndarray):
                print(f"Candidate {i}: numpy array shape {predictions.shape}")
            elif isinstance(predictions, list):
                print(f"Candidate {i}: list length {len(predictions)}")
            else:
                print(f"Candidate {i}: unknown type {type(predictions)}")
            expected_len = len(self.evaluator.objective_names)
            actual_len = len(predictions) if isinstance(predictions, list) else (predictions.shape[0] if hasattr(predictions, 'shape') else 0)
            if actual_len != expected_len:
                print(f"Warning: Candidate {i}: expected {expected_len} predictions, got {actual_len}")

    def _process_batch(self, batch_idx: int, progress_callback=None) -> List[Dict]:
        """Process a single batch of candidates"""
        batch_candidates = []
        batch_size = 1 if self.config.optimization_mode == OptimizationMode.SEQUENTIAL else self.config.batch_size
        print(f"Generating {batch_size} candidates for batch {batch_idx + 1}")
        for i in range(batch_size):
            try:
                parameters, trial_index = self.ax_client.get_next_trial()
                batch_candidates.append((parameters, trial_index))
            except Exception as e:
                print(f"Error generating candidate {i+1}: {e}")
                continue
        if not batch_candidates:
            print("No candidates generated in this batch")
            return []

        batch_results = []
        for parameters, trial_index in batch_candidates:
            try:
                distance, uncertainties, raw_data, objective_outputs, prediction_values, eval_meta = self._evaluate_candidate_with_uncertainty(parameters, trial_index=trial_index)
                proposed_parameters = eval_meta.get("proposed_parameters", parameters)
                evaluated_parameters = eval_meta.get("effective_parameters", proposed_parameters)
                constrained_parameters = evaluated_parameters
                effective_trial_index = trial_index
                proposed_trial_blocked = False
                parameters_adjusted = not self._parameter_dicts_equal(proposed_parameters, evaluated_parameters)
                constraints_adjusted = bool(eval_meta.get("constraints_adjusted", False))
                adaptive_adjusted = bool(eval_meta.get("adaptive_adjusted", False))
                measured_objectives = eval_meta.get("measured_objectives", {})
                measurement_metadata = eval_meta.get("measurement_metadata", {})

                if eval_meta.get("candidate_modified"):
                    observed_parameters = eval_meta.get("observed_parameters")
                    if not isinstance(observed_parameters, dict) or not observed_parameters:
                        raise ValueError("SDL marked candidate_modified but did not provide observed_parameters")

                    constrained_observed = self._enforce_parameter_constraints(observed_parameters)
                    print(f"Manual override detected. Proposed={proposed_parameters} Observed={constrained_observed}")

                    try:
                        self.ax_client.log_trial_failure(trial_index=trial_index)
                        proposed_trial_blocked = True
                    except Exception as log_err:
                        print(f"Warning: failed to mark proposed trial {trial_index} as failed: {log_err}")
                        proposed_trial_blocked = False

                    try:
                        _, observed_trial_index = self.ax_client.attach_trial(constrained_observed)
                        self.ax_client.complete_trial(trial_index=observed_trial_index, raw_data=raw_data)
                        constrained_parameters = constrained_observed
                        effective_trial_index = observed_trial_index
                    except Exception as attach_err:
                        print(f"Warning: failed to attach observed trial ({attach_err}); completing original trial instead")
                        self.ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                        constrained_parameters = proposed_parameters
                        effective_trial_index = trial_index
                        proposed_trial_blocked = False
                elif parameters_adjusted:
                    print(
                        "Candidate adjusted before evaluation. "
                        f"Proposed={proposed_parameters}, Adjusted={evaluated_parameters}"
                    )
                    try:
                        self.ax_client.log_trial_failure(trial_index=trial_index)
                        proposed_trial_blocked = True
                    except Exception as log_err:
                        print(f"Warning: failed to mark proposed trial {trial_index} as failed: {log_err}")
                        proposed_trial_blocked = False
                    try:
                        _, adjusted_trial_index = self.ax_client.attach_trial(evaluated_parameters)
                        self.ax_client.complete_trial(trial_index=adjusted_trial_index, raw_data=raw_data)
                        constrained_parameters = evaluated_parameters
                        effective_trial_index = adjusted_trial_index
                    except Exception as attach_err:
                        print(f"Warning: failed to attach adjusted trial ({attach_err}); completing original trial instead")
                        self.ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                        constrained_parameters = proposed_parameters
                        effective_trial_index = trial_index
                        proposed_trial_blocked = False
                else:
                    self.ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

                if isinstance(prediction_values, np.ndarray):
                    predictions_list = prediction_values.flatten().tolist() if prediction_values.ndim > 0 else [float(prediction_values)]
                else:
                    try:
                        predictions_list = list(prediction_values)
                    except Exception:
                        predictions_list = []
                objective_count = len(self.config.objectives)
                if len(predictions_list) < objective_count:
                    predictions_list.extend([0.0] * (objective_count - len(predictions_list)))

                best_models = {}
                if self.evaluator_type == EvaluatorType.VIRTUAL and self.evaluator is not None and getattr(self.evaluator, "metrics", {}):
                    best_models = self.evaluator.get_best_model_per_objective()

                candidate_data = {
                    'parameters': constrained_parameters,
                    'predictions': predictions_list,
                    'distance': distance,
                    'objective_values': objective_outputs,
                    'uncertainties': uncertainties,
                    'measured_objectives': measured_objectives,
                    'measurement_metadata': measurement_metadata,
                    'best_models_used': best_models,
                    'trial_index': effective_trial_index,
                    'source_trial_index': trial_index,
                    'proposed_trial_blocked': proposed_trial_blocked,
                    'candidate_modified': bool(eval_meta.get("candidate_modified")),
                    'parameters_adjusted': parameters_adjusted,
                    'constraints_adjusted': constraints_adjusted,
                    'adaptive_adjusted': adaptive_adjusted,
                    'proposed_parameters': proposed_parameters if parameters_adjusted or bool(eval_meta.get("candidate_modified")) else None,
                    'is_experimental': False
                }
                self.all_candidates.append(candidate_data)

                if distance < self.best_overall_distance:
                    self.best_overall_distance = distance
                    self.best_overall_parameters = constrained_parameters.copy()
                    self.best_overall_predictions = np.array(predictions_list)
                    print(f"NEW BEST: distance = {distance:.6f}")

                batch_results.append({
                    'parameters': constrained_parameters,
                    'distance': distance,
                    'uncertainties': uncertainties,
                    'measured_objectives': measured_objectives,
                    'measurement_metadata': measurement_metadata,
                    'trial_index': effective_trial_index,
                    'source_trial_index': trial_index,
                    'proposed_trial_blocked': proposed_trial_blocked,
                    'objective_values': objective_outputs
                })

                if progress_callback:
                    progress_callback({
                        'status': 'completed_candidate',
                        'distance': distance,
                        'parameters': constrained_parameters,
                        'uncertainties': uncertainties,
                        'measured_objectives': measured_objectives,
                        'measurement_metadata': measurement_metadata,
                        'trial_index': effective_trial_index,
                        'source_trial_index': trial_index,
                        'proposed_trial_blocked': proposed_trial_blocked,
                        'best_distance': self.best_overall_distance,
                        'objective_values': objective_outputs,
                        'best_models_used': best_models,
                    })
            except Exception as e:
                print(f"Error evaluating candidate {trial_index}: {e}")
                try:
                    # Mark trial failed in Ax so it doesn't expect metrics
                    self.ax_client.log_trial_failure(trial_index=trial_index)
                except Exception as log_err:
                    print(f"Warning: failed to log trial failure for {trial_index}: {log_err}")
                if progress_callback:
                    progress_callback({
                        'status': 'failed_candidate',
                        'trial_index': trial_index,
                        'error': str(e)
                    })
                continue
        return batch_results

    def _store_batch_history(self, batch_idx: int, batch_results: List[Dict]):
        """Store batch history information"""
        batch_info = {
            'batch': batch_idx + 1,
            'batch_min_distance': float('inf'),
            'batch_mean_distance': float('inf'),
            'best_overall_distance': self.best_overall_distance,
            'n_candidates_evaluated': len(self.all_candidates),
            'best_models': self.evaluator.get_best_model_per_objective() if (self.evaluator_type == EvaluatorType.VIRTUAL and self.evaluator is not None and getattr(self.evaluator, "metrics", {})) else {},
            'generation_strategy': self.config.generation_strategy.value,
            'adaptive_bounds': {
                k: [float(v[0]), float(v[1])]
                for k, v in getattr(self, 'current_adaptive_bounds', {}).items()
            }
        }
        if batch_results:
            batch_distances = [r['distance'] for r in batch_results]
            batch_info['batch_min_distance'] = np.min(batch_distances) if batch_distances else float('inf')
            batch_info['batch_mean_distance'] = np.mean(batch_distances) if batch_distances else float('inf')
            print(f"Batch {batch_idx + 1} Summary:")
            print(f"   Best in batch: {batch_info['batch_min_distance']:.6f}")
            print(f"   Best overall: {self.best_overall_distance:.6f}")
            print(f"   Total candidates evaluated: {len(self.all_candidates)}")
            print(f"   Batch size: {len(batch_results)}")
        else:
            print(f"Batch {batch_idx + 1} Summary: No candidates evaluated in this batch")
            print(f"   Best overall: {self.best_overall_distance:.6f}")
            print(f"   Total candidates evaluated: {len(self.all_candidates)}")
        self.history.append(batch_info)


    def _create_final_result(self) -> OptimizationResult:
        """Create final optimization result with robust Pareto handling"""
        def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            """Return True if point a Pareto-dominates point b."""
            better_or_equal = True
            strictly_better = False
            for obj in self.direct_objectives:
                direction = obj.type.value if hasattr(obj.type, "value") else obj.type
                name = obj.name
                if name not in a['outcomes'] or name not in b['outcomes']:
                    return False
                a_val = a['outcomes'][name]
                b_val = b['outcomes'][name]
                if direction == "minimize":
                    if a_val > b_val:
                        better_or_equal = False
                    if a_val < b_val:
                        strictly_better = True
                else:  # maximize
                    if a_val < b_val:
                        better_or_equal = False
                    if a_val > b_val:
                        strictly_better = True
            return better_or_equal and strictly_better

        pareto_candidates: List[Dict[str, Any]] = []
        if self.uses_direct_objectives:
            for cand in self.all_candidates:
                obj_values = cand.get('objective_values') or {}
                outcomes = {}
                missing = False
                for obj in self.direct_objectives:
                    if obj.name not in obj_values:
                        missing = True
                        break
                    try:
                        outcomes[obj.name] = float(obj_values[obj.name])
                    except Exception:
                        outcomes[obj.name] = obj_values[obj.name]
                if missing or not outcomes:
                    continue
                pareto_candidates.append({
                    'parameters': cand.get('parameters', {}),
                    'outcomes': outcomes
                })

        pareto_front: List[Dict[str, Any]] = []
        if pareto_candidates:
            for i, p in enumerate(pareto_candidates):
                dominated = False
                for j, q in enumerate(pareto_candidates):
                    if i == j:
                        continue
                    if _dominates(q, p):
                        dominated = True
                        break
                if not dominated:
                    pareto_front.append(p)

        best_predictions = self.best_overall_predictions
        if (best_predictions is None or len(best_predictions) == 0) and self.best_overall_parameters:
            for cand in reversed(self.all_candidates):
                if cand.get('parameters') == self.best_overall_parameters:
                    best_predictions = np.array(cand.get('predictions', []))
                    break

        model_perf = {
            'generation_strategy': self.config.generation_strategy.value if hasattr(self.config.generation_strategy, 'value') else self.config.generation_strategy,
            'acquisition_function': self._selected_acquisition_function().value,
            'initialization_strategy': self._initialization_strategy_value(),
            'initialization_trials': self._initialization_trials_value(),
            'adaptive_search_enabled': bool(getattr(self.config, 'use_adaptive_search', False)),
            'adaptive_search_config': getattr(self.config, 'adaptive_search_config', None),
            'evolving_constraints_enabled': bool(getattr(self.config, 'use_evolving_constraints', False)),
            'evolving_constraints_config': getattr(self.config, 'evolving_constraints_config', None),
            'uncertainty_config': getattr(self.config, 'uncertainty_config', None),
            'completed_batches': getattr(self, 'completed_batches', len(self.history)),
            'total_batches': self.config.batch_iterations
        }

        if self.uses_direct_objectives and not self.best_overall_parameters and pareto_front:
            self.best_overall_parameters = pareto_front[0].get('parameters', {})
            self.best_overall_distance = 0.0
            if best_predictions is None or len(best_predictions) == 0:
                best_predictions = np.array([])

        return OptimizationResult(
            best_parameters=self.best_overall_parameters or {},
            best_predictions=best_predictions if best_predictions is not None else np.array([]),
            best_distance=float(self.best_overall_distance),
            history=list(self.history),
            all_candidates=list(self.all_candidates),
            model_performance=model_perf,
            pareto_front=pareto_front,
            uses_direct_objectives=self.uses_direct_objectives
        )

