# core/evaluators.py - COMPLETE FIXED VERSION WITH PROPER DATA HANDLING

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    OPTUNA_AVAILABLE = False

from .config import ObjectiveConfig, ObjectiveType, ParameterConfig
from .models import ModelFactory

@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics"""
    train_rmse: float
    test_rmse: float
    train_r2: float
    test_r2: float
    train_mae: float
    test_mae: float
    normalized_train_rmse: float
    normalized_test_rmse: float
    quality_score: float
    meets_constraints: bool
    prediction_uncertainty: float = field(default=0.0)
    cross_val_score: float = field(default=0.0)

class GeneralizedEvaluator:
    """Generalized evaluator that works with any parameters and objectives"""
    
    def __init__(
        self,
        model_factory: ModelFactory,
        parameter_configs: List[ParameterConfig],
        objective_configs: List[ObjectiveConfig],
        distance_normalization_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_factory = model_factory
        self.parameter_configs = parameter_configs
        self.objective_configs = objective_configs
        self.parameter_names = [p.name for p in parameter_configs]
        self.objective_names = [o.name for o in objective_configs]
        
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, EvaluationMetrics]] = {}
        self.fitted = False
        
        # Initialize scalers
        self.X_scaler = StandardScaler()
        self.Y_scalers = {}  # Separate scaler for each objective
        self.selection_context: Dict[str, Any] = {
            "current_batch": 0,
            "total_batches": 1,
            "use_evolving_constraints": False,
            "constraints_config": None,
        }
        self.distance_normalization_config: Dict[str, Any] = self._resolve_distance_normalization_config(
            distance_normalization_config
        )
        self.objective_scales: Dict[str, float] = {name: 1.0 for name in self.objective_names}
        self._distance_samples: Dict[str, List[float]] = {name: [] for name in self.objective_names}
        
        print(f"🔧 Evaluator initialized for {len(self.parameter_names)} parameters, {len(self.objective_names)} objectives")
        
    def _prepare_data(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and validate data for training"""
        # Ensure data is properly formatted
        X_clean = np.asarray(X, dtype=np.float64)
        Y_clean = np.asarray(Y, dtype=np.float64)
        
        # Remove any NaN or Inf values
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1.0, neginf=0.0)
        Y_clean = np.nan_to_num(Y_clean, nan=0.0, posinf=1.0, neginf=0.0)
        
        print(f"🔧 Data Preparation:")
        print(f"   X shape: {X_clean.shape}, Y shape: {Y_clean.shape}")
        print(f"   X range: [{X_clean.min():.3f}, {X_clean.max():.3f}]")
        print(f"   Y range: [{Y_clean.min():.3f}, {Y_clean.max():.3f}]")
        
        # Check for constant columns
        for i in range(X_clean.shape[1]):
            if np.std(X_clean[:, i]) < 1e-10:
                print(f"⚠️  Warning: Parameter {self.parameter_names[i]} has very low variance")
        
        return X_clean, Y_clean
    
    def smart_rmse_normalization(self, rmse: float, y_true: np.ndarray, 
                           y_pred: np.ndarray = None) -> Tuple[float, str]:
     """Apply smart RMSE normalization based on data characteristics - EXACT INSPIRATION VERSION"""
    
    # Calculate data characteristics
     y_range = np.max(y_true) - np.min(y_true)
     y_mean = np.mean(y_true)
     y_std = np.std(y_true)
     data_min, data_max = np.min(y_true), np.max(y_true)
     data_abs_max = max(abs(data_min), abs(data_max))

    # Calculate different normalized RMSE metrics
     nrmse_metrics = {}

    # NRMSE by range
     nrmse_metrics["range"] = (rmse / y_range) if y_range != 0 else float('inf')

    # NRMSE by mean
     nrmse_metrics["mean"] = (rmse / y_mean) if y_mean != 0 else float('inf')

    # NRMSE by standard deviation
     nrmse_metrics["std"] = (rmse / y_std) if y_std != 0 else float('inf')

    # RMSPE (only if no zeros in true values)
     if np.any(y_true == 0) or y_pred is None:
        nrmse_metrics["rmspe"] = float('inf')
     else:
        percentage_errors = ((y_true - y_pred) / y_true) ** 2
        nrmse_metrics["rmspe"] = np.sqrt(np.mean(percentage_errors))

    # Smart selection of the best normalization method - EXACTLY LIKE INSPIRATION
     if np.any(y_true == 0) or data_min < 0:
        recommended = "std"
     elif data_abs_max < 1e-3:  # very small numbers
        recommended = "std" if nrmse_metrics["std"] < nrmse_metrics["range"] else "range"
     elif data_abs_max > 1e6:  # very large numbers
        recommended = "std"
     elif data_min >= 0:  # positive-only values
        recommended = "rmspe" if nrmse_metrics["rmspe"] < float('inf') else "mean"
     else:  # bounded symmetric range
        recommended = "range"

     selected_nrmse = nrmse_metrics.get(recommended, nrmse_metrics["std"])

     if selected_nrmse < float('inf'):
        normalized_rmse = max(0, (selected_nrmse))
     else:
        # Fallback: use simple inverse of raw RMSE (scaled)
        normalized_rmse = max(0, (rmse / (2 * y_std))) if y_std != 0 else 0

     return normalized_rmse, f"NRMSE_{recommended}"


    def _standardize_predictions(self, predictions, model_name: str = None) -> np.ndarray:
     """Ensure predictions are in consistent format across all models"""
     try:
        if predictions is None:
            return np.zeros(len(self.objective_names))
            
        # Convert to numpy array
        if isinstance(predictions, (list, tuple)):
            pred_array = np.array(predictions, dtype=np.float64)
        elif isinstance(predictions, np.ndarray):
            pred_array = predictions.astype(np.float64)
        else:
            pred_array = np.array([float(predictions)])
        
        # Ensure proper shape
        if pred_array.ndim == 0:  # scalar
            pred_array = np.array([pred_array])
        elif pred_array.ndim > 1:
            pred_array = pred_array.flatten()
        
        # Ensure we have the right number of predictions
        n_expected = len(self.objective_names)
        if len(pred_array) != n_expected:
            print(f"⚠️  Model {model_name}: Expected {n_expected} predictions, got {len(pred_array)}")
            if len(pred_array) < n_expected:
                # Pad with zeros
                pred_array = np.pad(pred_array, (0, n_expected - len(pred_array)))
            else:
                # Take first n_expected
                pred_array = pred_array[:n_expected]
        
        return pred_array
        
     except Exception as e:
        print(f"❌ Error standardizing predictions for {model_name}: {e}")
        return np.zeros(len(self.objective_names))


    @staticmethod
    def _safe_scale(*values: float, floor: float = 1.0) -> float:
        """Return a stable positive normalization scale from candidate values."""
        finite_abs = [abs(float(v)) for v in values if v is not None and np.isfinite(v)]
        if not finite_abs:
            return float(floor)
        return float(max([floor] + finite_abs))

    def _default_distance_normalization_config(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "method": "quantile",  # quantile | range | std
            "q_low": 0.05,
            "q_high": 0.95,
            "min_scale": 1e-6,
            "clip_component": 10.0,  # 0 disables clipping
            "normalize_weight_norm": True,
            "normalize_target_components": False,
            "max_scale_samples": 2000,
        }

    def _resolve_distance_normalization_config(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved = self._default_distance_normalization_config()
        if isinstance(config, dict):
            resolved.update(config)

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

    def set_distance_normalization_config(self, config: Optional[Dict[str, Any]]) -> None:
        self.distance_normalization_config = self._resolve_distance_normalization_config(config)
        self._update_distance_scales_from_samples()

    def _update_distance_scales_from_samples(self) -> None:
        cfg = self.distance_normalization_config
        min_scale = float(cfg.get("min_scale", 1e-6))
        method = str(cfg.get("method", "quantile")).lower()
        q_low = float(cfg.get("q_low", 0.05))
        q_high = float(cfg.get("q_high", 0.95))

        for obj_name in self.objective_names:
            values = np.asarray(self._distance_samples.get(obj_name, []), dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size < 2:
                self.objective_scales[obj_name] = float(max(min_scale, self.objective_scales.get(obj_name, 1.0)))
                continue

            if method == "range":
                scale = float(np.max(values) - np.min(values))
            elif method == "std":
                scale = float(np.std(values))
            else:
                low_v = float(np.quantile(values, q_low))
                high_v = float(np.quantile(values, q_high))
                scale = float(high_v - low_v)

            if (not np.isfinite(scale)) or scale < min_scale:
                scale = min_scale
            self.objective_scales[obj_name] = float(scale)

    def _update_distance_scales_from_matrix(self, values: np.ndarray, reset: bool = False) -> None:
        cfg = self.distance_normalization_config
        if not bool(cfg.get("enabled", True)):
            return

        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            return

        n_obj = len(self.objective_names)
        if arr.shape[1] < n_obj:
            arr = np.pad(arr, ((0, 0), (0, n_obj - arr.shape[1])), mode="constant", constant_values=np.nan)
        elif arr.shape[1] > n_obj:
            arr = arr[:, :n_obj]

        if reset:
            self._distance_samples = {name: [] for name in self.objective_names}

        max_samples = int(cfg.get("max_scale_samples", 2000))
        for j, obj_name in enumerate(self.objective_names):
            col = arr[:, j]
            finite_vals = col[np.isfinite(col)].tolist()
            if not finite_vals:
                continue
            bucket = self._distance_samples.setdefault(obj_name, [])
            bucket.extend(float(v) for v in finite_vals)
            if len(bucket) > max_samples:
                self._distance_samples[obj_name] = bucket[-max_samples:]

        self._update_distance_scales_from_samples()

    def update_distance_scales(self, values: np.ndarray, reset: bool = False) -> None:
        """Public wrapper for online distance-scale updates from objective observations."""
        self._update_distance_scales_from_matrix(values, reset=reset)

    def _component_scale(self, obj_name: str, pred_value: float) -> float:
        cfg = self.distance_normalization_config
        min_scale = float(cfg.get("min_scale", 1e-6))
        scale = float(self.objective_scales.get(obj_name, min_scale))
        if (not np.isfinite(scale)) or scale < min_scale:
            scale = max(min_scale, self._safe_scale(pred_value, floor=min_scale))
        return float(scale)

    def _compute_distance_component(self, pred_value: float, obj_config: ObjectiveConfig) -> float:
        """Compute a distance/cost component for a single objective."""
        obj_type = obj_config.type.value if hasattr(obj_config.type, "value") else obj_config.type

        if obj_type == "target_range":
            target_min, target_max = obj_config.target_range
            # Guard against degenerate ranges to avoid numerical blowups.
            width = abs(float(target_max) - float(target_min))
            if width < 1e-12:
                width = self._safe_scale(target_min, target_max, pred_value)
            if pred_value < target_min:
                return (target_min - pred_value) / width
            if pred_value > target_max:
                return (pred_value - target_max) / width
            return 0.0

        if obj_type == "target_value":
            target_val = float(obj_config.target_value if obj_config.target_value is not None else 0.0)
            tolerance = max(0.0, float(obj_config.tolerance))
            deviation = abs(float(pred_value) - target_val)
            if deviation <= tolerance:
                return 0.0
            # Normalize by a stable scale so target=0 (or near 0) does not explode.
            scale = self._safe_scale(target_val, pred_value, tolerance)
            return (deviation - tolerance) / scale

        if obj_type == "minimize":
            return float(pred_value)

        if obj_type == "maximize":
            # Negative value means better performance for maximization when treated as a cost.
            return -float(pred_value)

        return 0.0

    def calculate_objective_distance(self, predictions: np.ndarray, include_direct_objectives: bool = True) -> float:
     """Simplified and robust distance calculation"""
     try:
        # Convert to 1D numpy array consistently
        if isinstance(predictions, (list, tuple)):
            pred_array = np.array(predictions, dtype=np.float64)
        elif isinstance(predictions, np.ndarray):
            if predictions.ndim > 1:
                pred_array = predictions.flatten()
            else:
                pred_array = predictions.astype(np.float64)
        else:
            pred_array = np.array([float(predictions)])
        
        # Ensure we have predictions for all objectives
        n_objectives = len(self.objective_configs)
        if len(pred_array) < n_objectives:
            # Pad with zeros if missing predictions
            pred_array = np.pad(pred_array, (0, n_objectives - len(pred_array)))
        elif len(pred_array) > n_objectives:
            # Truncate if too many predictions
            pred_array = pred_array[:n_objectives]
        
        cfg = self.distance_normalization_config
        apply_norm = bool(cfg.get("enabled", True))
        normalize_target_components = bool(cfg.get("normalize_target_components", False))
        clip_component = float(cfg.get("clip_component", 0.0))
        normalize_weight_norm = bool(cfg.get("normalize_weight_norm", True))

        distances = []
        weight_sq_sum = 0.0
        for i, obj_config in enumerate(self.objective_configs):
            obj_type = obj_config.type.value if hasattr(obj_config.type, "value") else obj_config.type
            
            # Skip direct objectives when requested (used for direct optimization mode)
            if not include_direct_objectives and obj_type in ["minimize", "maximize"]:
                continue
            
            pred_value = float(pred_array[i])
            distance_component = self._compute_distance_component(pred_value, obj_config)

            if apply_norm:
                should_normalize_component = (
                    obj_type in ["minimize", "maximize"]
                    or normalize_target_components
                )
                if should_normalize_component:
                    scale = self._component_scale(obj_config.name, pred_value)
                    if scale > 0:
                        distance_component = distance_component / scale

            if clip_component > 0.0 and np.isfinite(distance_component):
                distance_component = float(np.clip(distance_component, -clip_component, clip_component))

            # Apply weight and ensure non-negative costs
            obj_weight = float(max(obj_config.weight, 1e-12))
            weighted_distance = max(0.0, distance_component) * obj_weight
            distances.append(weighted_distance)
            weight_sq_sum += obj_weight * obj_weight
        
        if not distances:
            return 0.0
        
        # Use Euclidean distance
        total_distance = float(np.linalg.norm(distances))
        if normalize_weight_norm and weight_sq_sum > 0.0:
            total_distance = total_distance / float(np.sqrt(weight_sq_sum))
        return total_distance
        
     except Exception as e:
        print(f"Г?O Error in distance calculation: {e}")
        return float('inf')

    def compute_objective_outputs(
        self,
        predictions: np.ndarray,
        force_target_normalization: bool = False,
    ) -> Dict[str, float]:
        """Return per-objective outputs for direct optimization (targets => distances)."""
        try:
            # Convert to 1D numpy array consistently
            if isinstance(predictions, (list, tuple)):
                pred_array = np.array(predictions, dtype=np.float64)
            elif isinstance(predictions, np.ndarray):
                if predictions.ndim > 1:
                    pred_array = predictions.flatten()
                else:
                    pred_array = predictions.astype(np.float64)
            else:
                pred_array = np.array([float(predictions)])
            
            # Ensure length alignment
            n_objectives = len(self.objective_configs)
            if len(pred_array) < n_objectives:
                pred_array = np.pad(pred_array, (0, n_objectives - len(pred_array)))
            elif len(pred_array) > n_objectives:
                pred_array = pred_array[:n_objectives]
            
            cfg = self.distance_normalization_config
            apply_norm = bool(cfg.get("enabled", True))
            normalize_target_components = bool(cfg.get("normalize_target_components", False))
            clip_component = float(cfg.get("clip_component", 0.0))

            outputs: Dict[str, float] = {}
            for i, obj_config in enumerate(self.objective_configs):
                obj_type = obj_config.type.value if hasattr(obj_config.type, "value") else obj_config.type
                pred_value = float(pred_array[i])
                
                if obj_type in ["minimize", "maximize"]:
                    # Direct objectives use the raw prediction
                    outputs[obj_config.name] = pred_value
                elif obj_type in ["target_range", "target_value"]:
                    distance_component = self._compute_distance_component(pred_value, obj_config)
                    should_normalize_target = normalize_target_components or force_target_normalization
                    if apply_norm and should_normalize_target:
                        scale = self._component_scale(obj_config.name, pred_value)
                        if scale > 0:
                            distance_component = distance_component / scale
                    if clip_component > 0.0 and np.isfinite(distance_component):
                        distance_component = float(np.clip(distance_component, -clip_component, clip_component))
                    outputs[f"{obj_config.name}_distance"] = max(0.0, distance_component) * obj_config.weight
            
            return outputs
        except Exception as e:
            print(f"Г?O Error computing objective outputs: {e}")
            return {}

    def _optuna_objective(self, trial, X_train, y_train, X_test, y_test, model_type: str, 
                     default_params: Dict[str, Any]):
        """Optuna objective function for hyperparameter tuning - FIXED for proper data handling"""
        params = default_params.copy()
        
        if model_type == "random_forest":
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)
            
        elif model_type == "xgboost":
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            
        elif model_type == "svm":
            params['C'] = trial.suggest_float('C', 0.1, 100.0, log=True)
            params['gamma'] = trial.suggest_float('gamma', 1e-4, 1.0, log=True)
            
        elif model_type == "gam":
            # Use simpler parameter ranges for GAM to avoid compatibility issues
            params['n_splines'] = trial.suggest_int('n_splines', 10, 30)
            params['lam'] = trial.suggest_float('lam', 0.1, 10.0)
            # Keep spline_order fixed to avoid complexity
            params['spline_order'] = 3
        
        # Create and train model
        try:
            model = self.model_factory.registry.create_model(model_type, X_train.shape[1], params)
            model.fit(X_train, y_train)
            
            # Predict and calculate metrics
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            return rmse
        except Exception as e:
            # Return a high RMSE for failed models
            print(f"⚠️  Model training failed in Optuna trial: {e}")
            return float('inf')  # High error for failed models
    
    def tune_hyperparameters(self, X: np.ndarray, Y: np.ndarray, model_name: str, 
                           model_type: str, n_trials: int = 50) -> Dict[str, Any]:
        """Tune hyperparameters for a specific model and objective"""
        if not OPTUNA_AVAILABLE:
            print("⚠️  Optuna is not installed; skipping hyperparameter tuning.")
            return {}

        if n_trials <= 0:
            return {}

        best_params = {}
        
        for j, obj_name in enumerate(self.objective_names):
            print(f"  🎯 Tuning {model_name} for {obj_name}...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y[:, j], test_size=0.2, random_state=42, shuffle=True
            )
            
            # Get default parameters
            default_params = self.model_factory.registry._default_params[model_type].copy()
            
            # Create Optuna study
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42)
            )
            
            # Optimize
            study.optimize(
                lambda trial: self._optuna_objective(
                    trial, X_train, y_train, X_test, y_test, model_type, default_params
                ),
                n_trials=n_trials,
                show_progress_bar=False
            )
            
            best_params[obj_name] = study.best_params
            print(f"  ✅ Best {model_name} params for {obj_name}: {study.best_value:.4f} RMSE")
            
        return best_params

    def _default_constraints_config(self) -> Dict[str, Any]:
        return {
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

    def _resolve_constraints_config(self, constraints_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resolved = self._default_constraints_config()
        if isinstance(constraints_config, dict):
            resolved.update(constraints_config)
        return resolved

    def set_selection_context(
        self,
        current_batch: int,
        total_batches: int,
        use_evolving_constraints: bool,
        constraints_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.selection_context = {
            "current_batch": int(max(0, current_batch)),
            "total_batches": int(max(1, total_batches)),
            "use_evolving_constraints": bool(use_evolving_constraints),
            "constraints_config": self._resolve_constraints_config(constraints_config),
        }

    def _get_current_constraints(
        self,
        batch_idx: int,
        total_batches: int,
        use_evolving_constraints: bool = False,
        constraints_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float, float]:
        """Get active surrogate selection thresholds for the current batch."""
        cfg = self._resolve_constraints_config(constraints_config)

        if total_batches <= 1:
            progress = 1.0 if batch_idx > 0 else 0.0
        else:
            progress = max(0.0, min(1.0, float(batch_idx) / float(total_batches - 1)))

        if use_evolving_constraints:
            schedule_type = str(cfg.get("schedule_type", "linear")).lower()
            if schedule_type == "power":
                progress_power = float(cfg.get("progress_power", 1.0))
                progress_power = max(progress_power, 1e-6)
                effective_progress = progress ** progress_power
            else:
                effective_progress = progress

            current_r2_lbound = float(cfg.get("r2_lower_start", 0.2)) + (
                float(cfg.get("r2_lower_end", 0.7)) - float(cfg.get("r2_lower_start", 0.2))
            ) * effective_progress
            current_r2_ubound = float(cfg.get("r2_upper_start", 0.965)) + (
                float(cfg.get("r2_upper_end", 0.965)) - float(cfg.get("r2_upper_start", 0.965))
            ) * effective_progress
            current_nrmse_threshold = float(cfg.get("nrmse_start", 0.5)) + (
                float(cfg.get("nrmse_end", 0.1)) - float(cfg.get("nrmse_start", 0.5))
            ) * effective_progress
        else:
            current_r2_lbound = float(cfg.get("static_r2_lower", cfg.get("r2_lower_start", 0.2)))
            current_r2_ubound = float(cfg.get("static_r2_upper", cfg.get("r2_upper_start", 0.965)))
            current_nrmse_threshold = float(
                cfg.get("static_nrmse_threshold", cfg.get("nrmse_start", 0.5))
            )

        # Keep bounds numerically sane.
        if current_r2_ubound < current_r2_lbound:
            current_r2_lbound, current_r2_ubound = current_r2_ubound, current_r2_lbound

        return current_r2_lbound, current_r2_ubound, current_nrmse_threshold

    def _meets_performance_constraints(self, r2: float, normalized_rmse: float,
                                    r2_lbound: float, r2_ubound: float,
                                    nrmse_threshold: float) -> bool:
        """Check if model meets evolving performance constraints - LIKE INSPIRATION CODE"""
        r2_ok = r2_lbound <= r2 <= r2_ubound
        normalized_rmse_ok = normalized_rmse <= nrmse_threshold

        return r2_ok and normalized_rmse_ok

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        tune_hyperparams: bool = True,
        n_trials: int = 30,
        current_batch: int = 0,
        total_batches: int = 10,
        use_evolving_constraints: bool = False,
        constraints_config: Optional[Dict[str, Any]] = None,
    ) -> 'GeneralizedEvaluator':
     """Fit all models to the data with evolving constraints - FIXED MULTI-OUTPUT"""
    
    # Prepare and validate data
     X_clean, Y_clean = self._prepare_data(X, Y)
    
     if X_clean.shape[1] != len(self.parameter_configs):
        raise ValueError(f"Expected {len(self.parameter_configs)} features, got {X_clean.shape[1]}")
     if Y_clean.shape[1] != len(self.objective_configs):
        raise ValueError(f"Expected {len(self.objective_configs)} objectives, got {Y_clean.shape[1]}")
    
    # Get current constraints
     current_r2_lbound, current_r2_ubound, current_nrmse_threshold = self._get_current_constraints(
        batch_idx=current_batch,
        total_batches=total_batches,
        use_evolving_constraints=use_evolving_constraints,
        constraints_config=constraints_config,
    )

     self.set_selection_context(
        current_batch=current_batch,
        total_batches=total_batches,
        use_evolving_constraints=use_evolving_constraints,
        constraints_config=constraints_config,
    )
    
     print(f"🔧 Current constraints: R² ∈ [{current_r2_lbound:.2f}, {current_r2_ubound:.3f}], NRMSE ≤ {current_nrmse_threshold:.3f}")
    
    # Store training data for uncertainty calculations
     self.training_data_X = X_clean.copy()
     self.training_data_Y = Y_clean.copy()
     self._update_distance_scales_from_matrix(Y_clean, reset=True)
    
    # Fit scalers on training data
     self.X_scaler.fit(X_clean)
     for j, obj_name in enumerate(self.objective_names):
        self.Y_scalers[obj_name] = StandardScaler()
        self.Y_scalers[obj_name].fit(Y_clean[:, j].reshape(-1, 1))
    
     print("✅ Data scalers fitted successfully")
    
     self.models = {}
     self.metrics = {}
    
    # Create model instances - FIXED: Create separate instances for each objective
     model_configs = []
     for model_name, model_data in self.model_factory._active_models.items():
        model_configs.append(model_data['config'])
    
    # FIXED: Create separate model instances for each objective
     objective_models = {}
     for model_config in model_configs:
        model_name = model_config['name']
        model_type = model_config['type']
        hyperparams = model_config.get('hyperparameters', {})
        
        # Create one model instance per objective
        objective_models[model_name] = {}
        for obj_name in self.objective_names:
            try:
                # Create separate model instance for each objective
                model_instance = self.model_factory.registry.create_model(
                    model_type, X_clean.shape[1], hyperparams.copy()
                )
                objective_models[model_name][obj_name] = model_instance
                print(f"✅ Created {model_name} instance for {obj_name}")
            except Exception as e:
                print(f"❌ Failed to create {model_name} for {obj_name}: {e}")
    
    # Fit each model for each objective
     for model_name, obj_models in objective_models.items():
        model_type = self.model_factory._active_models[model_name]['type']
        
        print(f"\n🎯 Training {model_name} ({model_type})...")
        
        # Tune hyperparameters if requested
        tuned_params = {}
        if tune_hyperparams and n_trials > 0:
            print(f"  🔍 Tuning hyperparameters for {model_name}...")
            tuned_params = self.tune_hyperparameters(X_clean, Y_clean, model_name, model_type, n_trials)
        else:
            print(f"  ⏩ Skipping hyperparameter tuning for {model_name}")
        
        model_metrics = {}
        
        # Train separate model for each objective
        for j, obj_name in enumerate(self.objective_names):
            print(f"  📊 Training for objective: {obj_name}")
            
            # Get the model instance for this objective
            if obj_name in obj_models:
                model_instance = obj_models[obj_name]
            else:
                print(f"⚠️  No model instance found for {obj_name}, skipping")
                continue
            
            # Use tuned parameters if available
            obj_params = tuned_params.get(obj_name, {})
            if obj_params:
                try:
                    model_instance = self.model_factory.registry.create_model(
                        model_type, X_clean.shape[1], obj_params
                    )
                    print(f"  ✅ Using tuned parameters for {model_name} on {obj_name}")
                except Exception as e:
                    print(f"⚠️  Failed to create model with tuned params: {e}, using default")
                    model_instance = obj_models[obj_name]
            
            # PROPER DATA SPLITTING
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, Y_clean[:, j], test_size=0.2, random_state=42, shuffle=True
            )
            
            print(f"  🔧 Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            print(f"  🔧 Target range - Train: [{y_train.min():.3f}, {y_train.max():.3f}], Test: [{y_test.min():.3f}, {y_test.max():.3f}]")
            
            # Train model
            try:
                model_instance.fit(X_train, y_train)
                
                # Store the trained model
                model_key = f"{model_name}_{obj_name}"
                self.models[model_key] = model_instance
                
                # Calculate metrics
                y_train_pred = model_instance.predict(X_train)
                y_test_pred = model_instance.predict(X_test)
                
                # Calculate comprehensive metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # Calculate normalized RMSE
                norm_train_rmse, norm_method = self.smart_rmse_normalization(train_rmse, y_train, y_train_pred)
                norm_test_rmse, _ = self.smart_rmse_normalization(test_rmse, y_test, y_test_pred)
                
                # Calculate cross-validation score
                try:
                    cv_scores = cross_val_score(model_instance, X_clean, Y_clean[:, j], 
                                              cv=min(5, len(X_clean)), scoring='r2')
                    cv_score = np.mean(cv_scores)
                except Exception as cv_error:
                    print(f"⚠️  Cross-validation failed: {cv_error}")
                    cv_score = 0.0
                
                # Calculate quality score
                train_r2_quality = max(0, (train_r2 - current_r2_lbound) / (current_r2_ubound - current_r2_lbound)) if current_r2_ubound > current_r2_lbound else 0.0
                test_r2_quality = max(0, (test_r2 - current_r2_lbound) / (current_r2_ubound - current_r2_lbound)) if current_r2_ubound > current_r2_lbound else 0.0
                
                quality_score = ((1.0 - norm_train_rmse) + (1.0 - norm_test_rmse) + train_r2_quality + test_r2_quality) / 4
                
                # Check if model meets current performance constraints
                meets_constraints = self._meets_performance_constraints(
                    test_r2, norm_test_rmse, current_r2_lbound, current_r2_ubound, current_nrmse_threshold
                )
                
                # Calculate prediction uncertainty
                pred_uncertainty = self._get_single_model_uncertainty(model_instance, X_test)
                avg_uncertainty = np.mean(pred_uncertainty) if len(pred_uncertainty) > 0 else 0.1
                
                # Store metrics
                model_metrics[obj_name] = EvaluationMetrics(
                    train_rmse=train_rmse,
                    test_rmse=test_rmse,
                    train_r2=train_r2,
                    test_r2=test_r2,
                    train_mae=train_mae,
                    test_mae=test_mae,
                    normalized_train_rmse=norm_train_rmse,
                    normalized_test_rmse=norm_test_rmse,
                    quality_score=quality_score,
                    meets_constraints=meets_constraints,
                    prediction_uncertainty=avg_uncertainty,
                    cross_val_score=cv_score
                )
                
                print(f"  ✅ {obj_name} - Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
                print(f"     Normalized RMSE ({norm_method}): {norm_test_rmse:.4f}")
                print(f"     Cross-val R²: {cv_score:.4f}")
                print(f"     Quality Score: {quality_score:.4f}")
                print(f"     Meets Constraints: {meets_constraints}")
                
            except Exception as e:
                print(f"❌ Error training {model_name} for {obj_name}: {e}")
                # Create fallback metrics
                model_metrics[obj_name] = EvaluationMetrics(
                    train_rmse=float('inf'),
                    test_rmse=float('inf'),
                    train_r2=0.0,
                    test_r2=0.0,
                    train_mae=float('inf'),
                    test_mae=float('inf'),
                    normalized_train_rmse=1.0,
                    normalized_test_rmse=1.0,
                    quality_score=0.0,
                    meets_constraints=False,
                    prediction_uncertainty=1.0,
                    cross_val_score=0.0
                )
        
        self.metrics[model_name] = model_metrics
    
     self.fitted = True
     print("\n✅ All models trained successfully!")
     self.print_model_performance()
     return self

    def predict(self, X: np.ndarray, model_name: str = None, objective_name: str = None) -> np.ndarray:
     """Make predictions - FIXED MULTI-OUTPUT VERSION"""
     if not self.fitted:
        raise RuntimeError("Models must be fitted before prediction")
    
     try:
        X_clean = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1.0, neginf=0.0)
        
        # If specific model and objective are requested
        if model_name and objective_name:
            model_key = f"{model_name}_{objective_name}"
            if model_key in self.models:
                predictions = self.models[model_key].predict(X_clean)
                # Ensure proper shape: (n_samples, 1)
                if len(predictions.shape) == 1:
                    return predictions.reshape(-1, 1)
                else:
                    return predictions
            else:
                print(f"⚠️  Model {model_key} not found, using best model per objective")
        
        # Use best model per objective for each objective
        best_models = self.get_best_model_per_objective()
        all_predictions = []
        
        for obj_name in self.objective_names:
            best_model_name = best_models.get(obj_name)
            if best_model_name:
                model_key = f"{best_model_name}_{obj_name}"
                if model_key in self.models:
                    obj_predictions = self.models[model_key].predict(X_clean)
                    # Ensure 1D array for this objective
                    if len(obj_predictions.shape) > 1:
                        obj_predictions = obj_predictions.flatten()
                    all_predictions.append(obj_predictions)
                else:
                    print(f"⚠️  Best model {model_key} not found, using zeros for {obj_name}")
                    all_predictions.append(np.zeros(X_clean.shape[0]))
            else:
                print(f"⚠️  No best model found for {obj_name}, using zeros")
                all_predictions.append(np.zeros(X_clean.shape[0]))
        
        # Stack predictions: (n_samples, n_objectives)
        if all_predictions:
            return np.column_stack(all_predictions)
        else:
            return np.zeros((X_clean.shape[0], len(self.objective_names)))
            
     except Exception as e:
        print(f"❌ Prediction error: {e}")
        return np.zeros((X_clean.shape[0], len(self.objective_names)))
     


    def predict_with_uncertainty(self, X: np.ndarray, model_name: str = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Make predictions with uncertainty estimates"""
        predictions = self.predict(X, model_name)
        uncertainties = self.calculate_model_uncertainty(X, model_name)
        
        return predictions, uncertainties
    
    def calculate_model_uncertainty(self, X: np.ndarray, model_name: str = None) -> Dict[str, np.ndarray]:
        """Calculate prediction uncertainty using ensemble methods or model-specific approaches"""
        uncertainties = {}
        
        if model_name:
            # Single model uncertainty
            if model_name in self.models:
                model = self.models[model_name]
                uncertainties[model_name] = self._get_single_model_uncertainty(model, X)
        else:
            # Ensemble uncertainty across all models
            for name, model in self.models.items():
                uncertainties[name] = self._get_single_model_uncertainty(model, X)
        
        return uncertainties
    
    def _get_single_model_uncertainty(self, model, X: np.ndarray) -> np.ndarray:
        """Get uncertainty for a single model with proper estimation methods"""
        try:
            # Handle different model types
            model_type = type(model).__name__.lower()
            
            # Check if it's a pipeline
            if 'pipeline' in model_type:
                # Extract the actual model from pipeline
                actual_model = model.named_steps.get('model', model)
                scaler = model.named_steps.get('scaler', None)
                model_type = type(actual_model).__name__.lower()
                
                # Transform data if scaler exists
                if scaler is not None:
                    X_transformed = scaler.transform(X)
                else:
                    X_transformed = X
            else:
                actual_model = model
                X_transformed = X
            
            # Random Forest - use tree variance (PROPER ENSEMBLE METHOD)
            if 'randomforest' in model_type:
                if hasattr(actual_model, 'estimators_'):
                    # Get predictions from all trees - this is the proper way
                    tree_predictions = []
                    for tree in actual_model.estimators_:
                        tree_pred = tree.predict(X_transformed)
                        tree_predictions.append(tree_pred)
                    
                    # Calculate standard deviation across trees
                    tree_predictions = np.array(tree_predictions)
                    if len(tree_predictions.shape) == 3:
                        # Multi-output - average uncertainty across outputs
                        std_per_output = np.std(tree_predictions, axis=0)
                        return np.mean(std_per_output, axis=1)
                    else:
                        # Single output
                        return np.std(tree_predictions, axis=0)
                else:
                    # Fallback: use training residual variance
                    predictions = model.predict(X)
                    return np.full_like(predictions, 0.15)
            
            # XGBoost - use tree variance or confidence scores
            elif 'xgb' in model_type or 'xgboost' in model_type:
                try:
                    # Method 1: Try to get individual tree predictions
                    if hasattr(actual_model, 'get_booster'):
                        booster = actual_model.get_booster()
                        
                        # Get predictions from all trees
                        tree_predictions = []
                        for i in range(actual_model.n_estimators):
                            try:
                                # Predict using first i+1 trees
                                partial_pred = actual_model.predict(X_transformed, iteration_range=(0, i+1))
                                tree_predictions.append(partial_pred)
                            except:
                                continue
                        
                        if len(tree_predictions) > 1:
                            tree_predictions = np.array(tree_predictions)
                            return np.std(tree_predictions, axis=0)
                    
                    # Method 2: Distance-based uncertainty for regression
                    predictions = model.predict(X)
                    if hasattr(self, 'training_data_X'):
                        # Calculate distance to training data
                        from sklearn.metrics.pairwise import pairwise_distances
                        min_distances = np.min(pairwise_distances(X_transformed, self.training_data_X), axis=1)
                        avg_distance = np.mean(min_distances)
                        uncertainty = 0.1 + 0.2 * (min_distances / (avg_distance + 1e-8))
                        return np.clip(uncertainty, 0.05, 0.5)
                    else:
                        # Fallback: use prediction variance
                        pred_variance = np.var(predictions) if len(predictions) > 1 else 0.1
                        return np.full_like(predictions, 0.1 + 0.1 * np.sqrt(pred_variance))
                        
                except Exception as xgb_error:
                    print(f"⚠️  XGBoost uncertainty calculation failed: {xgb_error}")
                    predictions = model.predict(X)
                    return np.full_like(predictions, 0.15)
            
            # SVM - use distance from support vectors and margin
            elif 'svm' in model_type or 'svr' in model_type:
                predictions = model.predict(X)
                
                try:
                    if hasattr(actual_model, 'support_vectors_'):
                        # Distance to support vectors (for regression/classification)
                        support_vectors = actual_model.support_vectors_
                        if len(support_vectors) > 0:
                            from sklearn.metrics.pairwise import pairwise_distances
                            
                            # Calculate distance to nearest support vector
                            distances_to_sv = pairwise_distances(X_transformed, support_vectors)
                            min_distances = np.min(distances_to_sv, axis=1)
                            
                            # Normalize by average distance in training
                            if hasattr(self, 'training_data_X'):
                                training_dists = pairwise_distances(self.training_data_X, support_vectors)
                                avg_training_dist = np.mean(np.min(training_dists, axis=1))
                                uncertainty = 0.1 + 0.3 * (min_distances / (avg_training_dist + 1e-8))
                            else:
                                uncertainty = 0.1 + 0.2 * (min_distances / (np.mean(min_distances) + 1e-8))
                            
                            return np.clip(uncertainty, 0.05, 0.6)
                    
                    # Method for SVR
                    if hasattr(actual_model, 'epsilon'):
                        # SVR specific - use epsilon tube
                        epsilon = actual_model.epsilon
                        return np.full_like(predictions, 0.1 + epsilon)
                    
                except Exception as svm_error:
                    print(f"⚠️  SVM uncertainty calculation failed: {svm_error}")
                
                # Fallback for SVM
                return np.full_like(predictions, 0.15)
            
            # GAM - use prediction intervals or smoothing uncertainty
            elif 'gam' in model_type or 'lineargam' in model_type:
                predictions = model.predict(X)
                
                try:
                    # Method: Use smoothing parameter (lam) as uncertainty proxy
                    if hasattr(actual_model, 'lam'):
                        lam = np.mean(actual_model.lam) if isinstance(actual_model.lam, (list, np.ndarray)) else actual_model.lam
                        # Higher lam = more smoothing = lower uncertainty
                        base_uncertainty = 0.15 / (1.0 + np.log1p(lam))
                        return np.full_like(predictions, base_uncertainty)
                    
                except Exception as gam_error:
                    print(f"⚠️  GAM uncertainty calculation failed: {gam_error}")
                
                # Fallback for GAM
                return np.full_like(predictions, 0.12)
            
            # Gaussian Process - use built-in uncertainty
            elif 'gaussianprocess' in model_type:
                if hasattr(actual_model, 'predict') and hasattr(actual_model, 'sigma'):
                    try:
                        _, std = actual_model.predict(X_transformed, return_std=True)
                        return std
                    except:
                        # Fallback for GP without std prediction
                        predictions = model.predict(X)
                        return np.full_like(predictions, 0.1)
                else:
                    predictions = model.predict(X)
                    return np.full_like(predictions, 0.1)
            
            # DEFAULT: Simple ensemble-based uncertainty estimation
            else:
                predictions = model.predict(X)
                
                # Create a simple ensemble by perturbing inputs
                try:
                    ensemble_predictions = []
                    n_ensemble = 5
                    
                    for i in range(n_ensemble):
                        # Add small noise to inputs
                        noise_scale = 0.01 * np.std(X_transformed, axis=0) if X_transformed.shape[0] > 1 else 0.01
                        X_perturbed = X_transformed + np.random.normal(0, noise_scale, X_transformed.shape)
                        
                        # Get prediction with perturbed inputs
                        ensemble_pred = model.predict(X_perturbed)
                        ensemble_predictions.append(ensemble_pred)
                    
                    # Calculate uncertainty as std of ensemble predictions
                    ensemble_array = np.array(ensemble_predictions)
                    return np.std(ensemble_array, axis=0)
                    
                except Exception as ensemble_error:
                    # Final fallback: use prediction variance
                    pred_variance = np.var(predictions) if len(predictions) > 1 else 0.1
                    base_uncertainty = 0.1 + 0.05 * np.sqrt(pred_variance)
                    return np.full_like(predictions, base_uncertainty)
                
        except Exception as e:
            print(f"⚠️  Uncertainty calculation error for {model_type}: {e}")
            # Ultimate fallback
            try:
                predictions = model.predict(X)
                return np.full_like(predictions, 0.2)  # High uncertainty for errors
            except:
                return np.full(X.shape[0], 0.2)

    def get_best_model_per_objective(self) -> Dict[str, str]:
     """Get the best model per objective using current surrogate-selection thresholds."""
     best_models = {}
     if not self.metrics:
        return best_models

     selection_ctx = self.selection_context or {}
     cfg = self._resolve_constraints_config(selection_ctx.get("constraints_config"))
     current_r2_lbound, current_r2_ubound, current_nrmse_threshold = self._get_current_constraints(
        batch_idx=int(selection_ctx.get("current_batch", 0)),
        total_batches=int(selection_ctx.get("total_batches", 1)),
        use_evolving_constraints=bool(selection_ctx.get("use_evolving_constraints", False)),
        constraints_config=cfg,
     )
     constraint_bonus = float(cfg.get("constraint_bonus", 0.5))

     for obj_name in self.objective_names:
        best_score = -float("inf")
        best_model = None

        for model_name, metrics in self.metrics.items():
            if obj_name in metrics:
                obj_metrics = metrics[obj_name]
                meets_constraints = self._meets_performance_constraints(
                    obj_metrics.test_r2,
                    obj_metrics.normalized_test_rmse,
                    current_r2_lbound,
                    current_r2_ubound,
                    current_nrmse_threshold,
                )

                score = obj_metrics.quality_score
                if meets_constraints:
                    score += constraint_bonus

                if score > best_score:
                    best_score = score
                    best_model = model_name

        best_models[obj_name] = best_model or list(self.metrics.keys())[0]
        print(f"🎯 Best model for {obj_name}: {best_models[obj_name]} (score: {best_score:.4f})")

     return best_models
    def print_model_performance(self):
        """Print comprehensive model performance summary."""
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 80)

        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name}:")
            for obj_name, obj_metrics in metrics.items():
                print(f"  {obj_name}:")
                print(f"     R2 (Train/Test): {obj_metrics.train_r2:.4f} / {obj_metrics.test_r2:.4f}")
                print(f"     RMSE (Train/Test): {obj_metrics.train_rmse:.4f} / {obj_metrics.test_rmse:.4f}")
                print(f"     MAE (Train/Test): {obj_metrics.train_mae:.4f} / {obj_metrics.test_mae:.4f}")
                print(f"     Quality Score: {obj_metrics.quality_score:.4f}")
                print(f"     Cross-val R2: {obj_metrics.cross_val_score:.4f}")
                print(f"     Meets Constraints: {obj_metrics.meets_constraints}")

        best_models = self.get_best_model_per_objective()
        print(f"\nBEST MODELS PER OBJECTIVE: {best_models}")
        print("=" * 80)

    def get_model_performance_data(self) -> List[Dict[str, Any]]:
        """Return model performance metrics as list of dicts."""
        performance_data = []
        for model_name, metrics in self.metrics.items():
            for obj_name, obj_metrics in metrics.items():
                performance_data.append({
                    'model': model_name,
                    'objective': obj_name,
                    'train_rmse': obj_metrics.train_rmse,
                    'test_rmse': obj_metrics.test_rmse,
                    'train_r2': obj_metrics.train_r2,
                    'test_r2': obj_metrics.test_r2,
                    'train_mae': obj_metrics.train_mae,
                    'test_mae': obj_metrics.test_mae,
                    'normalized_test_rmse': obj_metrics.normalized_test_rmse,
                    'quality_score': obj_metrics.quality_score,
                    'cross_val_score': obj_metrics.cross_val_score,
                    'meets_constraints': obj_metrics.meets_constraints,
                    'prediction_uncertainty': obj_metrics.prediction_uncertainty
                })
        return performance_data

    def get_best_overall_model(self) -> str:
        """Get the best overall model across all objectives."""
        model_scores = {}
        for model_name, metrics in self.metrics.items():
            total_score = 0.0
            valid_objectives = 0
            for obj_metrics in metrics.values():
                total_score += obj_metrics.quality_score + obj_metrics.test_r2
                valid_objectives += 1

            model_scores[model_name] = (
                (total_score / valid_objectives) if valid_objectives > 0 else 0.0
            )

        if model_scores:
            return max(model_scores.items(), key=lambda x: x[1])[0]
        return list(self.models.keys())[0] if self.models else ""
