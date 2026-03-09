# core/config.py - UPDATED WITH FULLYBAYESIAN STRATEGY

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import yaml
import json
import os


class ParameterType(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    DISCRETE = "discrete"

class ObjectiveType(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize" 
    TARGET_RANGE = "target_range"
    TARGET_VALUE = "target_value"

class BOGenerationStrategy(str, Enum):  # Renamed to avoid conflict
    DEFAULT = "default"
    GPEI = "GPEI"
    SAASBO = "SAASBO"
    FULLYBAYESIAN = "FULLYBAYESIAN"
    BOTORCH_MODULAR = "BOTORCH_MODULAR"
    BO_MIXED = "BO_MIXED"
    ST_MTGP = "ST_MTGP"
    SAAS_MTGP = "SAAS_MTGP"
    THOMPSON = "THOMPSON"
    EMPIRICAL_BAYES_THOMPSON = "EMPIRICAL_BAYES_THOMPSON"
    EB_ASHR = "EB_ASHR"
    FACTORIAL = "FACTORIAL"
    UNIFORM = "UNIFORM"

class BOAcquisitionFunction(str, Enum):
    AUTO = "auto"
    Q_LOG_NEI = "qLogNoisyExpectedImprovement"
    Q_NEI = "qNoisyExpectedImprovement"
    Q_EI = "qExpectedImprovement"
    Q_LOG_EI = "qLogExpectedImprovement"
    Q_KG = "qKnowledgeGradient"
    Q_SIMPLE_REGRET = "qSimpleRegret"
    Q_UCB = "qUpperConfidenceBound"
    Q_LOG_POF = "qLogProbabilityOfFeasibility"
    Q_LOG_NEHVI = "qLogNoisyExpectedHypervolumeImprovement"
    Q_NEHVI = "qNoisyExpectedHypervolumeImprovement"
    Q_EHVI = "qExpectedHypervolumeImprovement"
    Q_LOG_EHVI = "qLogExpectedHypervolumeImprovement"
    Q_LOG_NPAREGO = "qLogNParEGO"

class BOInitializationStrategy(str, Enum):
    NONE = "none"
    SOBOL = "sobol"
    UNIFORM = "uniform"

ACQF_CUSTOMIZABLE_STRATEGIES = {
    BOGenerationStrategy.DEFAULT,
    BOGenerationStrategy.BOTORCH_MODULAR,
    BOGenerationStrategy.SAASBO,
    BOGenerationStrategy.FULLYBAYESIAN,
    BOGenerationStrategy.BO_MIXED,
    BOGenerationStrategy.ST_MTGP,
    BOGenerationStrategy.SAAS_MTGP,
}

SINGLE_OBJECTIVE_ACQFS = (
    BOAcquisitionFunction.Q_LOG_NEI,
    BOAcquisitionFunction.Q_NEI,
    BOAcquisitionFunction.Q_EI,
    BOAcquisitionFunction.Q_LOG_EI,
    BOAcquisitionFunction.Q_KG,
    BOAcquisitionFunction.Q_SIMPLE_REGRET,
    BOAcquisitionFunction.Q_UCB,
    BOAcquisitionFunction.Q_LOG_POF,
)

MULTI_OBJECTIVE_ACQFS = (
    BOAcquisitionFunction.Q_LOG_NEHVI,
    BOAcquisitionFunction.Q_NEHVI,
    BOAcquisitionFunction.Q_EHVI,
    BOAcquisitionFunction.Q_LOG_EHVI,
    BOAcquisitionFunction.Q_LOG_NPAREGO,
    BOAcquisitionFunction.Q_LOG_POF,
)


class EvaluatorType(str, Enum):
    """How suggestions are evaluated."""
    VIRTUAL = "virtual"                 # ML ensemble (existing behaviour)
    SELF_DRIVING_LAB = "self_driving_lab"  # Physical/remote lab
    THIRD_PARTY_SIMULATOR = "third_party_simulator"  # Placeholder


class OptimizationMode(str, Enum):
    """Execution style for BO."""
    BATCH = "batch"
    SEQUENTIAL = "sequential"

@dataclass
class ParameterConfig:
    name: str
    type: Union[ParameterType, str]
    bounds: Optional[Tuple[float, float]] = None
    categories: Optional[List[str]] = None
    step: Optional[float] = None
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = ParameterType(self.type)
    
    def validate(self):
        if self.type == ParameterType.CONTINUOUS:
            if self.bounds is None:
                raise ValueError(f"Continuous parameter {self.name} requires bounds")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"Invalid bounds for {self.name}")
        elif self.type == ParameterType.CATEGORICAL:
            if not self.categories:
                raise ValueError(f"Categorical parameter {self.name} requires categories")
        elif self.type == ParameterType.DISCRETE:
            if self.bounds is None or self.step is None:
                raise ValueError(f"Discrete parameter {self.name} requires bounds and step")
    
# NEW: Add ParameterConstraint class
@dataclass
class ParameterConstraint:
    name: str
    type: str  # 'sum', 'order', 'linear', 'composition'
    expression: str  # e.g., "x1 + x2 <= 10", "x1 <= x2", "2*x1 + 3*x2 <= 15"
    description: str = ""
    
    def validate(self):
        valid_types = ['sum', 'order', 'linear', 'composition']
        if self.type not in valid_types:
            raise ValueError(f"Constraint type must be one of {valid_types}")
        if not self.expression.strip():
            raise ValueError("Constraint expression cannot be empty")

@dataclass
class ObjectiveConfig:
    name: str
    type: Union[ObjectiveType, str]  # Allow both enum and string
    target_range: Optional[Tuple[float, float]] = None
    target_value: Optional[float] = None
    tolerance: float = 0.0
    weight: float = 1.0
    
    def __post_init__(self):
        # Convert string to enum if needed
        if isinstance(self.type, str):
            self.type = ObjectiveType(self.type)
    
    def validate(self):
        if self.type == ObjectiveType.TARGET_RANGE:
            if self.target_range is None:
                raise ValueError(f"Objective {self.name} of type target_range requires target_range")
            if not isinstance(self.target_range, (list, tuple)) or len(self.target_range) != 2:
                raise ValueError(f"Objective {self.name} target_range must contain exactly two values")
            try:
                target_min = float(self.target_range[0])
                target_max = float(self.target_range[1])
            except (TypeError, ValueError):
                raise ValueError(f"Objective {self.name} target_range values must be numeric")
            if target_min >= target_max:
                raise ValueError(f"Objective {self.name} target_range must satisfy min < max")

        if self.type == ObjectiveType.TARGET_VALUE:
            if self.target_value is None:
                raise ValueError(f"Objective {self.name} of type target_value requires target_value")
            try:
                float(self.target_value)
            except (TypeError, ValueError):
                raise ValueError(f"Objective {self.name} target_value must be numeric")
            try:
                tol = float(self.tolerance)
            except (TypeError, ValueError):
                raise ValueError(f"Objective {self.name} tolerance must be numeric")
            if tol < 0:
                raise ValueError(f"Objective {self.name} tolerance must be non-negative")
        if self.weight <= 0:
            raise ValueError(f"Objective {self.name} weight must be positive")

@dataclass
class ModelConfig:
    name: str
    type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

# Update OptimizationConfig to include constraints
@dataclass
class OptimizationConfig:
    experiment_name: str
    parameters: List[ParameterConfig]
    objectives: List[ObjectiveConfig]
    models: List[ModelConfig]

    # NEW: evaluator and execution mode
    evaluator_type: EvaluatorType = EvaluatorType.VIRTUAL
    optimization_mode: OptimizationMode = OptimizationMode.BATCH
    sdl_settings: Dict[str, Any] = field(default_factory=dict)
    task_parameter_name: Optional[str] = None
    
    # NEW: Parameter constraints
    parameter_constraints: List[ParameterConstraint] = field(default_factory=list)
    
    # Existing fields...
    enable_hyperparameter_tuning: bool = True
    n_tuning_trials: int = 20
    batch_iterations: int = 10
    batch_size: int = 5
    max_iterations: int = 100
    random_seed: int = 42
    n_initial_points: int = 10
    generation_strategy: Union[BOGenerationStrategy, str] = BOGenerationStrategy.DEFAULT
    acquisition_function: Union[BOAcquisitionFunction, str] = BOAcquisitionFunction.AUTO
    initialization_strategy: Union[BOInitializationStrategy, str] = BOInitializationStrategy.SOBOL
    initialization_trials: int = 10
    # Backward-compatibility fields; superseded by initialization_strategy/trials.
    use_sobol: bool = True
    sobol_points: int = 10
    use_adaptive_search: bool = True
    adaptive_search_config: Optional[Dict[str, Any]] = None
    use_evolving_constraints: bool = False
    evolving_constraints_config: Optional[Dict[str, Any]] = None
    uncertainty_config: Optional[Dict[str, Any]] = None
    distance_normalization_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if isinstance(self.generation_strategy, str):
            self.generation_strategy = BOGenerationStrategy(self.generation_strategy)

        if isinstance(self.acquisition_function, str):
            self.acquisition_function = BOAcquisitionFunction(self.acquisition_function)

        if isinstance(self.initialization_strategy, str):
            self.initialization_strategy = BOInitializationStrategy(self.initialization_strategy)

        if isinstance(self.evaluator_type, str):
            self.evaluator_type = EvaluatorType(self.evaluator_type)

        if isinstance(self.optimization_mode, str):
            self.optimization_mode = OptimizationMode(self.optimization_mode)

        if isinstance(self.task_parameter_name, str):
            self.task_parameter_name = self.task_parameter_name.strip() or None

        # Synchronize legacy Sobol fields with new initialization fields.
        # New fields take precedence when explicitly set.
        if self.initialization_trials is None:
            self.initialization_trials = int(self.sobol_points) if int(self.sobol_points) > 0 else 0
        else:
            self.initialization_trials = int(self.initialization_trials)
        if (
            self.initialization_strategy != BOInitializationStrategy.NONE
            and self.initialization_trials <= 0
        ):
            self.initialization_trials = int(self.sobol_points) if int(self.sobol_points) > 0 else 1
        self.use_sobol = self.initialization_strategy == BOInitializationStrategy.SOBOL
        self.sobol_points = int(self.initialization_trials)
        
        # Derive sensible iteration defaults based on mode
        if self.optimization_mode == OptimizationMode.SEQUENTIAL:
            # Sequential => one candidate at a time
            if self.batch_size != 1:
                self.batch_size = 1
            if self.max_iterations is None:
                self.max_iterations = max(self.batch_iterations, 1)
            # Align batch_iterations with total iterations for progress tracking
            if self.batch_iterations <= 0:
                self.batch_iterations = self.max_iterations
        else:
            if self.batch_iterations > 0 and (self.max_iterations in [None, 100]):
                self.max_iterations = self.batch_iterations * self.batch_size
    
    def validate(self):
        if not self.parameters:
            raise ValueError("At least one parameter required")
        if not self.objectives:
            raise ValueError("At least one objective required")
        
        if self.evaluator_type == EvaluatorType.VIRTUAL:
            enabled_models = [m for m in self.models if m.enabled]
            if not enabled_models:
                model_names = [m.name for m in self.models]
                raise ValueError(f"No enabled models. Available models: {model_names}. Please enable at least one model.")
        else:
            # Non-virtual modes don't require models
            enabled_models = [m for m in self.models if m.enabled]
            if enabled_models:
                # Ensure models are disabled to avoid confusion
                for m in enabled_models:
                    m.enabled = False
        
        for param in self.parameters:
            param.validate()
        for obj in self.objectives:
            obj.validate()

        parameter_names = [p.name for p in self.parameters]
        has_continuous = any(
            (p.type.value if hasattr(p.type, "value") else str(p.type)) == ParameterType.CONTINUOUS.value
            for p in self.parameters
        )

        if self.task_parameter_name:
            if self.task_parameter_name not in parameter_names:
                raise ValueError(
                    f"task_parameter_name '{self.task_parameter_name}' is not one of the configured parameters: {parameter_names}"
                )

        mtgp_strategies = {BOGenerationStrategy.ST_MTGP, BOGenerationStrategy.SAAS_MTGP}
        if self.generation_strategy in mtgp_strategies:
            if not self.task_parameter_name:
                raise ValueError(
                    f"{self.generation_strategy.value} requires task_parameter_name."
                )
            task_param = next((p for p in self.parameters if p.name == self.task_parameter_name), None)
            if task_param is None:
                raise ValueError(f"Task parameter '{self.task_parameter_name}' not found.")
            task_type = task_param.type.value if hasattr(task_param.type, "value") else str(task_param.type)
            if task_type not in {ParameterType.CATEGORICAL.value, ParameterType.DISCRETE.value}:
                raise ValueError(
                    f"Task parameter '{self.task_parameter_name}' must be categorical or discrete for MTGP strategies."
                )

        discrete_only_strategies = {
            BOGenerationStrategy.THOMPSON,
            BOGenerationStrategy.EMPIRICAL_BAYES_THOMPSON,
            BOGenerationStrategy.EB_ASHR,
            BOGenerationStrategy.FACTORIAL,
        }
        if self.generation_strategy in discrete_only_strategies and has_continuous:
            raise ValueError(
                f"{self.generation_strategy.value} does not support continuous parameters."
            )

        selected_acqf = self.acquisition_function
        if selected_acqf != BOAcquisitionFunction.AUTO:
            if self.generation_strategy not in ACQF_CUSTOMIZABLE_STRATEGIES:
                raise ValueError(
                    f"{self.generation_strategy.value} does not support custom acquisition "
                    "functions. Use acquisition_function='auto'."
                )

            direct_objectives = [
                obj for obj in self.objectives
                if obj.type in [ObjectiveType.MINIMIZE, ObjectiveType.MAXIMIZE]
            ]
            target_objectives = [
                obj for obj in self.objectives
                if obj.type in [ObjectiveType.TARGET_RANGE, ObjectiveType.TARGET_VALUE]
            ]
            # In this platform, if no direct objectives exist we optimize a single
            # synthetic "distance" objective.
            is_multi_objective = (
                len(direct_objectives) > 0 and
                (len(direct_objectives) + len(target_objectives)) > 1
            )
            allowed = MULTI_OBJECTIVE_ACQFS if is_multi_objective else SINGLE_OBJECTIVE_ACQFS
            if selected_acqf not in allowed:
                problem_type = "multi-objective" if is_multi_objective else "single-objective"
                allowed_values = [a.value for a in allowed]
                raise ValueError(
                    f"Acquisition '{selected_acqf.value}' is not valid for {problem_type} "
                    f"setup. Allowed values: {allowed_values}."
                )
        
        # NEW: Validate constraints
        for constraint in self.parameter_constraints:
            constraint.validate()
        
        # Validate batch settings
        if self.batch_iterations <= 0:
            raise ValueError("Batch iterations must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if self.initialization_trials < 0:
            raise ValueError("Initialization trials must be non-negative")
        
        # Validate tuning settings
        if self.enable_hyperparameter_tuning and self.n_tuning_trials < 0:
            raise ValueError("Number of tuning trials must be non-negative")

        if self.use_adaptive_search:
            adaptive_cfg = self.adaptive_search_config or {}
            if not isinstance(adaptive_cfg, dict):
                raise ValueError("adaptive_search_config must be a dictionary.")
            warmup_batches = int(adaptive_cfg.get("warmup_batches", 2))
            update_frequency = int(adaptive_cfg.get("update_frequency", 1))
            top_fraction = float(adaptive_cfg.get("top_fraction", 0.3))
            min_candidates = int(adaptive_cfg.get("min_candidates", 5))
            margin_fraction = float(adaptive_cfg.get("margin_fraction", 0.2))
            min_relative_span = float(adaptive_cfg.get("min_relative_span", 0.15))

            if warmup_batches < 0:
                raise ValueError("adaptive_search_config.warmup_batches must be >= 0.")
            if update_frequency <= 0:
                raise ValueError("adaptive_search_config.update_frequency must be > 0.")
            if not (0.0 < top_fraction <= 1.0):
                raise ValueError("adaptive_search_config.top_fraction must be in (0, 1].")
            if min_candidates <= 0:
                raise ValueError("adaptive_search_config.min_candidates must be > 0.")
            if not (0.0 <= margin_fraction <= 1.0):
                raise ValueError("adaptive_search_config.margin_fraction must be in [0, 1].")
            if not (0.0 < min_relative_span <= 1.0):
                raise ValueError("adaptive_search_config.min_relative_span must be in (0, 1].")

        evolving_cfg = self.evolving_constraints_config or {}
        if not isinstance(evolving_cfg, dict):
            raise ValueError("evolving_constraints_config must be a dictionary.")

        required_numeric_fields = [
            "r2_lower_start",
            "r2_lower_end",
            "r2_upper_start",
            "r2_upper_end",
            "nrmse_start",
            "nrmse_end",
            "static_r2_lower",
            "static_r2_upper",
            "static_nrmse_threshold",
            "constraint_bonus",
            "progress_power",
        ]
        for field_name in required_numeric_fields:
            if field_name in evolving_cfg:
                try:
                    float(evolving_cfg[field_name])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"evolving_constraints_config.{field_name} must be numeric."
                    )

        schedule_type = str(evolving_cfg.get("schedule_type", "linear")).lower()
        if schedule_type not in {"linear", "power"}:
            raise ValueError(
                "evolving_constraints_config.schedule_type must be 'linear' or 'power'."
            )

        uncertainty_cfg = self.uncertainty_config or {}
        if not isinstance(uncertainty_cfg, dict):
            raise ValueError("uncertainty_config must be a dictionary.")

        bool_fields = ["enabled"]
        for field_name in bool_fields:
            if field_name in uncertainty_cfg and not isinstance(uncertainty_cfg[field_name], bool):
                raise ValueError(f"uncertainty_config.{field_name} must be boolean.")

        float_fields = ["fallback_sem", "min_sem", "virtual_sem_scale"]
        for field_name in float_fields:
            if field_name in uncertainty_cfg:
                try:
                    float(uncertainty_cfg[field_name])
                except (TypeError, ValueError):
                    raise ValueError(f"uncertainty_config.{field_name} must be numeric.")

        if float(uncertainty_cfg.get("fallback_sem", 0.0)) < 0.0:
            raise ValueError("uncertainty_config.fallback_sem must be >= 0.")
        if float(uncertainty_cfg.get("min_sem", 0.0)) < 0.0:
            raise ValueError("uncertainty_config.min_sem must be >= 0.")
        if float(uncertainty_cfg.get("virtual_sem_scale", 1.0)) <= 0.0:
            raise ValueError("uncertainty_config.virtual_sem_scale must be > 0.")

        std_mode = str(uncertainty_cfg.get("std_mode", "as_sem")).lower()
        if std_mode not in {"as_sem", "std_to_sem"}:
            raise ValueError("uncertainty_config.std_mode must be 'as_sem' or 'std_to_sem'.")

        if int(uncertainty_cfg.get("default_replicates", 1)) <= 0:
            raise ValueError("uncertainty_config.default_replicates must be > 0.")

        list_fields = ["data_sem_suffixes", "data_std_suffixes", "sdl_sem_keys", "sdl_std_keys"]
        for field_name in list_fields:
            if field_name in uncertainty_cfg and not isinstance(uncertainty_cfg[field_name], list):
                raise ValueError(f"uncertainty_config.{field_name} must be a list of strings.")

        distance_cfg = self.distance_normalization_config or {}
        if not isinstance(distance_cfg, dict):
            raise ValueError("distance_normalization_config must be a dictionary.")

        if "enabled" in distance_cfg and not isinstance(distance_cfg["enabled"], bool):
            raise ValueError("distance_normalization_config.enabled must be boolean.")
        if "normalize_weight_norm" in distance_cfg and not isinstance(distance_cfg["normalize_weight_norm"], bool):
            raise ValueError("distance_normalization_config.normalize_weight_norm must be boolean.")
        if "normalize_target_components" in distance_cfg and not isinstance(distance_cfg["normalize_target_components"], bool):
            raise ValueError("distance_normalization_config.normalize_target_components must be boolean.")

        method = str(distance_cfg.get("method", "quantile")).lower()
        if method not in {"quantile", "range", "std"}:
            raise ValueError("distance_normalization_config.method must be one of: quantile, range, std.")

        q_low = float(distance_cfg.get("q_low", 0.05))
        q_high = float(distance_cfg.get("q_high", 0.95))
        if not (0.0 <= q_low < q_high <= 1.0):
            raise ValueError("distance_normalization_config requires 0 <= q_low < q_high <= 1.")

        min_scale = float(distance_cfg.get("min_scale", 1e-6))
        if min_scale <= 0.0:
            raise ValueError("distance_normalization_config.min_scale must be > 0.")

        clip_component = float(distance_cfg.get("clip_component", 10.0))
        if clip_component < 0.0:
            raise ValueError("distance_normalization_config.clip_component must be >= 0.")

        max_scale_samples = int(distance_cfg.get("max_scale_samples", 2000))
        if max_scale_samples <= 0:
            raise ValueError("distance_normalization_config.max_scale_samples must be > 0.")

class ConfigManager:
    def __init__(self):
        self.current_config: Optional[OptimizationConfig] = None
    
    def create_config_from_ui(self, ui_params: Dict[str, Any]) -> OptimizationConfig:
        """Create config from UI input"""
        parameters = []
        for param_data in ui_params['parameters']:
            param_config = ParameterConfig(**param_data)
            parameters.append(param_config)
        
        objectives = []
        for obj_data in ui_params['objectives']:
            obj_config = ObjectiveConfig(**obj_data)
            objectives.append(obj_config)
        
        models = []
        for model_data in ui_params['models']:
            if model_data['type'] == 'custom_model':
                model_path = model_data.get('hyperparameters', {}).get('model_path', '')
                if model_path and os.path.exists(model_path):
                    model_data['enabled'] = True
                else:
                    model_data['enabled'] = False
                    print(f"⚠️  Custom model {model_data['name']} disabled - file not found: {model_path}")
            
            models.append(ModelConfig(**model_data))
        
        # NEW: Handle parameter constraints
        parameter_constraints = []
        for constraint_data in ui_params.get('parameter_constraints', []):
            constraint = ParameterConstraint(**constraint_data)
            parameter_constraints.append(constraint)
        
        evaluator_type = ui_params.get('evaluator_type', EvaluatorType.VIRTUAL)
        if isinstance(evaluator_type, str):
            evaluator_type = EvaluatorType(evaluator_type)

        enabled_models = [m for m in models if m.enabled]
        if evaluator_type == EvaluatorType.VIRTUAL:
            if not enabled_models:
                if models:
                    models[0].enabled = True
                    print(f"⚠️  No models enabled. Enabling first model: {models[0].name}")
                else:
                    raise ValueError("No models configured. Please add at least one model.")
        else:
            # Disable models for non-virtual evaluators to avoid accidental use
            for m in models:
                m.enabled = False

        default_init_strategy = ui_params.get('initialization_strategy')
        if not default_init_strategy:
            default_init_strategy = 'sobol' if ui_params.get('use_sobol', True) else 'none'
        default_init_trials = ui_params.get('initialization_trials')
        if default_init_trials is None:
            default_init_trials = 0 if default_init_strategy == 'none' else ui_params.get('sobol_points', 10)
        
        config = OptimizationConfig(
            experiment_name=ui_params['experiment_name'],
            parameters=parameters,
            objectives=objectives,
            models=models,
            evaluator_type=evaluator_type,
            optimization_mode=ui_params.get('optimization_mode', OptimizationMode.BATCH),
            sdl_settings=ui_params.get('sdl_settings', {}),
            task_parameter_name=ui_params.get('task_parameter_name'),
            parameter_constraints=parameter_constraints,  # NEW
            enable_hyperparameter_tuning=ui_params.get('enable_hyperparameter_tuning', True),
            n_tuning_trials=ui_params.get('n_tuning_trials', 20),
            batch_iterations=ui_params.get('batch_iterations', 10),
            batch_size=ui_params.get('batch_size', 5),
            max_iterations=ui_params.get('max_iterations', 100),
            random_seed=ui_params.get('random_seed', 42),
            n_initial_points=ui_params.get('n_initial_points', 10),
            generation_strategy=ui_params.get('generation_strategy', 'default'),
            acquisition_function=ui_params.get('acquisition_function', 'auto'),
            initialization_strategy=default_init_strategy,
            initialization_trials=default_init_trials,
            use_sobol=ui_params.get('use_sobol', True),
            sobol_points=ui_params.get('sobol_points', 10),
            use_adaptive_search=ui_params.get('use_adaptive_search', True),
            adaptive_search_config=ui_params.get('adaptive_search_config'),
            use_evolving_constraints=ui_params.get('use_evolving_constraints', False),
            evolving_constraints_config=ui_params.get('evolving_constraints_config'),
            uncertainty_config=ui_params.get('uncertainty_config'),
            distance_normalization_config=ui_params.get('distance_normalization_config'),
        )
        
        config.validate()
        self.current_config = config
        return config
    
    def save_config(self, filepath: str):
        """Save config to YAML file - ENSURES ALL SECTIONS ARE INCLUDED"""
        if not self.current_config:
            raise ValueError("No config to save")
        
        # Build complete config dictionary with all sections
        config_dict = {
            'experiment_name': self.current_config.experiment_name,
            'evaluator_type': self.current_config.evaluator_type.value if hasattr(self.current_config.evaluator_type, 'value') else self.current_config.evaluator_type,
            'optimization_mode': self.current_config.optimization_mode.value if hasattr(self.current_config.optimization_mode, 'value') else self.current_config.optimization_mode,
            'sdl_settings': self.current_config.sdl_settings,
            'task_parameter_name': self.current_config.task_parameter_name,
            'parameters': [
                {
                    'name': p.name,
                    'type': p.type.value,
                    'bounds': p.bounds,
                    'categories': p.categories,
                    'step': p.step
                } for p in self.current_config.parameters
            ],
            'objectives': [
                {
                    'name': o.name,
                    'type': o.type.value,
                    'target_range': o.target_range,
                    'target_value': o.target_value,
                    'tolerance': o.tolerance,
                    'weight': o.weight
                } for o in self.current_config.objectives
            ],
            'models': [
                {
                    'name': m.name,
                    'type': m.type,
                    'hyperparameters': m.hyperparameters,
                    'enabled': m.enabled
                } for m in self.current_config.models
            ],
            # NEW: Always include parameter constraints (even if empty)
            'parameter_constraints': [
                {
                    'name': c.name,
                    'type': c.type,
                    'expression': c.expression,
                    'description': c.description
                } for c in self.current_config.parameter_constraints
            ],
            'optimization_settings': {
                'enable_hyperparameter_tuning': self.current_config.enable_hyperparameter_tuning,
                'n_tuning_trials': self.current_config.n_tuning_trials,
                'batch_iterations': self.current_config.batch_iterations,
                'batch_size': self.current_config.batch_size,
                'max_iterations': self.current_config.max_iterations,
                'optimization_mode': self.current_config.optimization_mode.value if hasattr(self.current_config.optimization_mode, 'value') else self.current_config.optimization_mode,
                'random_seed': self.current_config.random_seed,
                'n_initial_points': self.current_config.n_initial_points,
                'generation_strategy': self.current_config.generation_strategy.value,
                'acquisition_function': self.current_config.acquisition_function.value,
                'initialization_strategy': self.current_config.initialization_strategy.value,
                'initialization_trials': self.current_config.initialization_trials,
                'use_sobol': self.current_config.use_sobol,
                'sobol_points': self.current_config.sobol_points,
                'use_adaptive_search': self.current_config.use_adaptive_search,
                'adaptive_search_config': self.current_config.adaptive_search_config,
                'use_evolving_constraints': self.current_config.use_evolving_constraints,
                'evolving_constraints_config': self.current_config.evolving_constraints_config,
                'uncertainty_config': self.current_config.uncertainty_config,
                'distance_normalization_config': self.current_config.distance_normalization_config,
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def load_config(self, filepath: str) -> OptimizationConfig:
        """Load config from YAML file"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        parameters = [
            ParameterConfig(
                name=p['name'],
                type=ParameterType(p['type']),
                bounds=p.get('bounds'),
                categories=p.get('categories'),
                step=p.get('step')
            ) for p in config_dict['parameters']
        ]
        
        objectives = [
            ObjectiveConfig(
                name=o['name'],
                type=ObjectiveType(o['type']),
                target_range=o.get('target_range'),
                target_value=o.get('target_value'),
                tolerance=o.get('tolerance', 0.0),
                weight=o.get('weight', 1.0)
            ) for o in config_dict['objectives']
        ]
        
        models = [
            ModelConfig(
                name=m['name'],
                type=m['type'],
                hyperparameters=m.get('hyperparameters', {}),
                enabled=m.get('enabled', True)
            ) for m in config_dict['models']
        ]
        
        # NEW: Load parameter constraints
        parameter_constraints = [
            ParameterConstraint(
                name=c['name'],
                type=c['type'],
                expression=c['expression'],
                description=c.get('description', '')
            ) for c in config_dict.get('parameter_constraints', [])
        ]
        
        opt_settings = config_dict.get('optimization_settings', {})
        default_init_strategy = opt_settings.get(
            'initialization_strategy',
            config_dict.get(
                'initialization_strategy',
                'sobol' if opt_settings.get('use_sobol', True) else 'none'
            )
        )
        default_init_trials = opt_settings.get('initialization_trials')
        if default_init_trials is None:
            default_init_trials = config_dict.get('initialization_trials')
        if default_init_trials is None:
            default_init_trials = 0 if default_init_strategy == 'none' else opt_settings.get('sobol_points', 10)
        config = OptimizationConfig(
            experiment_name=config_dict['experiment_name'],
            parameters=parameters,
            objectives=objectives,
            models=models,
            evaluator_type=config_dict.get('evaluator_type', EvaluatorType.VIRTUAL),
            optimization_mode=config_dict.get('optimization_mode', opt_settings.get('optimization_mode', OptimizationMode.BATCH)),
            sdl_settings=config_dict.get('sdl_settings', {}),
            task_parameter_name=config_dict.get('task_parameter_name'),
            parameter_constraints=parameter_constraints,  # NEW
            enable_hyperparameter_tuning=opt_settings.get('enable_hyperparameter_tuning', True),
            n_tuning_trials=opt_settings.get('n_tuning_trials', 20),
            batch_iterations=opt_settings.get('batch_iterations', 10),
            batch_size=opt_settings.get('batch_size', 5),
            max_iterations=opt_settings.get('max_iterations', 100),
            random_seed=opt_settings.get('random_seed', 42),
            n_initial_points=opt_settings.get('n_initial_points', 10),
            generation_strategy=opt_settings.get('generation_strategy', 'default'),
            acquisition_function=opt_settings.get(
                'acquisition_function',
                config_dict.get('acquisition_function', 'auto')
            ),
            initialization_strategy=default_init_strategy,
            initialization_trials=default_init_trials,
            use_sobol=opt_settings.get('use_sobol', True),
            sobol_points=opt_settings.get('sobol_points', 10),
            use_adaptive_search=opt_settings.get('use_adaptive_search', True),
            adaptive_search_config=opt_settings.get('adaptive_search_config'),
            use_evolving_constraints=opt_settings.get('use_evolving_constraints', False),
            evolving_constraints_config=opt_settings.get('evolving_constraints_config'),
            uncertainty_config=opt_settings.get('uncertainty_config'),
            distance_normalization_config=opt_settings.get('distance_normalization_config'),
        )
        
        config.validate()
        self.current_config = config
        return config
