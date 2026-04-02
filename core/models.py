# core/models.py - UPDATED WITH MULTI-FORMAT MODEL SUPPORT

from typing import Dict, Any, List, Optional
import numpy as np
import pickle
import os
import json
import tempfile

# Try to import optional dependencies
try:
    import skops.io as skops
    SKOPS_AVAILABLE = True
except ImportError:
    SKOPS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

np.int = int

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    from pygam import LinearGAM, s
    PYGAM_AVAILABLE = True
except ImportError:
    LinearGAM = None
    s = None
    PYGAM_AVAILABLE = False

warnings.filterwarnings('ignore')

class CustomModelLoader:
    """Loader for custom models in various formats with multi-model dictionary support"""
    
    def __init__(self, model_path: str, model_name: str = "Custom Model", model_format: str = "auto"):
        self.model_path = model_path
        self.model_name = model_name
        self.model_format = model_format
        self.model = None
        self.model_dict = None  # Store the dictionary if it's a multi-model file
        self.fitted = False
        self.is_multi_model = False
        self.load_model()
    
    def detect_format(self) -> str:
        """Detect model format from file extension"""
        ext = os.path.splitext(self.model_path)[1].lower()
        format_map = {
            '.pkl': 'pickle',
            '.pickle': 'pickle',
            '.skops': 'skops',
            '.onnx': 'onnx',
            '.json': 'json',
            '.joblib': 'joblib'
        }
        return format_map.get(ext, 'pickle')
    
    def _safe_pickle_load(self, filepath: str):
        """Safely load pickle file"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"âš ï¸  Pickle load failed: {e}")
            try:
                import joblib
                return joblib.load(filepath)
            except:
                raise e
    
    def _extract_model_from_dict(self, model_dict: dict, objective_name: str = None):
        """Extract the appropriate model from a dictionary based on objective name"""
        print(f"ðŸ” Extracting model from dictionary with keys: {list(model_dict.keys())}")
        
        # If we have an objective name, try to find a matching model
        if objective_name:
            # Try exact match first
            for key, model in model_dict.items():
                if objective_name.lower() in key.lower():
                    print(f"âœ… Found model for {objective_name}: {key}")
                    return model
            
            # Try partial matches
            if 'tensil' in objective_name.lower():
                for key in model_dict.keys():
                    if 'tensil' in key.lower():
                        print(f"âœ… Found Tensil model: {key}")
                        return model_dict[key]
            elif 'flexural' in objective_name.lower():
                for key in model_dict.keys():
                    if 'flexural' in key.lower():
                        print(f"âœ… Found Flexural model: {key}")
                        return model_dict[key]
        
        # Fallback: return the first model that has predict method
        for key, model in model_dict.items():
            if hasattr(model, 'predict'):
                print(f"ðŸ”„ Using first available model: {key}")
                return model
        
        # Last resort: return the first value
        first_key = list(model_dict.keys())[0]
        print(f"âš ï¸  Using first dictionary entry: {first_key}")
        return model_dict[first_key]
    
    def load_model(self):
        """Load model from various formats with multi-model dictionary support"""
        if self.model_format == "auto":
            self.model_format = self.detect_format()
        
        try:
            if self.model_format == "pickle":
                self._load_pickle()
            elif self.model_format == "skops":
                self._load_skops()
            elif self.model_format == "onnx":
                self._load_onnx()
            elif self.model_format == "json":
                self._load_json()
            elif self.model_format == "joblib":
                self._load_joblib()
            else:
                raise ValueError(f"Unsupported model format: {self.model_format}")
            
            self.fitted = True
            print(f"âœ… Loaded {self.model_format} model from: {self.model_path}")
            
        except Exception as e:
            print(f"âŒ Failed to load {self.model_format} model: {e}")
            self._create_fallback_model()
    
    def _load_pickle(self):
        """Load model from pickle format"""
        try:
            loaded_obj = self._safe_pickle_load(self.model_path)
            self._process_loaded_object(loaded_obj)
        except Exception as e:
            print(f"âŒ Pickle loading failed: {e}")
            raise
    
    def _load_skops(self):
        """Load model from skops format with multi-model support"""
        if not SKOPS_AVAILABLE:
            raise ImportError("skops is not available. Install with: pip install skops")
        
        try:
            loaded_obj = skops.load(self.model_path)
            self._process_loaded_object(loaded_obj)
        except Exception as e:
            print(f"âŒ Skops loading failed: {e}")
            # Fallback to pickle
            print("ðŸ”„ Falling back to pickle loading...")
            self._load_pickle()
    
    def _process_loaded_object(self, loaded_obj):
        """Process the loaded object and handle different types"""
        if isinstance(loaded_obj, dict):
            print("ðŸ” Loaded object is a dictionary - multi-model file detected")
            self.model_dict = loaded_obj
            self.is_multi_model = True
            
            # Store the first model as default
            first_model = self._extract_model_from_dict(loaded_obj)
            self.model = first_model
            
            print(f"ðŸ“ Multi-model file loaded with {len(loaded_obj)} models: {list(loaded_obj.keys())}")
        else:
            self.model = loaded_obj
            self.is_multi_model = False
            print(f"ðŸ” Loaded single model object: {type(loaded_obj)}")
    
    def _load_onnx(self):
        """Load model from ONNX format"""
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not available. Install with: pip install onnxruntime")
        self.model = ort.InferenceSession(self.model_path)
    
    def _load_json(self):
        """Load model from JSON format"""
        with open(self.model_path, 'r') as f:
            model_data = json.load(f)
        self.model = JSONLinearModel(model_data)
    
    def _load_joblib(self):
        """Load model from joblib format"""
        try:
            import joblib
            loaded_obj = joblib.load(self.model_path)
            self._process_loaded_object(loaded_obj)
        except ImportError:
            raise ImportError("joblib is not available. Install with: pip install joblib")
        except Exception as e:
            print(f"âŒ Joblib loading failed: {e}")
            self._load_pickle()
    
    def _create_fallback_model(self):
        """Create a fallback model when loading fails"""
        print("ðŸ”„ Creating fallback linear regression model...")
        from sklearn.linear_model import LinearRegression
        
        self.model = LinearRegression()
        self.model.coef_ = np.array([1.0])
        self.model.intercept_ = 0.0
        self.fitted = True
        print("âœ… Fallback model created successfully")
    
    def get_model_for_objective(self, objective_name: str):
        """Get the specific model for a given objective"""
        if not self.is_multi_model or self.model_dict is None:
            return self.model
        
        return self._extract_model_from_dict(self.model_dict, objective_name)
    
    def fit(self, X, y):
        """Custom models are assumed to be pre-trained"""
        print(f"âš ï¸  Custom model {self.model_name} is pre-trained, skipping fit")
        return self
    
    def predict(self, X, objective_name: str = None):
        """Predict using custom model with objective-specific model selection"""
        if not self.fitted:
            raise RuntimeError("Custom model not loaded properly")
        
        try:
            X_clean = np.asarray(X, dtype=np.float64)
            X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Select the appropriate model for the objective
            if self.is_multi_model and objective_name:
                model_to_use = self.get_model_for_objective(objective_name)
                print(f"ðŸŽ¯ Using specific model for {objective_name}")
            else:
                model_to_use = self.model
                if self.is_multi_model:
                    print("âš ï¸  Multi-model file detected but no objective specified, using default model")
            
            if self.model_format == "onnx":
                return self._predict_onnx(X_clean)
            else:
                predictions = model_to_use.predict(X_clean)
                
                # Ensure proper output format
                if isinstance(predictions, (list, np.ndarray)):
                    return np.array(predictions, dtype=np.float64)
                else:
                    return np.array([float(predictions)])
                    
        except Exception as e:
            print(f"âŒ Prediction failed for custom model {self.model_name}: {e}")
            print(f"ðŸ” Model type: {type(self.model)}, is_multi_model: {self.is_multi_model}")
            
            # Fallback prediction
            X_clean = np.asarray(X, dtype=np.float64)
            return np.zeros((X_clean.shape[0], 1))
    
    def _predict_onnx(self, X):
        """Predict using ONNX model"""
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        
        predictions = self.model.run([output_name], {input_name: X})[0]
        return predictions

class SkopsDictModel:
    """Wrapper for dictionary-based models loaded from skops files"""
    
    def __init__(self, model_dict: dict):
        self.model_dict = model_dict
        self._parse_model_dict()
    
    def _parse_model_dict(self):
        """Parse the model dictionary to extract model parameters"""
        self.coefficients = None
        self.intercept = 0.0
        self.feature_names = []
        
        # Try to extract coefficients and intercept
        if 'coef_' in self.model_dict:
            self.coefficients = np.array(self.model_dict['coef_'])
        elif 'coefficients' in self.model_dict:
            self.coefficients = np.array(self.model_dict['coefficients'])
        elif 'weights' in self.model_dict:
            self.coefficients = np.array(self.model_dict['weights'])
        
        if 'intercept_' in self.model_dict:
            self.intercept = self.model_dict['intercept_']
        elif 'intercept' in self.model_dict:
            self.intercept = self.model_dict['intercept']
        elif 'bias' in self.model_dict:
            self.intercept = self.model_dict['bias']
        
        if 'feature_names' in self.model_dict:
            self.feature_names = self.model_dict['feature_names']
        
        print(f"ðŸ” Parsed dictionary model: coefficients={self.coefficients is not None}, intercept={self.intercept}")
    
    def predict(self, X):
        """Predict using dictionary model coefficients"""
        X_array = np.asarray(X, dtype=np.float64)
        
        if self.coefficients is not None:
            # Handle coefficient dimension matching
            if len(self.coefficients) == X_array.shape[1]:
                # Standard matrix multiplication
                predictions = np.dot(X_array, self.coefficients) + self.intercept
            elif len(self.coefficients) > X_array.shape[1]:
                # Use first n coefficients
                predictions = np.dot(X_array, self.coefficients[:X_array.shape[1]]) + self.intercept
            else:
                # Pad with zeros
                padded_coef = np.pad(self.coefficients, (0, X_array.shape[1] - len(self.coefficients)))
                predictions = np.dot(X_array, padded_coef) + self.intercept
            
            return predictions
        else:
            # Fallback: return mean of first feature or zeros
            print("âš ï¸  No coefficients found in dictionary model, using feature mean")
            if X_array.shape[0] > 0:
                return np.full((X_array.shape[0],), np.mean(X_array[:, 0]))
            else:
                return np.zeros((X_array.shape[0],))
            



class PickleModelWrapper:
    """Wrapper for pickle-loaded objects that might not have standard ML interface"""
    
    def __init__(self, model_obj):
        self.model_obj = model_obj
        self._detect_prediction_method()
    
    def _detect_prediction_method(self):
        """Detect how to get predictions from the object"""
        self.predict_method = None
        
        # Check for various prediction methods
        if hasattr(self.model_obj, 'predict'):
            self.predict_method = 'predict'
        elif hasattr(self.model_obj, 'transform'):
            self.predict_method = 'transform'
        elif hasattr(self.model_obj, '__call__'):
            self.predict_method = '__call__'
        elif isinstance(self.model_obj, dict):
            self.predict_method = 'dict'
        else:
            self.predict_method = 'direct'
    
    def predict(self, X):
        """Universal predict method"""
        try:
            if self.predict_method == 'predict':
                return self.model_obj.predict(X)
            elif self.predict_method == 'transform':
                return self.model_obj.transform(X)
            elif self.predict_method == '__call__':
                return self.model_obj(X)
            elif self.predict_method == 'dict':
                # Handle dictionary objects
                dict_model = SkopsDictModel(self.model_obj)
                return dict_model.predict(X)
            else:
                # Try direct access or conversion
                if hasattr(self.model_obj, '__array__'):
                    return self.model_obj.__array__()
                else:
                    return np.array(self.model_obj)
        except Exception as e:
            print(f"âš ï¸  Prediction failed in wrapper: {e}")
            # Return zeros as fallback
            X_array = np.asarray(X, dtype=np.float64)
            return np.zeros((X_array.shape[0], 1))
    
class JSONLinearModel:
    """Simple linear model from JSON configuration"""
    
    def __init__(self, model_data: Dict):
        self.coefficients = np.array(model_data.get('coefficients', [1.0]))
        self.intercept = model_data.get('intercept', 0.0)
        self.feature_names = model_data.get('feature_names', [])
    
    def predict(self, X):
        """Predict using linear model: y = X * coefficients + intercept"""
        X_array = np.asarray(X, dtype=np.float64)
        
        # Handle dimension mismatch
        if X_array.shape[1] != len(self.coefficients):
            if X_array.shape[1] > len(self.coefficients):
                # Use first n coefficients
                coefficients = self.coefficients[:X_array.shape[1]]
            else:
                # Pad with zeros
                coefficients = np.pad(self.coefficients, (0, X_array.shape[1] - len(self.coefficients)))
        else:
            coefficients = self.coefficients
        
        return np.dot(X_array, coefficients) + self.intercept

class ModelRegistry:
    """Registry for all available ML models"""
    
    def __init__(self):
        self._models = {}
        self._default_params = {}
        self.register_default_models()
    
    def register_model(self, name: str, model_class, default_params: Dict[str, Any]):
        """Register a new model type"""
        self._models[name] = model_class
        self._default_params[name] = default_params
    
    def register_default_models(self):
        """Register all default models"""
        # Random Forest
        self.register_model(
            "random_forest",
            RandomForestRegressor,
            {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
        )
        
        # Gaussian Process
        self.register_model(
            "gaussian_process",
            GaussianProcessRegressor,
            {
                'kernel': None,  # Will be set dynamically
                'alpha': 1e-10,
                'normalize_y': True,
                'random_state': 42,
                'n_restarts_optimizer': 3
            }
        )
        
        # XGBoost (optional dependency)
        if XGBOOST_AVAILABLE:
            self.register_model(
                "xgboost",
                XGBRegressor,
                {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                }
            )
        
        # SVM
        self.register_model(
            "svm",
            SVR,
            {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'cache_size': 1000
            }
        )

        # Multi-layer perceptron
        self.register_model(
            "mlp",
            MLPRegressor,
            {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'random_state': 42,
            }
        )

        # Linear regression
        self.register_model(
            "linear_regression",
            LinearRegression,
            {}
        )
        
        # GAM - Use custom wrapper to handle compatibility issues
        self.register_model(
            "gam",
            GAMWrapper,  # Use custom wrapper
            {
                'n_splines': 20,
                'lam': 0.6,
                'spline_order': 3,
                'max_iter': 1000
            }
        )
        
        # Custom Model - UPDATED WITH FORMAT SUPPORT
        self.register_model(
            "custom_model",
            CustomModelLoader,
            {
                'model_path': '',
                'model_name': 'Custom Model',
                'model_format': 'auto'
            }
        )
    
    def get_available_models(self) -> List[str]:
        """Get list of all available model names"""
        return list(self._models.keys())
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported custom model formats"""
        formats = ['pickle', 'auto']
        if SKOPS_AVAILABLE:
            formats.append('skops')
        if ONNX_AVAILABLE:
            formats.append('onnx')
        formats.extend(['json', 'joblib'])
        return formats
    
    def create_model(self, model_type: str, n_features: int, custom_params: Dict[str, Any] = None) -> Any:
     """Create a model instance with given parameters - ENHANCED ERROR HANDLING"""
     if model_type not in self._models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(self._models.keys())}")
    
    # Merge default and custom parameters
     params = self._default_params[model_type].copy()
     if custom_params:
        params.update(custom_params)
    
     model_class = self._models[model_type]
    
     try:
        # Handle special cases
        if model_type == "gaussian_process":
            # Create kernel dynamically based on number of features
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=np.ones(n_features),
                length_scale_bounds=(1e-2, 1e3)
            ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-9, 1e-1))
            params['kernel'] = kernel
            return model_class(**params)
        
        elif model_type == "svm":
            # SVM usually benefits from scaling
            svr = model_class(**params)
            return Pipeline([
                ("scaler", StandardScaler()), 
                ("model", svr)
            ])

        elif model_type in {"mlp", "linear_regression"}:
            # Both models are more stable with standardized inputs.
            base_model = model_class(**params)
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", base_model)
            ])
        
        elif model_type == "gam":
            # GAM uses custom wrapper
            return model_class(n_features=n_features, **params)
        
        elif model_type == "custom_model":
            # Custom model - requires model_path
            if 'model_path' not in params or not params['model_path']:
                raise ValueError("Custom model requires 'model_path' parameter")
            
            # Enhanced validation for custom models
            model_path = params['model_path']
            if not os.path.exists(model_path):
                raise ValueError(f"Custom model file not found: {model_path}")
            
            return model_class(**params)
        
        else:
            return model_class(**params)
            
     except Exception as e:
        print(f"âŒ Error creating model {model_type}: {e}")
        
        # Fallback to simple model
        if model_type != "custom_model":
            print("ðŸ”„ Falling back to Random Forest...")
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=10, random_state=42)
        else:
            # For custom models, re-raise the error
            raise

class GAMWrapper:
    """Wrapper for GAM to handle compatibility issues"""
    
    def __init__(self, n_features: int, n_splines: int = 20, lam: float = 0.6, 
                 spline_order: int = 3, max_iter: int = 1000, **kwargs):
        self.n_features = n_features
        self.n_splines = n_splines
        self._lam = lam  # Use different name to avoid conflict
        self.spline_order = spline_order
        self.max_iter = max_iter
        self.kwargs = kwargs
        
        # Create the GAM model
        self._create_gam_model()
        
        self.scaler = StandardScaler()
        self.fitted = False
        
    def _create_gam_model(self):
        """Create the GAM model with proper terms"""
        if not PYGAM_AVAILABLE:
            self.gam = None
            return

        # Create spline terms for each feature
        terms = s(0, n_splines=self.n_splines, spline_order=self.spline_order, lam=self._lam)
        for i in range(1, self.n_features):
            terms += s(i, n_splines=self.n_splines, spline_order=self.spline_order, lam=self._lam)
        
        # Create GAM - remove problematic parameters
        safe_kwargs = self.kwargs.copy()
        # Remove any parameters that might cause issues
        safe_kwargs.pop('lam', None)
        safe_kwargs.pop('n_splines', None)
        safe_kwargs.pop('spline_order', None)
        
        self.gam = LinearGAM(
            terms, 
            max_iter=self.max_iter,
            **safe_kwargs
        )
    
    def _ensure_dense(self, X):
        """Ensure input is dense numpy array"""
        if hasattr(X, 'toarray'):
            return X.toarray()
        elif hasattr(X, 'A'):
            return X.A
        else:
            return np.asarray(X, dtype=np.float64)
    
    def fit(self, X, y):
        """Fit the GAM model with proper data handling"""
        try:
            # Convert to dense arrays
            X_dense = self._ensure_dense(X)
            y_dense = self._ensure_dense(y).flatten()
            
            # Remove any NaN or Inf values
            X_dense = np.nan_to_num(X_dense, nan=0.0, posinf=1.0, neginf=0.0)
            y_dense = np.nan_to_num(y_dense, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Scale the features
            X_scaled = self.scaler.fit_transform(X_dense)
            
            print(f"ðŸ”§ GAM Training: X shape {X_scaled.shape}, y shape {y_dense.shape}")
            print(f"ðŸ”§ GAM Data ranges - X: [{X_scaled.min():.3f}, {X_scaled.max():.3f}], y: [{y_dense.min():.3f}, {y_dense.max():.3f}]")
            
            # Fit the GAM model
            self.gam.fit(X_scaled, y_dense)
            self.fitted = True
            
            # Calculate training score for debugging
            train_score = self.gam.statistics_['pseudo_r2']['explained_deviance']
            print(f"âœ… GAM fitted successfully - Training RÂ²: {train_score:.4f}")
            
        except Exception as e:
            print(f"âš ï¸  GAM fitting failed: {e}")
            # Fallback: create a simple linear model
            from sklearn.linear_model import LinearRegression
            self.fallback_model = LinearRegression()
            
            # Prepare data for fallback
            X_dense = self._ensure_dense(X)
            X_scaled = self.scaler.fit_transform(X_dense)
            y_dense = self._ensure_dense(y).flatten()
            
            self.fallback_model.fit(X_scaled, y_dense)
            self.fitted = True
            print("âœ… Using LinearRegression fallback for GAM")
        
        return self
    
    def predict(self, X):
        """Predict with the GAM model"""
        if not self.fitted:
            raise RuntimeError("GAM model must be fitted before prediction")
        
        # Convert to dense array
        X_dense = self._ensure_dense(X)
        X_scaled = self.scaler.transform(X_dense)
        
        try:
            # Try to use GAM prediction
            if hasattr(self, 'gam') and self.gam is not None:
                predictions = self.gam.predict(X_scaled)
                print(f"ðŸ”§ GAM Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
                return predictions
            else:
                # Fallback to linear model
                predictions = self.fallback_model.predict(X_scaled)
                print(f"ðŸ”§ GAM Fallback Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
                return predictions
        except Exception as e:
            print(f"âš ï¸  GAM prediction failed: {e}")
            # Ultimate fallback: return mean prediction
            if hasattr(self, 'fallback_model'):
                predictions = self.fallback_model.predict(X_scaled)
            else:
                predictions = np.full(X_scaled.shape[0], 0.5)
            return predictions

class ModelFactory:
    """Factory for creating and managing model instances"""
    
    def __init__(self, registry: ModelRegistry = None):
        self.registry = registry or ModelRegistry()
        self._active_models = {}
    
    def create_model_instances(self, model_configs: List[Dict[str, Any]], n_features: int) -> Dict[str, Any]:
        """Create multiple model instances from configuration"""
        models = {}
        for config in model_configs:
            if config.get('enabled', True):
                model_name = config['name']
                model_type = config['type']
                hyperparams = config.get('hyperparameters', {})
                
                try:
                    # For custom models, we don't need n_features
                    if model_type == "custom_model":
                        model = self.registry.create_model(model_type, 1, hyperparams)  # n_features doesn't matter for custom models
                    else:
                        model = self.registry.create_model(model_type, n_features, hyperparams)
                    
                    models[model_name] = {
                        'instance': model,
                        'type': model_type,
                        'config': config
                    }
                    print(f"âœ… Created model: {model_name} ({model_type})")
                except Exception as e:
                    print(f"âŒ Failed to create model {model_name} ({model_type}): {e}")
        
        self._active_models = models
        return models
    
    def get_model(self, name: str) -> Optional[Any]:
        """Get a model instance by name"""
        return self._active_models.get(name, {}).get('instance')
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all active model instances"""
        return {name: data['instance'] for name, data in self._active_models.items()}
    
    def get_supported_formats(self) -> List[str]:
        """Get supported custom model formats"""
        return self.registry.get_supported_formats()
