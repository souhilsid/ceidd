# utils/data_loader.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import io

class DataLoader:
    """Generalized data loader for various formats"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
    
    def load_data(self, file_path: str = None, file_object = None, **kwargs) -> pd.DataFrame:
        """Load data from file path or file object"""
        try:
            if file_object is not None:
                return self._load_from_file_object(file_object, **kwargs)
            elif file_path:
                return self._load_from_file_path(file_path, **kwargs)
            else:
                raise ValueError("Either file_path or file_object must be provided")
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
    
    def _load_from_file_object(self, file_object, **kwargs) -> pd.DataFrame:
        """Load data from file object"""
        if hasattr(file_object, 'name'):
            file_extension = self._get_file_extension(file_object.name)
        else:
            # Try to detect format from content
            file_extension = self._detect_file_format(file_object)
        
        file_object.seek(0)  # Reset file pointer
        
        if file_extension == '.csv':
            return pd.read_csv(file_object, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_object, **kwargs)
        elif file_extension == '.json':
            return pd.read_json(file_object, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_from_file_path(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from file path"""
        file_extension = self._get_file_extension(file_path)
        
        if file_extension == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, **kwargs)
        elif file_extension == '.json':
            return pd.read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        return '.' + filename.split('.')[-1].lower()
    
    def _detect_file_format(self, file_object) -> str:
        """Detect file format from content"""
        # Simple detection based on first few bytes
        content_start = file_object.read(100).decode('utf-8', errors='ignore')
        file_object.seek(0)
        
        if content_start.strip().startswith('{'):
            return '.json'
        else:
            # Default to CSV
            return '.csv'
    
    def validate_data_for_optimization(self, df: pd.DataFrame, parameter_columns: List[str], 
                                    objective_columns: List[str]) -> Tuple[bool, List[str]]:
        """Validate data suitability for optimization"""
        errors = []
        
        # Check if all columns exist
        missing_params = [col for col in parameter_columns if col not in df.columns]
        missing_objs = [col for col in objective_columns if col not in df.columns]
        
        if missing_params:
            errors.append(f"Missing parameter columns: {missing_params}")
        if missing_objs:
            errors.append(f"Missing objective columns: {missing_objs}")
        
        if errors:
            return False, errors
        
        # Check for sufficient data
        if len(df) < 5:
            errors.append("Insufficient data points (minimum 5 required)")
        
        # Check for missing values
        relevant_columns = parameter_columns + objective_columns
        missing_counts = df[relevant_columns].isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
        
        if columns_with_missing:
            errors.append(f"Missing values in columns: {columns_with_missing}")
        
        # Check data types
        for col in parameter_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Parameter column '{col}' must be numeric")
        
        for col in objective_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Objective column '{col}' must be numeric")
        
        return len(errors) == 0, errors

    def validate_data_sufficiency(self, df: pd.DataFrame, min_samples: int = 5) -> Tuple[bool, str]:
        """Validate that we have enough data for optimization"""
        if len(df) < min_samples:
            return False, f"Insufficient data: {len(df)} samples. Minimum {min_samples} required."
        
        # Check for enough unique parameter combinations
        param_columns = [col for col in df.columns if 'param' in col.lower() or any(keyword in col.lower() for keyword in ['feature', 'input', 'x'])]
        if len(param_columns) > 0:
            unique_combinations = df[param_columns].drop_duplicates().shape[0]
            if unique_combinations < min_samples:
                return False, f"Only {unique_combinations} unique parameter combinations. More experimental variety needed."
        
        return True, "Data sufficient for optimization"