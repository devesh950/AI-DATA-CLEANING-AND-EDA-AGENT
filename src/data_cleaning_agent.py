"""
Core Data Cleaning Agent with AI-powered preprocessing capabilities
"""

import pandas as pd
import numpy as np
import polars as pl
import dask.dataframe as dd
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
import logging
import gc
from pathlib import Path
from .large_dataset_handler import LargeDatasetHandler

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class DataCleaningAgent:
    """
    AI-powered data cleaning agent for intelligent preprocessing
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 chunk_size: int = 100000,
                 memory_limit_gb: float = 1.5,
                 use_large_dataset_optimization: bool = True):
        """
        Initialize the Data Cleaning Agent
        
        Args:
            config: Configuration dictionary for customizing behavior
            chunk_size: Number of rows to process in each chunk for large datasets
            memory_limit_gb: Memory limit in GB for processing
            use_large_dataset_optimization: Enable optimizations for large datasets
        """
        self.config = config or {}
        self.data = None
        self.original_data = None
        
        # Large dataset handling
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        self.use_large_dataset_optimization = use_large_dataset_optimization
        self.large_dataset_handler = LargeDatasetHandler(
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb
        ) if use_large_dataset_optimization else None
        
        self.is_large_dataset = False
        self.dataset_size_mb = 0
        self.cleaning_report = {}
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, data_source: Union[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Load data from various sources with intelligent format detection and large dataset support
        
        Args:
            data_source: Path to data file or DataFrame
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded pandas DataFrame (optimized for memory usage)
        """
        try:
            # Auto-detect file format and handle large datasets
            if isinstance(data_source, str):
                file_path = Path(data_source)
                
                # Check file size for optimization
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                self.dataset_size_mb = file_size_mb
                self.is_large_dataset = file_size_mb > 200
                
                if self.is_large_dataset and self.use_large_dataset_optimization:
                    self.logger.info(f"🚀 Large dataset detected ({file_size_mb:.1f} MB). Using optimized loading...")
                    
                    # Use large dataset handler
                    self.data = self.large_dataset_handler.load_large_dataset(
                        file_path, **kwargs
                    )
                    
                    # Convert to pandas for compatibility (with sampling if needed)
                    if isinstance(self.data, pl.DataFrame):
                        if self.data.shape[0] > 1000000:
                            self.data = self.data.sample(n=500000).to_pandas()
                        else:
                            self.data = self.data.to_pandas()
                    elif isinstance(self.data, dd.DataFrame):
                        self.data = self.data.sample(frac=0.5).compute()
                    
                    # Apply memory optimization
                    self.data = self.large_dataset_handler.reduce_memory_usage(self.data)
                else:
                    # Standard loading
                    if data_source.endswith('.csv'):
                        self.data = pd.read_csv(data_source, **kwargs)
                    elif data_source.endswith(('.xlsx', '.xls')):
                        self.data = pd.read_excel(data_source, **kwargs)
                    elif data_source.endswith('.parquet'):
                        self.data = pd.read_parquet(data_source, **kwargs)
                    elif data_source.endswith('.json'):
                        self.data = pd.read_json(data_source, **kwargs)
                    else:
                        # Try CSV as default
                        self.data = pd.read_csv(data_source, **kwargs)
                        
            elif isinstance(data_source, pd.DataFrame):
                self.data = data_source.copy()
                
                # Check DataFrame size
                memory_usage_mb = self.data.memory_usage(deep=True).sum() / 1024**2
                self.dataset_size_mb = memory_usage_mb
                self.is_large_dataset = memory_usage_mb > 200
                
                if self.is_large_dataset and self.use_large_dataset_optimization:
                    self.data = self.large_dataset_handler.reduce_memory_usage(self.data)
            else:
                raise ValueError("Unsupported data source type")
                
            self.original_data = self.data.copy()
            self.logger.info(f"✅ Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def assess_data_quality(self) -> Dict[str, Any]:
        """
        Comprehensive data quality assessment with AI insights
        
        Returns:
            Dictionary containing quality metrics and recommendations
        """
        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")
        
        quality_report = {
            'basic_info': self._get_basic_info(),
            'missing_values': self._analyze_missing_values(),
            'duplicates': self._analyze_duplicates(),
            'data_types': self._analyze_data_types(),
            'outliers': self._detect_outliers(),
            'inconsistencies': self._detect_inconsistencies(),
            'recommendations': []
        }
        
        # Generate AI-powered recommendations
        quality_report['recommendations'] = self._generate_cleaning_recommendations(quality_report)
        
        self.cleaning_report = quality_report
        return quality_report
    
    def _get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the dataset"""
        return {
            'shape': self.data.shape,
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'columns': list(self.data.columns),
            'dtypes': dict(self.data.dtypes),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
        }
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing value patterns"""
        missing_counts = self.data.isnull().sum()
        missing_percentage = (missing_counts / len(self.data)) * 100
        
        # Identify missing value patterns
        missing_patterns = {}
        for col in self.data.columns:
            if missing_counts[col] > 0:
                missing_patterns[col] = {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percentage[col]),
                    'pattern': self._identify_missing_pattern(col)
                }
        
        return {
            'total_missing': int(missing_counts.sum()),
            'columns_with_missing': missing_patterns,
            'missing_correlation': self._analyze_missing_correlation()
        }
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate records"""
        duplicate_rows = self.data.duplicated().sum()
        
        # Find partial duplicates (subset of columns)
        partial_duplicates = {}
        for subset_size in range(2, min(len(self.data.columns), 6)):
            for cols in self._get_column_combinations(subset_size):
                dups = self.data.duplicated(subset=list(cols)).sum()
                if dups > 0:
                    partial_duplicates[str(cols)] = int(dups)
        
        return {
            'exact_duplicates': int(duplicate_rows),
            'partial_duplicates': partial_duplicates,
            'duplicate_percentage': float((duplicate_rows / len(self.data)) * 100)
        }
    
    def _analyze_data_types(self) -> Dict[str, Any]:
        """Analyze and suggest optimal data types"""
        type_suggestions = {}
        
        for col in self.data.columns:
            current_type = str(self.data[col].dtype)
            suggested_type = self._suggest_optimal_dtype(col)
            
            if current_type != suggested_type:
                type_suggestions[col] = {
                    'current': current_type,
                    'suggested': suggested_type,
                    'memory_savings': self._calculate_memory_savings(col, suggested_type)
                }
        
        return {
            'current_types': dict(self.data.dtypes),
            'optimization_suggestions': type_suggestions
        }
    
    def _detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        
        for col in numeric_columns:
            if self.data[col].notna().sum() > 0:
                outlier_report[col] = {
                    'iqr_outliers': self._detect_iqr_outliers(col),
                    'zscore_outliers': self._detect_zscore_outliers(col),
                    'isolation_forest': self._detect_isolation_forest_outliers(col)
                }
        
        return outlier_report
    
    def _detect_inconsistencies(self) -> Dict[str, List]:
        """Detect data inconsistencies"""
        inconsistencies = {}
        
        # Check text columns for inconsistencies
        text_columns = self.data.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            issues = []
            
            # Case inconsistencies
            if self.data[col].notna().sum() > 0:
                unique_values = self.data[col].dropna().unique()
                case_issues = self._find_case_inconsistencies(unique_values)
                if case_issues:
                    issues.extend(case_issues)
            
            # Whitespace issues
            whitespace_issues = self._find_whitespace_issues(col)
            if whitespace_issues:
                issues.extend(whitespace_issues)
            
            if issues:
                inconsistencies[col] = issues
        
        return inconsistencies
    
    def auto_clean(self, aggressive: bool = False) -> pd.DataFrame:
        """
        Automatically clean the dataset based on AI recommendations
        
        Args:
            aggressive: Whether to apply aggressive cleaning (may remove more data)
            
        Returns:
            Cleaned DataFrame
        """
        if self.cleaning_report is None:
            self.assess_data_quality()
        
        cleaned_data = self.data.copy()
        
        # Apply cleaning steps based on recommendations
        for recommendation in self.cleaning_report['recommendations']:
            if recommendation['priority'] == 'high' or (aggressive and recommendation['priority'] == 'medium'):
                cleaned_data = self._apply_cleaning_step(cleaned_data, recommendation)
        
        self.data = cleaned_data
        self.logger.info(f"✅ Auto-cleaning completed. New shape: {self.data.shape}")
        return self.data
    
    def _identify_missing_pattern(self, column: str) -> str:
        """Identify the pattern of missing values"""
        # Simple pattern detection - can be enhanced with AI
        missing_mask = self.data[column].isnull()
        
        if missing_mask.all():
            return "completely_missing"
        elif missing_mask.sum() / len(self.data) > 0.5:
            return "majority_missing"
        elif missing_mask.iloc[:int(len(self.data)*0.1)].all():
            return "missing_at_start"
        elif missing_mask.iloc[-int(len(self.data)*0.1):].all():
            return "missing_at_end"
        else:
            return "scattered"
    
    def _analyze_missing_correlation(self) -> Dict[str, float]:
        """Analyze correlation between missing values across columns"""
        missing_df = self.data.isnull().astype(int)
        correlation_matrix = missing_df.corr()
        
        # Find highly correlated missing patterns
        high_correlations = {}
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.7:  # High correlation threshold
                        high_correlations[f"{col1}_vs_{col2}"] = float(corr_value)
        
        return high_correlations
    
    def _get_column_combinations(self, size: int) -> List[Tuple]:
        """Generate column combinations for duplicate analysis"""
        from itertools import combinations
        return list(combinations(self.data.columns, size))[:10]  # Limit for performance
    
    def _suggest_optimal_dtype(self, column: str) -> str:
        """Suggest optimal data type for a column"""
        col_data = self.data[column].dropna()
        
        if len(col_data) == 0:
            return str(self.data[column].dtype)
        
        # Try to convert to numeric
        try:
            pd.to_numeric(col_data, errors='raise')
            # Check if it can be integer
            if col_data.apply(lambda x: float(x).is_integer()).all():
                max_val = col_data.astype(float).max()
                min_val = col_data.astype(float).min()
                
                if min_val >= 0:
                    if max_val < 256:
                        return 'uint8'
                    elif max_val < 65536:
                        return 'uint16'
                    elif max_val < 4294967296:
                        return 'uint32'
                    else:
                        return 'uint64'
                else:
                    if min_val >= -128 and max_val < 128:
                        return 'int8'
                    elif min_val >= -32768 and max_val < 32768:
                        return 'int16'
                    elif min_val >= -2147483648 and max_val < 2147483648:
                        return 'int32'
                    else:
                        return 'int64'
            else:
                return 'float32'
        except:
            # Check if it's categorical
            unique_ratio = len(col_data.unique()) / len(col_data)
            if unique_ratio < 0.5:  # Less than 50% unique values
                return 'category'
            else:
                return 'object'
    
    def _calculate_memory_savings(self, column: str, new_dtype: str) -> int:
        """Calculate memory savings from dtype optimization"""
        current_memory = self.data[column].memory_usage(deep=True)
        # Simplified calculation - would need actual conversion for precise measurement
        dtype_sizes = {
            'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
            'uint8': 1, 'uint16': 2, 'uint32': 4, 'uint64': 8,
            'float32': 4, 'float64': 8, 'category': 2
        }
        
        if new_dtype in dtype_sizes:
            estimated_new_memory = len(self.data) * dtype_sizes[new_dtype]
            return max(0, current_memory - estimated_new_memory)
        
        return 0
    
    def _detect_iqr_outliers(self, column: str) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        data = self.data[column].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    def _detect_zscore_outliers(self, column: str, threshold: float = 3) -> Dict[str, Any]:
        """Detect outliers using Z-score method"""
        data = self.data[column].dropna()
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = data[z_scores > threshold]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'threshold': threshold
        }
    
    def _detect_isolation_forest_outliers(self, column: str) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest"""
        try:
            data = self.data[column].dropna().values.reshape(-1, 1)
            if len(data) > 10:  # Minimum samples required
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(data)
                outlier_count = (outliers == -1).sum()
                
                return {
                    'count': int(outlier_count),
                    'percentage': (outlier_count / len(data)) * 100
                }
        except Exception as e:
            self.logger.warning(f"Isolation Forest failed for {column}: {str(e)}")
        
        return {'count': 0, 'percentage': 0}
    
    def _find_case_inconsistencies(self, values: np.ndarray) -> List[str]:
        """Find case inconsistencies in text data"""
        issues = []
        value_lower_map = {}
        
        for value in values:
            if isinstance(value, str):
                lower_value = value.lower()
                if lower_value in value_lower_map:
                    if value_lower_map[lower_value] != value:
                        issues.append(f"Case inconsistency: '{value}' vs '{value_lower_map[lower_value]}'")
                else:
                    value_lower_map[lower_value] = value
        
        return issues[:5]  # Limit to first 5 issues
    
    def _find_whitespace_issues(self, column: str) -> List[str]:
        """Find whitespace inconsistencies"""
        issues = []
        text_data = self.data[column].dropna()
        
        # Check for leading/trailing whitespace
        has_leading = text_data.astype(str).str.startswith(' ').any()
        has_trailing = text_data.astype(str).str.endswith(' ').any()
        
        if has_leading:
            issues.append("Leading whitespace detected")
        if has_trailing:
            issues.append("Trailing whitespace detected")
        
        return issues
    
    def _generate_cleaning_recommendations(self, quality_report: Dict) -> List[Dict]:
        """Generate AI-powered cleaning recommendations"""
        recommendations = []
        
        # Missing values recommendations
        for col, info in quality_report['missing_values']['columns_with_missing'].items():
            if info['percentage'] > 80:
                recommendations.append({
                    'action': f'drop_column_{col}',
                    'description': f"Drop column '{col}' (>{info['percentage']:.1f}% missing)",
                    'priority': 'high',
                    'column': col
                })
            elif info['percentage'] > 50:
                recommendations.append({
                    'action': f'impute_{col}',
                    'description': f"Consider imputation for '{col}' ({info['percentage']:.1f}% missing)",
                    'priority': 'medium',
                    'column': col
                })
        
        # Duplicate recommendations
        if quality_report['duplicates']['exact_duplicates'] > 0:
            recommendations.append({
                'action': 'remove_duplicates',
                'description': f"Remove {quality_report['duplicates']['exact_duplicates']} duplicate rows",
                'priority': 'high'
            })
        
        # Data type optimization
        for col, info in quality_report['data_types']['optimization_suggestions'].items():
            if info['memory_savings'] > 1000:  # Significant savings
                recommendations.append({
                    'action': f'optimize_dtype_{col}',
                    'description': f"Optimize '{col}' dtype: {info['current']} → {info['suggested']}",
                    'priority': 'low',
                    'column': col,
                    'new_dtype': info['suggested']
                })
        
        return recommendations
    
    def _apply_cleaning_step(self, data: pd.DataFrame, recommendation: Dict) -> pd.DataFrame:
        """Apply a specific cleaning step"""
        action = recommendation['action']
        
        try:
            if action.startswith('drop_column_'):
                col = recommendation['column']
                return data.drop(columns=[col])
            
            elif action.startswith('impute_'):
                col = recommendation['column']
                if data[col].dtype in ['object', 'category']:
                    # Mode imputation for categorical
                    data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown', inplace=True)
                else:
                    # Median imputation for numerical
                    data[col].fillna(data[col].median(), inplace=True)
                return data
            
            elif action == 'remove_duplicates':
                return data.drop_duplicates()
            
            elif action.startswith('optimize_dtype_'):
                col = recommendation['column']
                new_dtype = recommendation['new_dtype']
                try:
                    data[col] = data[col].astype(new_dtype)
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to {new_dtype}: {str(e)}")
                return data
            
            else:
                self.logger.warning(f"Unknown cleaning action: {action}")
                return data
                
        except Exception as e:
            self.logger.error(f"Error applying cleaning step {action}: {str(e)}")
            return data
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of cleaning operations"""
        if self.original_data is None or self.data is None:
            return {}
        
        return {
            'original_shape': self.original_data.shape,
            'cleaned_shape': self.data.shape,
            'rows_removed': self.original_data.shape[0] - self.data.shape[0],
            'columns_removed': self.original_data.shape[1] - self.data.shape[1],
            'memory_usage_before': self.original_data.memory_usage(deep=True).sum(),
            'memory_usage_after': self.data.memory_usage(deep=True).sum(),
            'cleaning_report': self.cleaning_report
        }