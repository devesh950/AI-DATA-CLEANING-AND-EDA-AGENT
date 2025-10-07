"""
Preprocessing Agent for intelligent data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, chi2, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class PreprocessingAgent:
    """
    AI-powered preprocessing agent for intelligent feature engineering and data preparation
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, target_column: Optional[str] = None):
        """
        Initialize the Preprocessing Agent
        
        Args:
            data: DataFrame to preprocess
            target_column: Name of the target column (for supervised learning)
        """
        self.data = data
        self.original_data = None
        self.target_column = target_column
        self.preprocessing_steps = []
        self.feature_names = []
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}
        
    def set_data(self, data: pd.DataFrame, target_column: Optional[str] = None) -> None:
        """Set the data for preprocessing"""
        self.data = data.copy()
        self.original_data = data.copy()
        self.target_column = target_column
        self.preprocessing_steps = []
        
    def analyze_preprocessing_needs(self) -> Dict[str, Any]:
        """
        Analyze the data and recommend preprocessing steps
        
        Returns:
            Dictionary containing preprocessing recommendations
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        analysis = {
            'missing_values': self._analyze_missing_values(),
            'categorical_variables': self._analyze_categorical_variables(),
            'numerical_variables': self._analyze_numerical_variables(),
            'feature_engineering': self._analyze_feature_engineering_opportunities(),
            'scaling_needs': self._analyze_scaling_needs(),
            'feature_selection': self._analyze_feature_selection_needs(),
            'recommendations': []
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_preprocessing_recommendations(analysis)
        
        return analysis
    
    def auto_preprocess(self, 
                       target_column: Optional[str] = None,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
        """
        Automatically preprocess the data with intelligent defaults
        
        Args:
            target_column: Target column for supervised learning
            test_size: Size of test split
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing preprocessed data and preprocessing info
        """
        if target_column:
            self.target_column = target_column
        
        # Step 1: Handle missing values
        self._handle_missing_values()
        
        # Step 2: Encode categorical variables
        self._encode_categorical_variables()
        
        # Step 3: Scale numerical variables
        self._scale_numerical_variables()
        
        # Step 4: Feature engineering
        self._create_engineered_features()
        
        # Step 5: Feature selection (if target is provided)
        if self.target_column and self.target_column in self.data.columns:
            self._perform_feature_selection()
        
        # Step 6: Split data if target is provided
        result = {
            'processed_data': self.data.copy(),
            'preprocessing_steps': self.preprocessing_steps.copy(),
            'feature_names': self.feature_names.copy() if self.feature_names else list(self.data.columns),
            'encoders': self.encoders.copy(),
            'scalers': self.scalers.copy(),
            'imputers': self.imputers.copy()
        }
        
        if self.target_column and self.target_column in self.data.columns:
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if self._is_classification_target(y) else None
            )
            
            result.update({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'train_test_split_info': {
                    'test_size': test_size,
                    'random_state': random_state,
                    'stratify': self._is_classification_target(y)
                }
            })
        
        return result
    
    def _analyze_missing_values(self) -> Dict[str, Any]:
        """Analyze missing value patterns and suggest strategies"""
        missing_info = {}
        
        for col in self.data.columns:
            missing_count = self.data[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(self.data)) * 100
                
                # Determine strategy based on missing percentage and data type
                if missing_pct > 50:
                    strategy = "drop_column"
                elif self.data[col].dtype in ['object', 'category']:
                    strategy = "mode_imputation" if missing_pct < 20 else "new_category"
                else:
                    if missing_pct < 5:
                        strategy = "mean_median_imputation"
                    elif missing_pct < 20:
                        strategy = "knn_imputation"
                    else:
                        strategy = "advanced_imputation"
                
                missing_info[col] = {
                    'missing_count': missing_count,
                    'missing_percentage': missing_pct,
                    'data_type': str(self.data[col].dtype),
                    'recommended_strategy': strategy
                }
        
        return missing_info
    
    def _analyze_categorical_variables(self) -> Dict[str, Any]:
        """Analyze categorical variables and suggest encoding strategies"""
        categorical_info = {}
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            unique_count = self.data[col].nunique()
            total_count = len(self.data[col].dropna())
            cardinality_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Determine encoding strategy
            if unique_count == 2:
                encoding_strategy = "binary_encoding"
            elif unique_count <= 10 and cardinality_ratio < 0.5:
                encoding_strategy = "one_hot_encoding"
            elif unique_count <= 50:
                encoding_strategy = "label_encoding"
            elif cardinality_ratio > 0.5:
                encoding_strategy = "target_encoding"
            else:
                encoding_strategy = "frequency_encoding"
            
            categorical_info[col] = {
                'unique_count': unique_count,
                'cardinality_ratio': cardinality_ratio,
                'recommended_encoding': encoding_strategy,
                'sample_values': self.data[col].dropna().unique()[:5].tolist()
            }
        
        return categorical_info
    
    def _analyze_numerical_variables(self) -> Dict[str, Any]:
        """Analyze numerical variables and suggest preprocessing strategies"""
        numerical_info = {}
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_columns:
            col_data = self.data[col].dropna()
            if len(col_data) == 0:
                continue
            
            # Calculate statistics
            skewness = abs(col_data.skew())
            kurtosis = col_data.kurtosis()
            range_val = col_data.max() - col_data.min()
            
            # Detect outliers
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = len(col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)])
            
            # Recommend scaling strategy
            if range_val > 1000 or col_data.std() > 100:
                scaling_strategy = "standard_scaling"
            elif skewness > 1:
                scaling_strategy = "robust_scaling"
            elif col_data.min() >= 0 and col_data.max() <= 1:
                scaling_strategy = "no_scaling"
            else:
                scaling_strategy = "minmax_scaling"
            
            # Recommend transformation
            transformation = "none"
            if skewness > 1.5 and (col_data > 0).all():
                transformation = "log_transform"
            elif skewness > 1:
                transformation = "box_cox"
            
            numerical_info[col] = {
                'statistics': {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'skewness': skewness,
                    'kurtosis': kurtosis
                },
                'outlier_count': outlier_count,
                'outlier_percentage': (outlier_count / len(col_data)) * 100,
                'recommended_scaling': scaling_strategy,
                'recommended_transformation': transformation
            }
        
        return numerical_info
    
    def _analyze_feature_engineering_opportunities(self) -> Dict[str, List[str]]:
        """Identify feature engineering opportunities"""
        opportunities = {
            'polynomial_features': [],
            'interaction_features': [],
            'binning_candidates': [],
            'ratio_features': [],
            'aggregation_features': []
        }
        
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Polynomial features for skewed distributions
        for col in numerical_columns:
            col_data = self.data[col].dropna()
            if len(col_data) > 0 and abs(col_data.skew()) > 1:
                opportunities['polynomial_features'].append(col)
        
        # Interaction features for correlated variables
        if len(numerical_columns) >= 2:
            corr_matrix = self.data[numerical_columns].corr()
            for i, col1 in enumerate(numerical_columns):
                for col2 in numerical_columns[i+1:]:
                    if abs(corr_matrix.loc[col1, col2]) > 0.3:  # Moderate correlation
                        opportunities['interaction_features'].append(f"{col1}_x_{col2}")
        
        # Binning for continuous variables with wide ranges
        for col in numerical_columns:
            col_data = self.data[col].dropna()
            if len(col_data) > 0:
                unique_ratio = len(col_data.unique()) / len(col_data)
                if unique_ratio > 0.1 and (col_data.max() - col_data.min()) > col_data.std() * 3:
                    opportunities['binning_candidates'].append(col)
        
        # Ratio features
        for i, col1 in enumerate(numerical_columns):
            for col2 in numerical_columns[i+1:]:
                if (self.data[col2] != 0).all():  # Avoid division by zero
                    opportunities['ratio_features'].append(f"{col1}_div_{col2}")
        
        return opportunities
    
    def _analyze_scaling_needs(self) -> Dict[str, Any]:
        """Analyze scaling requirements"""
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_columns) < 2:
            return {'scaling_needed': False, 'reason': 'Insufficient numerical columns'}
        
        # Check scale differences
        scales = {}
        for col in numerical_columns:
            col_data = self.data[col].dropna()
            if len(col_data) > 0:
                scales[col] = {
                    'mean': abs(col_data.mean()),
                    'std': col_data.std(),
                    'range': col_data.max() - col_data.min()
                }
        
        # Calculate scale ratios
        means = [scales[col]['mean'] for col in scales]
        ranges = [scales[col]['range'] for col in scales]
        
        max_mean_ratio = max(means) / min(means) if min(means) > 0 else float('inf')
        max_range_ratio = max(ranges) / min(ranges) if min(ranges) > 0 else float('inf')
        
        scaling_needed = max_mean_ratio > 10 or max_range_ratio > 10
        
        return {
            'scaling_needed': scaling_needed,
            'mean_ratio': max_mean_ratio,
            'range_ratio': max_range_ratio,
            'recommended_scaler': 'StandardScaler' if max_mean_ratio > max_range_ratio else 'MinMaxScaler'
        }
    
    def _analyze_feature_selection_needs(self) -> Dict[str, Any]:
        """Analyze feature selection requirements"""
        n_features = self.data.shape[1]
        n_samples = self.data.shape[0]
        
        # Rule of thumb: need at least 10 samples per feature
        feature_selection_needed = n_features > n_samples / 10
        
        return {
            'feature_selection_needed': feature_selection_needed,
            'current_features': n_features,
            'samples': n_samples,
            'recommended_max_features': max(10, n_samples // 10),
            'dimensionality_ratio': n_features / n_samples
        }
    
    def _generate_preprocessing_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate preprocessing recommendations based on analysis"""
        recommendations = []
        
        # Missing value recommendations
        high_missing_cols = [
            col for col, info in analysis['missing_values'].items()
            if info['missing_percentage'] > 30
        ]
        if high_missing_cols:
            recommendations.append({
                'step': 'missing_values',
                'action': f'Handle missing values in {len(high_missing_cols)} columns',
                'priority': 'high',
                'details': f"Columns with >30% missing: {', '.join(high_missing_cols[:3])}"
            })
        
        # Categorical encoding recommendations
        if analysis['categorical_variables']:
            recommendations.append({
                'step': 'categorical_encoding',
                'action': f'Encode {len(analysis["categorical_variables"])} categorical variables',
                'priority': 'high',
                'details': 'Required for machine learning algorithms'
            })
        
        # Scaling recommendations
        if analysis['scaling_needs']['scaling_needed']:
            recommendations.append({
                'step': 'feature_scaling',
                'action': f'Apply {analysis["scaling_needs"]["recommended_scaler"]}',
                'priority': 'medium',
                'details': f'Scale ratio: {analysis["scaling_needs"]["range_ratio"]:.2f}'
            })
        
        # Feature selection recommendations
        if analysis['feature_selection']['feature_selection_needed']:
            recommendations.append({
                'step': 'feature_selection',
                'action': f'Reduce features from {analysis["feature_selection"]["current_features"]} to ~{analysis["feature_selection"]["recommended_max_features"]}',
                'priority': 'medium',
                'details': 'Prevent overfitting and improve performance'
            })
        
        # Feature engineering recommendations
        engineering_opportunities = analysis['feature_engineering']
        if any(len(opportunities) > 0 for opportunities in engineering_opportunities.values()):
            recommendations.append({
                'step': 'feature_engineering',
                'action': 'Create engineered features',
                'priority': 'low',
                'details': 'Polynomial, interaction, and ratio features available'
            })
        
        return recommendations
    
    def _handle_missing_values(self) -> None:
        """Handle missing values based on analysis"""
        missing_analysis = self._analyze_missing_values()
        
        for col, info in missing_analysis.items():
            strategy = info['recommended_strategy']
            
            if strategy == "drop_column":
                self.data.drop(columns=[col], inplace=True)
                self.preprocessing_steps.append(f"Dropped column '{col}' (>{info['missing_percentage']:.1f}% missing)")
                
            elif strategy == "mode_imputation":
                mode_value = self.data[col].mode().iloc[0] if not self.data[col].mode().empty else 'Unknown'
                self.data[col].fillna(mode_value, inplace=True)
                self.preprocessing_steps.append(f"Imputed '{col}' with mode: '{mode_value}'")
                
            elif strategy == "new_category":
                self.data[col].fillna('Missing', inplace=True)
                self.preprocessing_steps.append(f"Filled missing values in '{col}' with 'Missing' category")
                
            elif strategy == "mean_median_imputation":
                if self.data[col].skew() > 1:
                    fill_value = self.data[col].median()
                    method = "median"
                else:
                    fill_value = self.data[col].mean()
                    method = "mean"
                
                self.data[col].fillna(fill_value, inplace=True)
                self.imputers[col] = {'method': method, 'value': fill_value}
                self.preprocessing_steps.append(f"Imputed '{col}' with {method}: {fill_value:.3f}")
                
            elif strategy in ["knn_imputation", "advanced_imputation"]:
                # Use KNN imputation for more sophisticated missing value handling
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if col in numeric_cols:
                    imputer = KNNImputer(n_neighbors=5)
                    self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
                    self.imputers[col] = {'method': 'knn', 'imputer': imputer}
                    self.preprocessing_steps.append(f"Applied KNN imputation to '{col}'")
                else:
                    # For categorical, use mode
                    mode_value = self.data[col].mode().iloc[0] if not self.data[col].mode().empty else 'Unknown'
                    self.data[col].fillna(mode_value, inplace=True)
                    self.preprocessing_steps.append(f"Imputed categorical '{col}' with mode: '{mode_value}'")
    
    def _encode_categorical_variables(self) -> None:
        """Encode categorical variables"""
        categorical_analysis = self._analyze_categorical_variables()
        
        for col, info in categorical_analysis.items():
            if col not in self.data.columns:  # Column might have been dropped
                continue
                
            strategy = info['recommended_encoding']
            
            if strategy == "binary_encoding":
                # Simple binary encoding (0, 1)
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.encoders[col] = {'method': 'label', 'encoder': le}
                self.preprocessing_steps.append(f"Binary encoded '{col}'")
                
            elif strategy == "one_hot_encoding":
                # One-hot encoding
                encoded_cols = pd.get_dummies(self.data[col], prefix=col)
                self.data = self.data.drop(columns=[col])
                self.data = pd.concat([self.data, encoded_cols], axis=1)
                self.encoders[col] = {'method': 'onehot', 'columns': encoded_cols.columns.tolist()}
                self.preprocessing_steps.append(f"One-hot encoded '{col}' into {len(encoded_cols.columns)} columns")
                
            elif strategy == "label_encoding":
                # Label encoding
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.encoders[col] = {'method': 'label', 'encoder': le}
                self.preprocessing_steps.append(f"Label encoded '{col}'")
                
            elif strategy == "frequency_encoding":
                # Frequency encoding
                freq_map = self.data[col].value_counts().to_dict()
                self.data[col] = self.data[col].map(freq_map)
                self.encoders[col] = {'method': 'frequency', 'mapping': freq_map}
                self.preprocessing_steps.append(f"Frequency encoded '{col}'")
    
    def _scale_numerical_variables(self) -> None:
        """Scale numerical variables"""
        scaling_analysis = self._analyze_scaling_needs()
        
        if not scaling_analysis['scaling_needed']:
            return
        
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numerical_columns) == 0:
            return
        
        # Choose scaler based on analysis
        scaler_name = scaling_analysis['recommended_scaler']
        
        if scaler_name == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_name == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()  # Default to robust scaler
        
        # Fit and transform
        scaled_data = scaler.fit_transform(self.data[numerical_columns])
        self.data[numerical_columns] = scaled_data
        
        self.scalers['numerical'] = {'scaler': scaler, 'columns': numerical_columns.tolist()}
        self.preprocessing_steps.append(f"Applied {scaler_name} to {len(numerical_columns)} numerical columns")
    
    def _create_engineered_features(self) -> None:
        """Create engineered features"""
        engineering_opportunities = self._analyze_feature_engineering_opportunities()
        
        # Create polynomial features for top candidates
        poly_candidates = engineering_opportunities['polynomial_features'][:3]  # Limit to top 3
        for col in poly_candidates:
            if col in self.data.columns:
                self.data[f"{col}_squared"] = self.data[col] ** 2
                self.preprocessing_steps.append(f"Created polynomial feature: {col}_squared")
        
        # Create interaction features for top candidates
        interaction_candidates = engineering_opportunities['interaction_features'][:5]  # Limit to top 5
        for interaction in interaction_candidates:
            col1, col2 = interaction.split('_x_')
            if col1 in self.data.columns and col2 in self.data.columns:
                self.data[interaction] = self.data[col1] * self.data[col2]
                self.preprocessing_steps.append(f"Created interaction feature: {interaction}")
        
        # Create ratio features for top candidates  
        ratio_candidates = engineering_opportunities['ratio_features'][:3]  # Limit to top 3
        for ratio in ratio_candidates:
            col1, col2 = ratio.split('_div_')
            if col1 in self.data.columns and col2 in self.data.columns:
                # Avoid division by zero
                self.data[ratio] = self.data[col1] / (self.data[col2] + 1e-8)
                self.preprocessing_steps.append(f"Created ratio feature: {ratio}")
    
    def _perform_feature_selection(self) -> None:
        """Perform feature selection if target is available"""
        if not self.target_column or self.target_column not in self.data.columns:
            return
        
        selection_analysis = self._analyze_feature_selection_needs()
        
        if not selection_analysis['feature_selection_needed']:
            return
        
        # Prepare data
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # Choose selection method based on target type
        if self._is_classification_target(y):
            # Classification: use chi2 for categorical, f_classif for numerical
            selector = SelectKBest(score_func=chi2, k=selection_analysis['recommended_max_features'])
        else:
            # Regression: use f_regression
            selector = SelectKBest(score_func=f_regression, k=selection_analysis['recommended_max_features'])
        
        try:
            # Ensure all features are non-negative for chi2
            if self._is_classification_target(y):
                X_min = X.min()
                if (X_min < 0).any():
                    # Shift to make all values non-negative
                    X = X - X_min
            
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Update data with selected features
            self.data = pd.concat([
                pd.DataFrame(X_selected, columns=selected_features, index=self.data.index),
                y
            ], axis=1)
            
            self.feature_names = selected_features
            self.preprocessing_steps.append(f"Selected {len(selected_features)} features using {selector.score_func.__name__}")
            
        except Exception as e:
            # If feature selection fails, continue without it
            self.preprocessing_steps.append(f"Feature selection skipped due to error: {str(e)}")
    
    def _is_classification_target(self, target: pd.Series) -> bool:
        """Determine if target is for classification or regression"""
        # Simple heuristic: if target has few unique values or is categorical, it's classification
        unique_ratio = len(target.unique()) / len(target)
        return unique_ratio < 0.1 or target.dtype in ['object', 'category', 'bool']
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get a summary of all preprocessing steps performed"""
        return {
            'original_shape': self.original_data.shape if self.original_data is not None else None,
            'final_shape': self.data.shape if self.data is not None else None,
            'preprocessing_steps': self.preprocessing_steps,
            'encoders_used': list(self.encoders.keys()),
            'scalers_used': list(self.scalers.keys()),
            'imputers_used': list(self.imputers.keys()),
            'feature_names': self.feature_names
        }