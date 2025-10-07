"""
ML Model Agent for intelligent model selection, training, and evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class MLModelAgent:
    """
    AI-powered ML model agent for intelligent model selection, training, and evaluation
    """
    
    def __init__(self):
        """Initialize the ML Model Agent"""
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.is_classification = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def set_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_train: pd.Series, y_test: pd.Series) -> None:
        """Set the training and test data"""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is_classification = self._determine_problem_type(y_train)
        
    def analyze_ml_requirements(self) -> Dict[str, Any]:
        """
        Analyze the data and determine ML requirements
        
        Returns:
            Dictionary containing ML analysis and recommendations
        """
        if self.X_train is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        analysis = {
            'problem_type': 'classification' if self.is_classification else 'regression',
            'dataset_info': self._analyze_dataset(),
            'feature_analysis': self._analyze_features(),
            'target_analysis': self._analyze_target(),
            'recommended_models': self._recommend_models(),
            'evaluation_strategy': self._recommend_evaluation_strategy(),
            'hyperparameter_tuning': self._recommend_hyperparameter_strategy()
        }
        
        return analysis
    
    def auto_ml_pipeline(self, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Run automated ML pipeline with model selection and evaluation
        
        Args:
            quick_mode: If True, use faster models and less extensive search
            
        Returns:
            Dictionary containing all model results and recommendations
        """
        print("🚀 Starting Auto-ML Pipeline...")
        
        # Step 1: Get model recommendations
        analysis = self.analyze_ml_requirements()
        recommended_models = analysis['recommended_models']
        
        # Step 2: Train multiple models
        print(f"📚 Training {len(recommended_models)} models...")
        results = {}
        
        for model_name in recommended_models:
            print(f"  ⏳ Training {model_name}...")
            try:
                model_result = self._train_single_model(model_name, quick_mode=quick_mode)
                results[model_name] = model_result
                print(f"  ✅ {model_name} completed")
            except Exception as e:
                print(f"  ❌ {model_name} failed: {str(e)}")
                continue
        
        # Step 3: Compare models and select best
        print("🔍 Comparing model performance...")
        best_model_info = self._select_best_model(results)
        
        # Step 4: Generate comprehensive report
        print("📊 Generating ML report...")
        ml_report = {
            'analysis': analysis,
            'model_results': results,
            'best_model': best_model_info,
            'model_comparison': self._create_model_comparison(results),
            'recommendations': self._generate_ml_recommendations(results, analysis),
            'next_steps': self._suggest_next_steps(best_model_info, analysis)
        }
        
        print("✅ Auto-ML Pipeline completed!")
        return ml_report
    
    def train_specific_model(self, model_name: str, hyperparameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Train a specific model with optional hyperparameters"""
        if self.X_train is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        model = self._get_model(model_name, hyperparameters)
        
        # Train model
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        if self.is_classification:
            metrics = self._calculate_classification_metrics(
                self.y_train, y_pred_train, self.y_test, y_pred_test, model
            )
        else:
            metrics = self._calculate_regression_metrics(
                self.y_train, y_pred_train, self.y_test, y_pred_test
            )
        
        # Feature importance (if available)
        feature_importance = self._get_feature_importance(model)
        
        result = {
            'model': model,
            'model_name': model_name,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
        
        # Store result
        self.models[model_name] = model
        self.model_results[model_name] = result
        
        return result
    
    def _determine_problem_type(self, target: pd.Series) -> bool:
        """Determine if it's a classification or regression problem"""
        # Classification if target has few unique values or is categorical
        unique_ratio = len(target.unique()) / len(target)
        return unique_ratio < 0.1 or target.dtype in ['object', 'category', 'bool']
    
    def _analyze_dataset(self) -> Dict[str, Any]:
        """Analyze dataset characteristics"""
        return {
            'n_samples': len(self.X_train),
            'n_features': self.X_train.shape[1],
            'n_classes': len(self.y_train.unique()) if self.is_classification else None,
            'class_balance': self._analyze_class_balance() if self.is_classification else None,
            'feature_types': {
                'numerical': len(self.X_train.select_dtypes(include=[np.number]).columns),
                'categorical': len(self.X_train.select_dtypes(exclude=[np.number]).columns)
            },
            'data_size_category': self._categorize_data_size()
        }
    
    def _analyze_features(self) -> Dict[str, Any]:
        """Analyze feature characteristics"""
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns
        
        feature_analysis = {
            'total_features': self.X_train.shape[1],
            'numerical_features': len(numerical_cols),
            'categorical_features': self.X_train.shape[1] - len(numerical_cols),
            'missing_values': self.X_train.isnull().sum().sum(),
            'feature_correlation': 'high' if len(numerical_cols) > 1 and self.X_train[numerical_cols].corr().abs().mean().mean() > 0.7 else 'moderate'
        }
        
        # Dimensionality analysis
        samples_per_feature = len(self.X_train) / self.X_train.shape[1]
        if samples_per_feature < 10:
            feature_analysis['dimensionality_issue'] = 'high_dimensional'
        elif samples_per_feature < 100:
            feature_analysis['dimensionality_issue'] = 'moderate'
        else:
            feature_analysis['dimensionality_issue'] = 'low'
            
        return feature_analysis
    
    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze target variable characteristics"""
        target_analysis = {
            'unique_values': len(self.y_train.unique()),
            'data_type': str(self.y_train.dtype)
        }
        
        if self.is_classification:
            target_analysis.update({
                'class_distribution': self.y_train.value_counts().to_dict(),
                'balance_ratio': self.y_train.value_counts().min() / self.y_train.value_counts().max(),
                'is_binary': len(self.y_train.unique()) == 2
            })
        else:
            target_analysis.update({
                'mean': self.y_train.mean(),
                'std': self.y_train.std(),
                'range': self.y_train.max() - self.y_train.min(),
                'skewness': self.y_train.skew()
            })
            
        return target_analysis
    
    def _analyze_class_balance(self) -> Dict[str, Any]:
        """Analyze class balance for classification problems"""
        if not self.is_classification:
            return None
        
        class_counts = self.y_train.value_counts()
        balance_ratio = class_counts.min() / class_counts.max()
        
        if balance_ratio >= 0.8:
            balance_status = 'balanced'
        elif balance_ratio >= 0.5:
            balance_status = 'slightly_imbalanced'
        elif balance_ratio >= 0.1:
            balance_status = 'moderately_imbalanced'
        else:
            balance_status = 'severely_imbalanced'
        
        return {
            'balance_ratio': balance_ratio,
            'balance_status': balance_status,
            'class_counts': class_counts.to_dict(),
            'majority_class': class_counts.index[0],
            'minority_class': class_counts.index[-1]
        }
    
    def _categorize_data_size(self) -> str:
        """Categorize dataset size"""
        n_samples = len(self.X_train)
        
        if n_samples < 1000:
            return 'small'
        elif n_samples < 10000:
            return 'medium'
        elif n_samples < 100000:
            return 'large'
        else:
            return 'very_large'
    
    def _recommend_models(self) -> List[str]:
        """Recommend models based on data characteristics"""
        dataset_info = self._analyze_dataset()
        feature_info = self._analyze_features()
        
        recommended = []
        
        if self.is_classification:
            # Always include these robust classifiers
            recommended.extend(['RandomForest', 'XGBoost', 'LightGBM'])
            
            # Add based on dataset size
            if dataset_info['data_size_category'] in ['small', 'medium']:
                recommended.extend(['SVM', 'KNeighbors'])
            
            # Add based on interpretability needs
            recommended.append('LogisticRegression')
            
            # Add for class imbalance
            if dataset_info['class_balance'] and dataset_info['class_balance']['balance_status'] != 'balanced':
                recommended.append('GradientBoosting')
        
        else:  # Regression
            # Always include these robust regressors
            recommended.extend(['RandomForest', 'XGBoost', 'LightGBM'])
            
            # Add linear models for interpretability
            recommended.extend(['LinearRegression', 'Ridge'])
            
            # Add based on dataset characteristics
            if feature_info['dimensionality_issue'] == 'high_dimensional':
                recommended.append('Lasso')
            
            if dataset_info['data_size_category'] in ['small', 'medium']:
                recommended.extend(['SVR', 'KNeighbors'])
        
        return list(set(recommended))  # Remove duplicates
    
    def _recommend_evaluation_strategy(self) -> Dict[str, Any]:
        """Recommend evaluation strategy based on problem characteristics"""
        dataset_info = self._analyze_dataset()
        
        strategy = {
            'cv_folds': 5 if dataset_info['data_size_category'] != 'small' else 3,
            'scoring_metric': self._get_primary_metric(),
            'additional_metrics': self._get_additional_metrics(),
            'validation_approach': 'cross_validation'
        }
        
        # Adjust for class imbalance
        if self.is_classification and dataset_info['class_balance']:
            if dataset_info['class_balance']['balance_status'] != 'balanced':
                strategy['stratify'] = True
                strategy['additional_considerations'] = ['precision_recall_curve', 'roc_curve']
        
        return strategy
    
    def _recommend_hyperparameter_strategy(self) -> Dict[str, Any]:
        """Recommend hyperparameter tuning strategy"""
        dataset_info = self._analyze_dataset()
        
        if dataset_info['data_size_category'] in ['small', 'medium']:
            search_type = 'grid_search'
            n_iter = 50
        else:
            search_type = 'random_search'
            n_iter = 20
        
        return {
            'search_type': search_type,
            'n_iter': n_iter,
            'cv_folds': 3,
            'scoring': self._get_primary_metric()
        }
    
    def _get_primary_metric(self) -> str:
        """Get the primary evaluation metric"""
        if self.is_classification:
            dataset_info = self._analyze_dataset()
            if dataset_info['class_balance'] and dataset_info['class_balance']['balance_status'] != 'balanced':
                return 'f1_macro'
            else:
                return 'accuracy'
        else:
            return 'neg_mean_squared_error'
    
    def _get_additional_metrics(self) -> List[str]:
        """Get additional evaluation metrics"""
        if self.is_classification:
            return ['precision_macro', 'recall_macro', 'f1_macro']
        else:
            return ['neg_mean_absolute_error', 'r2']
    
    def _train_single_model(self, model_name: str, quick_mode: bool = False) -> Dict[str, Any]:
        """Train a single model with cross-validation"""
        # Get model with default or tuned hyperparameters
        if quick_mode:
            model = self._get_model(model_name)
        else:
            model = self._get_tuned_model(model_name)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5, scoring=self._get_primary_metric()
        )
        
        # Train final model on full training set
        model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Calculate metrics
        if self.is_classification:
            metrics = self._calculate_classification_metrics(
                self.y_train, y_pred_train, self.y_test, y_pred_test, model
            )
        else:
            metrics = self._calculate_regression_metrics(
                self.y_train, y_pred_train, self.y_test, y_pred_test
            )
        
        # Feature importance
        feature_importance = self._get_feature_importance(model)
        
        return {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'metrics': metrics,
            'feature_importance': feature_importance
        }
    
    def _get_model(self, model_name: str, hyperparameters: Optional[Dict] = None) -> Any:
        """Get a model instance"""
        params = hyperparameters or {}
        
        if self.is_classification:
            models = {
                'RandomForest': RandomForestClassifier(random_state=42, **params),
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', **params),
                'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1, **params),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, **params),
                'SVM': SVC(random_state=42, probability=True, **params),
                'KNeighbors': KNeighborsClassifier(**params),
                'GradientBoosting': GradientBoostingClassifier(random_state=42, **params),
                'DecisionTree': DecisionTreeClassifier(random_state=42, **params),
                'ExtraTrees': ExtraTreesClassifier(random_state=42, **params),
                'NaiveBayes': GaussianNB(**params)
            }
        else:
            models = {
                'RandomForest': RandomForestRegressor(random_state=42, **params),
                'XGBoost': xgb.XGBRegressor(random_state=42, **params),
                'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1, **params),
                'LinearRegression': LinearRegression(**params),
                'Ridge': Ridge(random_state=42, **params),
                'Lasso': Lasso(random_state=42, **params),
                'ElasticNet': ElasticNet(random_state=42, **params),
                'SVR': SVR(**params),
                'KNeighbors': KNeighborsRegressor(**params),
                'GradientBoosting': GradientBoostingRegressor(random_state=42, **params),
                'DecisionTree': DecisionTreeRegressor(random_state=42, **params),
                'ExtraTrees': ExtraTreesRegressor(random_state=42, **params)
            }
        
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not supported")
        
        return models[model_name]
    
    def _get_tuned_model(self, model_name: str) -> Any:
        """Get a model with tuned hyperparameters using grid/random search"""
        base_model = self._get_model(model_name)
        param_grids = self._get_param_grids()
        
        if model_name not in param_grids:
            return base_model
        
        # Use RandomizedSearchCV for efficiency
        search = RandomizedSearchCV(
            base_model,
            param_grids[model_name],
            n_iter=20,
            cv=3,
            scoring=self._get_primary_metric(),
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(self.X_train, self.y_train)
        return search.best_estimator_
    
    def _get_param_grids(self) -> Dict[str, Dict]:
        """Get parameter grids for hyperparameter tuning"""
        if self.is_classification:
            return {
                'RandomForest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'LightGBM': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'LogisticRegression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2']
                }
            }
        else:
            return {
                'RandomForest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'Ridge': {
                    'alpha': [0.1, 1, 10, 100]
                },
                'Lasso': {
                    'alpha': [0.1, 1, 10, 100]
                }
            }
    
    def _calculate_classification_metrics(self, y_train_true, y_train_pred, 
                                        y_test_true, y_test_pred, model) -> Dict[str, Any]:
        """Calculate classification metrics"""
        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train_true, y_train_pred),
                'precision': precision_score(y_train_true, y_train_pred, average='macro', zero_division=0),
                'recall': recall_score(y_train_true, y_train_pred, average='macro', zero_division=0),
                'f1': f1_score(y_train_true, y_train_pred, average='macro', zero_division=0)
            },
            'test': {
                'accuracy': accuracy_score(y_test_true, y_test_pred),
                'precision': precision_score(y_test_true, y_test_pred, average='macro', zero_division=0),
                'recall': recall_score(y_test_true, y_test_pred, average='macro', zero_division=0),
                'f1': f1_score(y_test_true, y_test_pred, average='macro', zero_division=0)
            }
        }
        
        # Add ROC-AUC for binary classification if model supports probability prediction
        if len(np.unique(y_test_true)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(self.X_test)[:, 1]
                metrics['test']['roc_auc'] = roc_auc_score(y_test_true, y_proba)
            except:
                pass
        
        return metrics
    
    def _calculate_regression_metrics(self, y_train_true, y_train_pred, 
                                    y_test_true, y_test_pred) -> Dict[str, Any]:
        """Calculate regression metrics"""
        return {
            'train': {
                'mse': mean_squared_error(y_train_true, y_train_pred),
                'mae': mean_absolute_error(y_train_true, y_train_pred),
                'r2': r2_score(y_train_true, y_train_pred)
            },
            'test': {
                'mse': mean_squared_error(y_test_true, y_test_pred),
                'mae': mean_absolute_error(y_test_true, y_test_pred),
                'r2': r2_score(y_test_true, y_test_pred)
            }
        }
    
    def _get_feature_importance(self, model) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
            else:
                return None
            
            feature_names = self.X_train.columns.tolist()
            return dict(zip(feature_names, importance))
        except:
            return None
    
    def _select_best_model(self, results: Dict) -> Dict[str, Any]:
        """Select the best model based on performance"""
        if not results:
            return None
        
        # Choose metric for comparison
        metric_key = 'accuracy' if self.is_classification else 'r2'
        
        best_score = -np.inf
        best_model_name = None
        
        for model_name, result in results.items():
            score = result['metrics']['test'][metric_key]
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        return {
            'model_name': best_model_name,
            'model': results[best_model_name]['model'],
            'performance': results[best_model_name]['metrics'],
            'cv_performance': {
                'mean': results[best_model_name]['cv_mean'],
                'std': results[best_model_name]['cv_std']
            }
        }
    
    def _create_model_comparison(self, results: Dict) -> pd.DataFrame:
        """Create a comparison table of all models"""
        comparison_data = []
        
        for model_name, result in results.items():
            row = {
                'Model': model_name,
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std']
            }
            
            # Add test metrics
            for metric_name, value in result['metrics']['test'].items():
                row[f'Test_{metric_name.title()}'] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values('CV_Mean', ascending=False)
    
    def _generate_ml_recommendations(self, results: Dict, analysis: Dict) -> List[Dict]:
        """Generate ML recommendations based on results and analysis"""
        recommendations = []
        
        # Model performance recommendations
        best_models = sorted(results.items(), 
                           key=lambda x: x[1]['cv_mean'], reverse=True)[:3]
        
        recommendations.append({
            'category': 'model_selection',
            'title': 'Top Performing Models',
            'description': f"Best models: {', '.join([m[0] for m in best_models[:3]])}",
            'priority': 'high'
        })
        
        # Overfitting check
        for model_name, result in results.items():
            test_metric = 'accuracy' if self.is_classification else 'r2'
            train_score = result['metrics']['train'][test_metric]
            test_score = result['metrics']['test'][test_metric]
            
            if train_score - test_score > 0.1:  # Significant gap
                recommendations.append({
                    'category': 'overfitting',
                    'title': f'{model_name} shows overfitting',
                    'description': f'Train score: {train_score:.3f}, Test score: {test_score:.3f}',
                    'priority': 'medium'
                })
        
        # Data recommendations
        if analysis['dataset_info']['data_size_category'] == 'small':
            recommendations.append({
                'category': 'data_collection',
                'title': 'Consider collecting more data',
                'description': 'Small dataset may benefit from more samples',
                'priority': 'medium'
            })
        
        # Feature recommendations
        if analysis['feature_analysis']['dimensionality_issue'] == 'high_dimensional':
            recommendations.append({
                'category': 'feature_selection',
                'title': 'High dimensional data detected',
                'description': 'Consider feature selection or dimensionality reduction',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _suggest_next_steps(self, best_model_info: Dict, analysis: Dict) -> List[str]:
        """Suggest next steps based on results"""
        next_steps = []
        
        # Model improvement
        next_steps.append(f"Fine-tune {best_model_info['model_name']} hyperparameters further")
        
        # Ensemble methods
        if len(self.model_results) > 1:
            next_steps.append("Consider ensemble methods to combine top models")
        
        # Feature engineering
        next_steps.append("Explore advanced feature engineering techniques")
        
        # Model interpretation
        next_steps.append("Analyze feature importance and model interpretability")
        
        # Deployment
        next_steps.append("Prepare model for production deployment")
        
        return next_steps