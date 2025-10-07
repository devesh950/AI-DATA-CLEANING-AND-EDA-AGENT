"""
EDA (Exploratory Data Analysis) Agent with AI-powered insights
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

warnings.filterwarnings('ignore')

class EDAAgent:
    """
    AI-powered Exploratory Data Analysis Agent
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the EDA Agent
        
        Args:
            data: DataFrame to analyze
        """
        self.data = data
        self.insights = {}
        self.visualizations = {}
        self.logger = logging.getLogger(__name__)
        
    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data for analysis"""
        self.data = data.copy()
        self.insights = {}
        self.visualizations = {}
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive EDA report with AI insights
        
        Returns:
            Dictionary containing all analysis results
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() or initialize with data.")
        
        report = {
            'dataset_overview': self._analyze_dataset_overview(),
            'statistical_summary': self._generate_statistical_summary(),
            'correlation_analysis': self._analyze_correlations(),
            'distribution_analysis': self._analyze_distributions(),
            'categorical_analysis': self._analyze_categorical_variables(),
            'outlier_analysis': self._analyze_outliers_comprehensive(),
            'feature_importance': self._analyze_feature_importance(),
            'dimensionality_analysis': self._analyze_dimensionality(),
            'clustering_insights': self._perform_clustering_analysis(),
            'ai_insights': self._generate_ai_insights(),
            'recommendations': []
        }
        
        # Generate recommendations based on findings
        report['recommendations'] = self._generate_eda_recommendations(report)
        
        self.insights = report
        return report
    
    def _analyze_dataset_overview(self) -> Dict[str, Any]:
        """Analyze basic dataset characteristics"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'shape': self.data.shape,
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'column_types': {
                'numeric': len(numeric_cols),
                'categorical': len(categorical_cols),
                'datetime': len(datetime_cols)
            },
            'column_names': {
                'numeric': numeric_cols,
                'categorical': categorical_cols,
                'datetime': datetime_cols
            },
            'missing_data_summary': {
                'total_missing_values': self.data.isnull().sum().sum(),
                'missing_percentage': (self.data.isnull().sum().sum() / self.data.size) * 100,
                'columns_with_missing': self.data.columns[self.data.isnull().any()].tolist()
            }
        }
    
    def _generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate enhanced statistical summary"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'message': 'No numeric columns found'}
        
        # Basic statistics
        basic_stats = numeric_data.describe()
        
        # Additional statistics
        additional_stats = {}
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                additional_stats[col] = {
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data)),
                    'coefficient_of_variation': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else np.inf,
                    'outlier_count_iqr': self._count_iqr_outliers(col_data),
                    'normality_test_pvalue': float(stats.normaltest(col_data)[1]) if len(col_data) >= 8 else None
                }
        
        return {
            'basic_statistics': basic_stats.to_dict(),
            'advanced_statistics': additional_stats
        }
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between variables"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {'message': 'Insufficient numeric columns for correlation analysis'}
        
        # Pearson correlation
        pearson_corr = numeric_data.corr()
        
        # Spearman correlation (rank-based)
        spearman_corr = numeric_data.corr(method='spearman')
        
        # Find highly correlated pairs
        high_correlations = self._find_high_correlations(pearson_corr)
        
        return {
            'pearson_correlation': pearson_corr.to_dict(),
            'spearman_correlation': spearman_corr.to_dict(),
            'high_correlations': high_correlations,
            'correlation_insights': self._interpret_correlations(pearson_corr)
        }
    
    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze distributions of numeric variables"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        distribution_analysis = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                distribution_analysis[col] = {
                    'distribution_type': self._identify_distribution_type(col_data),
                    'normality': {
                        'is_normal': self._test_normality(col_data),
                        'shapiro_pvalue': float(stats.shapiro(col_data[:5000])[1]) if len(col_data) <= 5000 else None
                    },
                    'modality': self._analyze_modality(col_data),
                    'transformation_suggestions': self._suggest_transformations(col_data)
                }
        
        return distribution_analysis
    
    def _analyze_categorical_variables(self) -> Dict[str, Any]:
        """Analyze categorical variables"""
        categorical_data = self.data.select_dtypes(include=['object', 'category'])
        categorical_analysis = {}
        
        for col in categorical_data.columns:
            col_data = categorical_data[col].dropna()
            value_counts = col_data.value_counts()
            
            categorical_analysis[col] = {
                'unique_count': len(value_counts),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'frequency_distribution': value_counts.head(10).to_dict(),
                'cardinality_ratio': len(value_counts) / len(col_data) if len(col_data) > 0 else 0,
                'imbalance_ratio': value_counts.iloc[0] / value_counts.sum() if len(value_counts) > 0 else 0,
                'encoding_suggestions': self._suggest_categorical_encoding(col, value_counts)
            }
        
        return categorical_analysis
    
    def _analyze_outliers_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive outlier analysis"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        outlier_analysis = {}
        
        for col in numeric_data.columns:
            col_data = numeric_data[col].dropna()
            if len(col_data) > 0:
                outlier_analysis[col] = {
                    'iqr_outliers': self._detect_iqr_outliers_detailed(col_data),
                    'zscore_outliers': self._detect_zscore_outliers_detailed(col_data),
                    'modified_zscore': self._detect_modified_zscore_outliers(col_data),
                    'outlier_impact': self._assess_outlier_impact(col_data)
                }
        
        return outlier_analysis
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze feature importance and relationships"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {'message': 'Insufficient features for importance analysis'}
        
        # Calculate variance-based importance
        variances = numeric_data.var().sort_values(ascending=False)
        
        # Calculate correlation-based importance (average absolute correlation with other features)
        corr_matrix = numeric_data.corr().abs()
        avg_correlations = corr_matrix.mean().sort_values(ascending=False)
        
        return {
            'variance_ranking': variances.to_dict(),
            'correlation_importance': avg_correlations.to_dict(),
            'low_variance_features': variances[variances < 0.01].index.tolist(),
            'highly_correlated_groups': self._find_correlated_groups(corr_matrix)
        }
    
    def _analyze_dimensionality(self) -> Dict[str, Any]:
        """Analyze dataset dimensionality and suggest dimensionality reduction"""
        numeric_data = self.data.select_dtypes(include=[np.number]).dropna()
        
        if numeric_data.shape[1] < 3 or numeric_data.shape[0] < 10:
            return {'message': 'Insufficient data for dimensionality analysis'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # PCA analysis
        pca = PCA()
        pca.fit(scaled_data)
        
        # Calculate explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find components for 95% variance
        components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        return {
            'original_dimensions': numeric_data.shape[1],
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'components_for_95_variance': int(components_95),
            'dimensionality_recommendation': self._recommend_dimensionality_reduction(explained_variance_ratio)
        }
    
    def _perform_clustering_analysis(self) -> Dict[str, Any]:
        """Perform basic clustering analysis to identify data structure"""
        numeric_data = self.data.select_dtypes(include=[np.number]).dropna()
        
        if numeric_data.shape[0] < 10 or numeric_data.shape[1] < 2:
            return {'message': 'Insufficient data for clustering analysis'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Try different numbers of clusters
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(11, len(numeric_data) // 3))
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                sil_score = silhouette_score(scaled_data, cluster_labels)
                silhouette_scores.append(sil_score)
                
            except Exception as e:
                self.logger.warning(f"Clustering failed for k={k}: {str(e)}")
        
        # Find optimal k using elbow method
        optimal_k = self._find_optimal_clusters(inertias, k_range)
        
        return {
            'optimal_clusters': optimal_k,
            'inertias': dict(zip(k_range, inertias)) if inertias else {},
            'silhouette_scores': dict(zip(k_range, silhouette_scores)) if silhouette_scores else {},
            'clustering_feasibility': len(inertias) > 0
        }
    
    def _generate_ai_insights(self) -> List[str]:
        """Generate AI-powered insights about the dataset"""
        insights = []
        
        # Dataset size insights
        if self.data.shape[0] > 100000:
            insights.append("Large dataset detected - consider sampling for initial analysis")
        elif self.data.shape[0] < 100:
            insights.append("Small dataset - be cautious about overfitting in ML models")
        
        # Missing data insights
        missing_percentage = (self.data.isnull().sum().sum() / self.data.size) * 100
        if missing_percentage > 20:
            insights.append(f"High missing data ({missing_percentage:.1f}%) - implement robust imputation strategy")
        elif missing_percentage > 5:
            insights.append(f"Moderate missing data ({missing_percentage:.1f}%) - consider impact on analysis")
        
        # Column type insights
        numeric_ratio = len(self.data.select_dtypes(include=[np.number]).columns) / len(self.data.columns)
        if numeric_ratio > 0.8:
            insights.append("Predominantly numeric dataset - suitable for regression and clustering")
        elif numeric_ratio < 0.2:
            insights.append("Predominantly categorical dataset - focus on classification techniques")
        
        # Dimensionality insights
        if self.data.shape[1] > 50:
            insights.append("High-dimensional dataset - consider dimensionality reduction techniques")
        
        # Memory usage insights
        memory_mb = self.data.memory_usage(deep=True).sum() / 1024**2
        if memory_mb > 100:
            insights.append(f"Large memory footprint ({memory_mb:.1f}MB) - consider data type optimization")
        
        return insights
    
    def _generate_eda_recommendations(self, report: Dict) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on EDA findings"""
        recommendations = []
        
        # Missing data recommendations
        missing_pct = report['dataset_overview']['missing_data_summary']['missing_percentage']
        if missing_pct > 15:
            recommendations.append({
                'category': 'Data Quality',
                'recommendation': 'Implement comprehensive missing data imputation strategy',
                'priority': 'High',
                'reason': f'{missing_pct:.1f}% of data is missing'
            })
        
        # Correlation recommendations
        if 'high_correlations' in report['correlation_analysis']:
            high_corr_count = len(report['correlation_analysis']['high_correlations'])
            if high_corr_count > 5:
                recommendations.append({
                    'category': 'Feature Engineering',
                    'recommendation': 'Remove highly correlated features to reduce multicollinearity',
                    'priority': 'Medium',
                    'reason': f'{high_corr_count} highly correlated feature pairs found'
                })
        
        # Dimensionality recommendations
        if 'components_for_95_variance' in report['dimensionality_analysis']:
            original_dims = report['dimensionality_analysis']['original_dimensions']
            reduced_dims = report['dimensionality_analysis']['components_for_95_variance']
            if reduced_dims < original_dims * 0.7:
                recommendations.append({
                    'category': 'Dimensionality',
                    'recommendation': f'Consider PCA: reduce from {original_dims} to {reduced_dims} dimensions',
                    'priority': 'Medium',
                    'reason': 'Significant dimensionality reduction possible with minimal information loss'
                })
        
        # Clustering recommendations
        if 'optimal_clusters' in report['clustering_insights']:
            if report['clustering_insights']['clustering_feasibility']:
                recommendations.append({
                    'category': 'Pattern Discovery',
                    'recommendation': f"Data shows natural clustering structure - explore {report['clustering_insights']['optimal_clusters']} clusters",
                    'priority': 'Low',
                    'reason': 'Clear cluster structure detected in the data'
                })
        
        return recommendations
    
    # Helper methods
    def _count_iqr_outliers(self, data: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return int(((data < lower_bound) | (data > upper_bound)).sum())
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
        """Find highly correlated variable pairs"""
        high_corr_pairs = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlation
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) > threshold:
                        high_corr_pairs.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': float(corr_value)
                        })
        
        return high_corr_pairs
    
    def _interpret_correlations(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Provide interpretation of correlation patterns"""
        # Find strongest positive and negative correlations
        corr_values = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    corr_values.append({
                        'pair': f'{col1}_vs_{col2}',
                        'correlation': corr_matrix.loc[col1, col2]
                    })
        
        if not corr_values:
            return {}
        
        corr_df = pd.DataFrame(corr_values)
        
        return {
            'strongest_positive': corr_df.loc[corr_df['correlation'].idxmax()].to_dict() if len(corr_df) > 0 else None,
            'strongest_negative': corr_df.loc[corr_df['correlation'].idxmin()].to_dict() if len(corr_df) > 0 else None,
            'average_correlation': float(corr_df['correlation'].abs().mean()),
            'correlation_distribution': {
                'strong_positive': int((corr_df['correlation'] > 0.7).sum()),
                'moderate_positive': int(((corr_df['correlation'] > 0.3) & (corr_df['correlation'] <= 0.7)).sum()),
                'weak': int((corr_df['correlation'].abs() <= 0.3).sum()),
                'moderate_negative': int(((corr_df['correlation'] < -0.3) & (corr_df['correlation'] >= -0.7)).sum()),
                'strong_negative': int((corr_df['correlation'] < -0.7).sum())
            }
        }
    
    def _identify_distribution_type(self, data: pd.Series) -> str:
        """Identify the likely distribution type of the data"""
        # Simple heuristic-based identification
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return "approximately_normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        elif kurtosis < -1:
            return "light_tailed"
        else:
            return "unknown"
    
    def _test_normality(self, data: pd.Series, alpha: float = 0.05) -> bool:
        """Test if data follows normal distribution"""
        if len(data) < 8:
            return False
        
        try:
            # Use Shapiro-Wilk test for small samples, Anderson-Darling for larger
            if len(data) <= 5000:
                _, p_value = stats.shapiro(data)
            else:
                _, p_value = stats.normaltest(data)
            
            return p_value > alpha
        except:
            return False
    
    def _analyze_modality(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze modality of the distribution"""
        try:
            # Simple peak detection using histogram
            hist, bins = np.histogram(data, bins='auto')
            
            # Find peaks (local maxima)
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append(i)
            
            return {
                'peak_count': len(peaks),
                'modality': 'unimodal' if len(peaks) == 1 else 'multimodal' if len(peaks) > 1 else 'uniform'
            }
        except:
            return {'peak_count': 0, 'modality': 'unknown'}
    
    def _suggest_transformations(self, data: pd.Series) -> List[str]:
        """Suggest transformations to improve normality"""
        suggestions = []
        
        skewness = abs(stats.skew(data))
        
        if skewness > 1:
            if (data > 0).all():
                suggestions.append("log_transform")
                suggestions.append("sqrt_transform")
            suggestions.append("box_cox_transform")
        
        if (data >= 0).all() and skewness > 0.5:
            suggestions.append("yeo_johnson_transform")
        
        return suggestions
    
    def _suggest_categorical_encoding(self, column: str, value_counts: pd.Series) -> List[str]:
        """Suggest appropriate encoding methods for categorical variables"""
        unique_count = len(value_counts)
        total_count = value_counts.sum()
        
        suggestions = []
        
        if unique_count == 2:
            suggestions.append("binary_encoding")
        elif unique_count < 10:
            suggestions.append("one_hot_encoding")
        elif unique_count > total_count * 0.5:
            suggestions.append("target_encoding")
            suggestions.append("frequency_encoding")
        else:
            suggestions.append("label_encoding")
            suggestions.append("ordinal_encoding")
        
        # Check for high cardinality
        if unique_count > 50:
            suggestions.append("embedding")
            suggestions.append("feature_hashing")
        
        return suggestions
    
    def _detect_iqr_outliers_detailed(self, data: pd.Series) -> Dict[str, Any]:
        """Detailed IQR-based outlier detection"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
            'outlier_values': outliers.tolist()[:10] if len(outliers) <= 10 else outliers.tolist()[:10]
        }
    
    def _detect_zscore_outliers_detailed(self, data: pd.Series, threshold: float = 3) -> Dict[str, Any]:
        """Detailed Z-score based outlier detection"""
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = data[z_scores > threshold]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'threshold': threshold,
            'max_zscore': float(z_scores.max()),
            'outlier_values': outliers.tolist()[:10] if len(outliers) <= 10 else outliers.tolist()[:10]
        }
    
    def _detect_modified_zscore_outliers(self, data: pd.Series, threshold: float = 3.5) -> Dict[str, Any]:
        """Modified Z-score using median and MAD"""
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return {'count': 0, 'percentage': 0}
        
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = data[np.abs(modified_z_scores) > threshold]
        
        return {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'threshold': threshold
        }
    
    def _assess_outlier_impact(self, data: pd.Series) -> Dict[str, float]:
        """Assess the impact of outliers on statistics"""
        # Calculate statistics with and without outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        if len(clean_data) == 0:
            return {}
        
        return {
            'mean_change': float(abs(data.mean() - clean_data.mean())),
            'std_change': float(abs(data.std() - clean_data.std())),
            'median_change': float(abs(data.median() - clean_data.median()))
        }
    
    def _find_correlated_groups(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[List[str]]:
        """Find groups of highly correlated features"""
        # This is a simplified approach - could be enhanced with graph algorithms
        visited = set()
        groups = []
        
        for col1 in corr_matrix.columns:
            if col1 not in visited:
                group = [col1]
                visited.add(col1)
                
                for col2 in corr_matrix.columns:
                    if col2 != col1 and col2 not in visited:
                        if abs(corr_matrix.loc[col1, col2]) > threshold:
                            group.append(col2)
                            visited.add(col2)
                
                if len(group) > 1:
                    groups.append(group)
        
        return groups
    
    def _recommend_dimensionality_reduction(self, explained_variance_ratio: np.ndarray) -> str:
        """Recommend dimensionality reduction strategy"""
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find components needed for different variance thresholds
        components_80 = np.argmax(cumulative_variance >= 0.8) + 1
        components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        total_components = len(explained_variance_ratio)
        
        reduction_ratio = components_95 / total_components
        
        if reduction_ratio < 0.3:
            return "Highly recommended - significant dimensionality reduction possible"
        elif reduction_ratio < 0.6:
            return "Recommended - moderate dimensionality reduction beneficial"
        elif reduction_ratio < 0.8:
            return "Consider - minor dimensionality reduction possible"
        else:
            return "Not recommended - minimal benefit from dimensionality reduction"
    
    def _find_optimal_clusters(self, inertias: List[float], k_range: range) -> int:
        """Find optimal number of clusters using elbow method"""
        if len(inertias) < 2:
            return 2
        
        # Simple elbow detection - calculate rate of change
        rate_changes = []
        for i in range(1, len(inertias)):
            rate_change = inertias[i-1] - inertias[i]
            rate_changes.append(rate_change)
        
        # Find the point where rate of change decreases significantly
        if len(rate_changes) > 1:
            max_change_idx = np.argmax(rate_changes)
            return list(k_range)[max_change_idx + 1]
        
        return list(k_range)[0]