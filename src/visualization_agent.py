"""
Visualization Agent for creating intelligent data visualizations
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

warnings.filterwarnings('ignore')

class VisualizationAgent:
    """
    AI-powered visualization agent that recommends and creates optimal visualizations
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the Visualization Agent
        
        Args:
            data: DataFrame to visualize
        """
        self.data = data
        self.viz_recommendations = {}
        
    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data for visualization"""
        self.data = data.copy()
        self.viz_recommendations = {}
    
    def recommend_visualizations(self) -> Dict[str, List[Dict]]:
        """
        Recommend optimal visualizations based on data characteristics
        
        Returns:
            Dictionary of visualization recommendations by category
        """
        if self.data is None:
            raise ValueError("No data provided. Use set_data() first.")
        
        recommendations = {
            'distribution': self._recommend_distribution_plots(),
            'relationships': self._recommend_relationship_plots(),
            'categorical': self._recommend_categorical_plots(),
            'temporal': self._recommend_temporal_plots(),
            'multivariate': self._recommend_multivariate_plots()
        }
        
        self.viz_recommendations = recommendations
        return recommendations
    
    def create_auto_dashboard(self) -> List[go.Figure]:
        """
        Create an automated dashboard with the most relevant visualizations
        
        Returns:
            List of Plotly figures
        """
        if not self.viz_recommendations:
            self.recommend_visualizations()
        
        figures = []
        
        # Create top recommended visualizations
        for category, recs in self.viz_recommendations.items():
            for rec in recs[:2]:  # Top 2 from each category
                try:
                    fig = self._create_visualization(rec)
                    if fig:
                        figures.append(fig)
                except Exception as e:
                    print(f"Error creating {rec['type']}: {str(e)}")
                    continue
        
        return figures
    
    def _recommend_distribution_plots(self) -> List[Dict]:
        """Recommend distribution-related visualizations"""
        recommendations = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols[:5]:  # Limit to avoid too many recommendations
            # Check distribution characteristics
            col_data = self.data[col].dropna()
            if len(col_data) == 0:
                continue
            
            skewness = abs(col_data.skew())
            unique_ratio = len(col_data.unique()) / len(col_data)
            
            # Histogram for continuous variables
            if unique_ratio > 0.05:
                recommendations.append({
                    'type': 'histogram',
                    'column': col,
                    'title': f'Distribution of {col}',
                    'priority': 'high' if skewness > 1 else 'medium',
                    'reason': f'Understand distribution shape (skewness: {skewness:.2f})'
                })
            
            # Box plot for outlier detection
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = len(col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)])
            
            if outlier_count > 0:
                recommendations.append({
                    'type': 'box',
                    'column': col,
                    'title': f'Box Plot of {col}',
                    'priority': 'high' if outlier_count > len(col_data) * 0.05 else 'medium',
                    'reason': f'Identify outliers ({outlier_count} potential outliers)'
                })
        
        return recommendations
    
    def _recommend_relationship_plots(self) -> List[Dict]:
        """Recommend relationship visualizations"""
        recommendations = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Correlation heatmap
            corr_matrix = self.data[numeric_cols].corr()
            high_corr_count = (abs(corr_matrix) > 0.7).sum().sum() - len(numeric_cols)  # Exclude diagonal
            
            recommendations.append({
                'type': 'correlation_heatmap',
                'columns': numeric_cols,
                'title': 'Correlation Matrix',
                'priority': 'high' if high_corr_count > 0 else 'medium',
                'reason': f'Analyze relationships between variables ({high_corr_count} strong correlations found)'
            })
            
            # Scatter plots for highly correlated pairs
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_val = abs(corr_matrix.loc[col1, col2])
                    if corr_val > 0.5:  # Strong correlation threshold
                        recommendations.append({
                            'type': 'scatter',
                            'x_column': col1,
                            'y_column': col2,
                            'title': f'{col2} vs {col1}',
                            'priority': 'high' if corr_val > 0.7 else 'medium',
                            'reason': f'Strong correlation detected (r={corr_val:.3f})'
                        })
        
        return recommendations[:10]  # Limit recommendations
    
    def _recommend_categorical_plots(self) -> List[Dict]:
        """Recommend categorical visualizations"""
        recommendations = []
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        for cat_col in categorical_cols[:3]:  # Limit to first 3
            unique_count = self.data[cat_col].nunique()
            
            # Bar chart for value counts
            if unique_count <= 20:  # Reasonable number of categories
                value_counts = self.data[cat_col].value_counts()
                imbalance = value_counts.iloc[0] / value_counts.sum() if len(value_counts) > 0 else 0
                
                recommendations.append({
                    'type': 'bar',
                    'column': cat_col,
                    'title': f'Distribution of {cat_col}',
                    'priority': 'high' if imbalance > 0.7 else 'medium',
                    'reason': f'Show category distribution (imbalance ratio: {imbalance:.2f})'
                })
            
            # Box plots for categorical vs numeric
            for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                recommendations.append({
                    'type': 'grouped_box',
                    'x_column': cat_col,
                    'y_column': num_col,
                    'title': f'{num_col} by {cat_col}',
                    'priority': 'medium',
                    'reason': f'Compare {num_col} across {cat_col} groups'
                })
        
        return recommendations
    
    def _recommend_temporal_plots(self) -> List[Dict]:
        """Recommend temporal visualizations"""
        recommendations = []
        
        # Look for date/time columns
        date_cols = []
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'year', 'month']):
                date_cols.append(col)
        
        date_cols.extend(self.data.select_dtypes(include=['datetime64']).columns.tolist())
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        for date_col in date_cols[:2]:  # Limit to first 2 date columns
            for num_col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                recommendations.append({
                    'type': 'line',
                    'x_column': date_col,
                    'y_column': num_col,
                    'title': f'{num_col} Over Time ({date_col})',
                    'priority': 'high',
                    'reason': 'Identify temporal trends and patterns'
                })
        
        return recommendations
    
    def _recommend_multivariate_plots(self) -> List[Dict]:
        """Recommend multivariate visualizations"""
        recommendations = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Pair plot for numeric variables
        if len(numeric_cols) >= 3:
            recommendations.append({
                'type': 'pairplot',
                'columns': numeric_cols[:4],  # Limit for performance
                'title': 'Pairwise Relationships',
                'priority': 'medium',
                'reason': 'Comprehensive view of variable relationships'
            })
        
        # 3D scatter if we have enough numeric columns
        if len(numeric_cols) >= 3:
            recommendations.append({
                'type': 'scatter_3d',
                'x_column': numeric_cols[0],
                'y_column': numeric_cols[1],
                'z_column': numeric_cols[2],
                'color_column': categorical_cols[0] if categorical_cols else None,
                'title': '3D Scatter Plot',
                'priority': 'low',
                'reason': 'Explore three-dimensional relationships'
            })
        
        return recommendations
    
    def _create_visualization(self, recommendation: Dict) -> Optional[go.Figure]:
        """Create a visualization based on recommendation"""
        viz_type = recommendation['type']
        
        try:
            if viz_type == 'histogram':
                return self._create_histogram(recommendation)
            elif viz_type == 'box':
                return self._create_box_plot(recommendation)
            elif viz_type == 'correlation_heatmap':
                return self._create_correlation_heatmap(recommendation)
            elif viz_type == 'scatter':
                return self._create_scatter_plot(recommendation)
            elif viz_type == 'bar':
                return self._create_bar_chart(recommendation)
            elif viz_type == 'grouped_box':
                return self._create_grouped_box_plot(recommendation)
            elif viz_type == 'line':
                return self._create_line_plot(recommendation)
            elif viz_type == 'scatter_3d':
                return self._create_3d_scatter(recommendation)
            else:
                return None
                
        except Exception as e:
            print(f"Error creating {viz_type}: {str(e)}")
            return None
    
    def _create_histogram(self, rec: Dict) -> go.Figure:
        """Create histogram"""
        col = rec['column']
        fig = px.histogram(
            self.data, 
            x=col, 
            title=rec['title'],
            marginal='box'  # Add box plot on top
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title='Frequency'
        )
        return fig
    
    def _create_box_plot(self, rec: Dict) -> go.Figure:
        """Create box plot"""
        col = rec['column']
        fig = px.box(
            self.data,
            y=col,
            title=rec['title']
        )
        fig.update_layout(yaxis_title=col)
        return fig
    
    def _create_correlation_heatmap(self, rec: Dict) -> go.Figure:
        """Create correlation heatmap"""
        cols = rec['columns']
        corr_matrix = self.data[cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title=rec['title'],
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(
            xaxis_title='Variables',
            yaxis_title='Variables'
        )
        return fig
    
    def _create_scatter_plot(self, rec: Dict) -> go.Figure:
        """Create scatter plot"""
        fig = px.scatter(
            self.data,
            x=rec['x_column'],
            y=rec['y_column'],
            title=rec['title']
        )
        
        # Add trend line
        fig.update_traces(mode='markers+lines')
        return fig
    
    def _create_bar_chart(self, rec: Dict) -> go.Figure:
        """Create bar chart"""
        col = rec['column']
        value_counts = self.data[col].value_counts().head(20)  # Top 20 categories
        
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=rec['title']
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title='Count'
        )
        return fig
    
    def _create_grouped_box_plot(self, rec: Dict) -> go.Figure:
        """Create grouped box plot"""
        fig = px.box(
            self.data,
            x=rec['x_column'],
            y=rec['y_column'],
            title=rec['title']
        )
        return fig
    
    def _create_line_plot(self, rec: Dict) -> go.Figure:
        """Create line plot"""
        # Try to convert x-axis to datetime if it's not already
        data_copy = self.data.copy()
        x_col = rec['x_column']
        
        try:
            if data_copy[x_col].dtype == 'object':
                data_copy[x_col] = pd.to_datetime(data_copy[x_col])
        except:
            pass  # If conversion fails, use original data
        
        fig = px.line(
            data_copy,
            x=rec['x_column'],
            y=rec['y_column'],
            title=rec['title']
        )
        return fig
    
    def _create_3d_scatter(self, rec: Dict) -> go.Figure:
        """Create 3D scatter plot"""
        color_col = rec.get('color_column')
        
        if color_col:
            fig = px.scatter_3d(
                self.data,
                x=rec['x_column'],
                y=rec['y_column'],
                z=rec['z_column'],
                color=color_col,
                title=rec['title']
            )
        else:
            fig = px.scatter_3d(
                self.data,
                x=rec['x_column'],
                y=rec['y_column'],
                z=rec['z_column'],
                title=rec['title']
            )
        
        return fig
    
    def create_custom_visualization(self, viz_config: Dict) -> go.Figure:
        """
        Create custom visualization based on user configuration
        
        Args:
            viz_config: Configuration dictionary with visualization parameters
            
        Returns:
            Plotly figure
        """
        viz_type = viz_config.get('type', 'scatter')
        
        if viz_type == 'correlation_matrix':
            numeric_data = self.data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
            
            fig = px.imshow(
                corr_matrix,
                title=viz_config.get('title', 'Correlation Matrix'),
                color_continuous_scale=viz_config.get('colorscale', 'RdBu_r')
            )
            
        elif viz_type == 'distribution_grid':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            n_cols = min(len(numeric_cols), 3)
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=numeric_cols,
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(numeric_cols):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(x=self.data[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title_text=viz_config.get('title', 'Distribution Grid'),
                height=300 * n_rows
            )
            
        else:
            # Default to scatter plot
            x_col = viz_config.get('x', self.data.columns[0])
            y_col = viz_config.get('y', self.data.columns[1] if len(self.data.columns) > 1 else self.data.columns[0])
            
            fig = px.scatter(
                self.data,
                x=x_col,
                y=y_col,
                title=viz_config.get('title', f'{y_col} vs {x_col}'),
                color=viz_config.get('color'),
                size=viz_config.get('size')
            )
        
        return fig