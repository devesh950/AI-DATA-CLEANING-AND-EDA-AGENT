# AI Data Cleaning and EDA Agent with ML Pipeline

## Complete Machine Learning Workflow Example

This notebook demonstrates the full capabilities of the AI Data Cleaning and EDA Agent, including the new ML Pipeline features.

```python
# Import libraries
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import pandas as pd
import numpy as np
from src.data_cleaning_agent import DataCleaningAgent
from src.eda_agent import EDAAgent
from src.preprocessing_agent import PreprocessingAgent
from src.ml_model_agent import MLModelAgent
from src.visualization_agent import VisualizationAgent

# Load sample data
from sklearn.datasets import load_wine, load_boston, load_iris
import warnings
warnings.filterwarnings('ignore')
```

## 1. Data Loading and Initial Exploration

```python
# Load the Wine Quality dataset for demonstration
wine_data = load_wine()
df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
df['target'] = wine_data.target

print(f"Dataset shape: {df.shape}")
print(f"Target classes: {np.unique(df['target'])}")
df.head()
```

## 2. Data Quality Assessment

```python
# Initialize Data Cleaning Agent
cleaning_agent = DataCleaningAgent()
cleaning_agent.load_data(df)

# Perform comprehensive data quality assessment
quality_report = cleaning_agent.assess_data_quality()

print("=== Data Quality Report ===")
print(f"Overall Quality Score: {quality_report['overall_score']:.2f}/100")
print(f"Total Issues Found: {quality_report['total_issues']}")

# Display quality metrics
for metric, score in quality_report['quality_metrics'].items():
    print(f"{metric.replace('_', ' ').title()}: {score:.2f}")
```

## 3. Automated Data Cleaning

```python
# Get cleaning recommendations
recommendations = cleaning_agent.get_cleaning_recommendations()

print("=== Cleaning Recommendations ===")
for rec in recommendations:
    priority = "🔴 HIGH" if rec['priority'] == 'high' else "🟡 MEDIUM" if rec['priority'] == 'medium' else "🔵 LOW"
    print(f"{priority}: {rec['issue']} - {rec['recommendation']}")

# Apply automated cleaning
cleaned_data = cleaning_agent.auto_clean_data()
print(f"\nCleaned data shape: {cleaned_data.shape}")
```

## 4. Exploratory Data Analysis

```python
# Initialize EDA Agent
eda_agent = EDAAgent()
eda_agent.set_data(cleaned_data)

# Generate comprehensive EDA report
eda_report = eda_agent.generate_comprehensive_report()

print("=== EDA Summary ===")
print(f"Dataset Type: {eda_report['dataset_summary']['dataset_type']}")
print(f"Quality Score: {eda_report['dataset_summary']['data_quality_score']}")

# Statistical insights
print("\n=== Key Statistical Insights ===")
for insight in eda_report['statistical_insights']:
    print(f"• {insight['insight']}")

# Feature relationships
print(f"\n=== Feature Relationships ===")
print(f"Strong Correlations Found: {len(eda_report['correlation_analysis']['strong_correlations'])}")
```

## 5. Advanced Preprocessing Pipeline

```python
# Initialize Preprocessing Agent
preprocessing_agent = PreprocessingAgent()
preprocessing_agent.set_data(cleaned_data, target_column='target')

# Analyze preprocessing requirements
preprocessing_analysis = preprocessing_agent.analyze_preprocessing_needs()

print("=== Preprocessing Analysis ===")
print(f"Scaling needed: {preprocessing_analysis['scaling_needs']['scaling_needed']}")
print(f"Feature selection needed: {preprocessing_analysis['feature_selection']['feature_selection_needed']}")

# Automated preprocessing
preprocessed_result = preprocessing_agent.auto_preprocess(target_column='target')

print(f"\nOriginal features: {df.shape[1]-1}")
print(f"Final features: {preprocessed_result['X_train'].shape[1]}")
print(f"Training samples: {preprocessed_result['X_train'].shape[0]}")
print(f"Test samples: {preprocessed_result['X_test'].shape[0]}")
```

## 6. Machine Learning Pipeline

```python
# Initialize ML Model Agent
ml_agent = MLModelAgent()
ml_agent.set_data(
    preprocessed_result['X_train'],
    preprocessed_result['X_test'],
    preprocessed_result['y_train'],
    preprocessed_result['y_test']
)

# Analyze ML requirements
ml_analysis = ml_agent.analyze_ml_requirements()

print("=== ML Analysis ===")
print(f"Problem Type: {ml_analysis['problem_type']}")
print(f"Dataset Size: {ml_analysis['dataset_info']['data_size_category']}")
print(f"Recommended Models: {', '.join(ml_analysis['recommended_models'])}")
```

## 7. Automated Model Training and Selection

```python
# Run Auto-ML Pipeline
print("Running Auto-ML Pipeline...")
auto_ml_report = ml_agent.auto_ml_pipeline(quick_mode=True)

# Display results
best_model = auto_ml_report['best_model']
print(f"\n🏆 Best Model: {best_model['model_name']}")

# Performance metrics
test_metrics = best_model['performance']['test']
for metric, value in test_metrics.items():
    print(f"{metric.title()}: {value:.4f}")

print(f"Cross-validation: {best_model['cv_performance']['mean']:.4f} ± {best_model['cv_performance']['std']:.4f}")
```

## 8. Model Comparison and Analysis

```python
# Create model comparison
comparison_df = auto_ml_report['model_comparison']
print("=== Model Performance Comparison ===")
print(comparison_df.round(4))

# Show recommendations
print("\n=== ML Recommendations ===")
for rec in auto_ml_report['recommendations']:
    priority = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🔵"
    print(f"{priority} {rec['title']}: {rec['description']}")
```

## 9. Intelligent Visualizations

```python
# Initialize Visualization Agent
viz_agent = VisualizationAgent()
viz_agent.set_data(cleaned_data)

# Generate intelligent visualization recommendations
viz_recommendations = viz_agent.recommend_visualizations()

print("=== Visualization Recommendations ===")
for rec in viz_recommendations:
    print(f"• {rec['chart_type']}: {rec['description']}")
    print(f"  Columns: {rec['columns']}")
    print(f"  Reasoning: {rec['reasoning']}")
    print()

# Create automated dashboard
dashboard_plots = viz_agent.create_automated_dashboard(max_plots=6)
print(f"\n📊 Generated {len(dashboard_plots)} plots for automated dashboard")
```

## 10. Complete Workflow Summary

```python
# Summary of the entire pipeline
print("=== COMPLETE AI DATA PIPELINE SUMMARY ===")
print(f"✅ Data Quality Assessment: {quality_report['overall_score']:.1f}/100")
print(f"✅ Data Cleaning: {len(recommendations)} issues addressed")
print(f"✅ EDA Analysis: {len(eda_report['statistical_insights'])} insights generated")
print(f"✅ Preprocessing: {len(preprocessed_result['preprocessing_steps'])} steps applied")
print(f"✅ ML Pipeline: {len(auto_ml_report['model_results'])} models trained")
print(f"✅ Best Model: {best_model['model_name']} ({list(test_metrics.keys())[0]}: {list(test_metrics.values())[0]:.3f})")
print(f"✅ Visualizations: {len(dashboard_plots)} intelligent plots created")

print("\n🎉 AI-Powered Data Science Pipeline Complete!")
print("Your data has been automatically:")
print("  • Assessed for quality issues")
print("  • Cleaned and preprocessed")
print("  • Analyzed for insights")
print("  • Prepared for machine learning")
print("  • Trained with multiple ML models")
print("  • Visualized with intelligent recommendations")
```

## Key Features Demonstrated:

### 🤖 **AI-Powered Automation**
- Intelligent data quality assessment
- Automated cleaning recommendations
- Smart preprocessing pipeline
- Intelligent model selection

### 📊 **Comprehensive Analysis**
- Statistical insights generation
- Correlation and distribution analysis
- Feature importance analysis
- Automated EDA reporting

### 🔧 **Advanced Preprocessing**
- Missing value handling strategies
- Categorical encoding optimization
- Feature scaling and selection
- Engineered feature creation

### 🎯 **Machine Learning Excellence**
- Multiple model training and comparison
- Hyperparameter optimization
- Cross-validation and evaluation
- Performance visualization

### 📈 **Intelligent Visualizations**
- Context-aware plot recommendations
- Automated dashboard generation
- Interactive visualization creation
- Statistical plot optimization

## Next Steps:
1. **Model Deployment**: Export the best model for production use
2. **Feature Engineering**: Create additional domain-specific features
3. **Hyperparameter Tuning**: Fine-tune the best performing models
4. **Ensemble Methods**: Combine multiple models for better performance
5. **Model Monitoring**: Set up monitoring for production deployment

This notebook demonstrates the complete capabilities of the AI Data Cleaning and EDA Agent, from raw data input to production-ready ML models with comprehensive analysis and visualization.