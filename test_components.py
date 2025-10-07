"""
Quick test script to verify all components work correctly
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.data_cleaning_agent import DataCleaningAgent
    from src.eda_agent import EDAAgent
    from src.preprocessing_agent import PreprocessingAgent
    from src.ml_model_agent import MLModelAgent
    from src.visualization_agent import VisualizationAgent
    
    print("✅ All imports successful!")
    
    # Load sample data
    wine_data = load_wine()
    df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target
    
    print(f"✅ Sample data loaded: {df.shape}")
    
    # Test Data Cleaning Agent
    cleaning_agent = DataCleaningAgent()
    cleaning_agent.load_data(df)
    quality_report = cleaning_agent.assess_data_quality()
    print(f"✅ Data quality assessment: {len(quality_report)} quality metrics analyzed")
    
    # Test EDA Agent
    eda_agent = EDAAgent()
    eda_agent.set_data(df)
    comprehensive_report = eda_agent.generate_comprehensive_report()
    print(f"✅ EDA analysis: {len(comprehensive_report)} analysis sections completed")
    
    # Test Preprocessing Agent
    preprocessing_agent = PreprocessingAgent()
    preprocessing_agent.set_data(df, target_column='target')
    analysis = preprocessing_agent.analyze_preprocessing_needs()
    print(f"✅ Preprocessing analysis: {len(analysis['recommendations'])} recommendations")
    
    # Test ML Model Agent
    # Simple train-test split for testing
    from sklearn.model_selection import train_test_split
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ml_agent = MLModelAgent()
    ml_agent.set_data(X_train, X_test, y_train, y_test)
    ml_analysis = ml_agent.analyze_ml_requirements()
    print(f"✅ ML analysis: {ml_analysis['problem_type']} problem detected")
    
    # Test Visualization Agent
    viz_agent = VisualizationAgent()
    viz_agent.set_data(df)
    recommendations = viz_agent.recommend_visualizations()
    print(f"✅ Visualization recommendations: {len(recommendations)} plots suggested")
    
    print("\n🎉 All components working correctly!")
    print("The AI Data Cleaning and EDA Agent is ready to use!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    print("Please check the component implementation.")