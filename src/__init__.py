"""
AI Data Cleaning and EDA Agent

An intelligent system for automated data preprocessing, quality assessment,
and exploratory data analysis powered by machine learning.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_cleaning_agent import DataCleaningAgent
from .eda_agent import EDAAgent
from .preprocessing_agent import PreprocessingAgent
from .visualization_agent import VisualizationAgent

__all__ = [
    'DataCleaningAgent',
    'EDAAgent', 
    'PreprocessingAgent',
    'VisualizationAgent'
]