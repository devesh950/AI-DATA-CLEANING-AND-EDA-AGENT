"""
Main Streamlit Application for AI Data Cleaning and EDA Agent
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import tempfile
from typing import Optional
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_cleaning_agent import DataCleaningAgent
from src.eda_agent import EDAAgent
from src.preprocessing_agent import PreprocessingAgent
from src.ml_model_agent import MLModelAgent

# Page configuration
st.set_page_config(
    page_title="AI Data Cleaning & EDA Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for Beautiful & Interactive Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Content Area - Enhanced Text Readability */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.98) !important;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin: 1rem;
    }
    
    /* Global Text Styling for Better Readability */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
        font-weight: 700 !important;
    }
    
    p, span, div, li, label {
        color: #34495e !important;
        font-weight: 500 !important;
    }
    
    /* Streamlit specific overrides */
    .stMarkdown, .stText {
        color: #34495e !important;
    }
    
    /* Welcome Text - Enhanced Visibility */
    .welcome-text {
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        color: #2c3e50 !important;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.8);
        background: rgba(255, 255, 255, 0.9);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        display: inline-block;
        margin: 0 auto;
        width: fit-content;
    }
    
    /* Header Styles - Ultra Bold with High Contrast */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900 !important;
        text-align: center;
        margin-bottom: 1rem;
        color: #2c3e50 !important;
        text-shadow: 0 4px 8px rgba(255, 255, 255, 0.9), 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        background: rgba(255, 255, 255, 0.95) !important;
        padding: 1rem 2rem;
        border-radius: 25px;
        display: inline-block;
        width: fit-content;
        margin: 1rem auto;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        border: 2px solid rgba(102, 126, 234, 0.2);
    }
    
    .main-header strong {
        font-weight: 900 !important;
        color: #2c3e50 !important;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.9) !important;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #2c3e50 !important;
        text-align: center;
        font-weight: 600 !important;
        margin-bottom: 2rem;
        animation: fadeInUp 0.6s ease-out;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.8) !important;
        background: rgba(255, 255, 255, 0.9);
        padding: 0.8rem 1.5rem;
        border-radius: 20px;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar Styling - Enhanced Text Visibility */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 0 20px 20px 0;
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1rem;
    }
    
    /* Sidebar text styling */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4) !important;
        font-weight: 700 !important;
    }
    
    .css-1d391kg p, .css-1d391kg span, .css-1d391kg div, .css-1d391kg label {
        color: white !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        font-weight: 500 !important;
    }
    
    /* Buttons & Interactive Elements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Metric Cards - Enhanced Readability */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%) !important;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card h1, .metric-card h2, .metric-card h3, .metric-card h4, .metric-card h5, .metric-card h6 {
        color: #2c3e50 !important;
        text-shadow: none !important;
        font-weight: 700 !important;
    }
    
    .metric-card p, .metric-card span, .metric-card div, .metric-card li {
        color: #34495e !important;
        text-shadow: none !important;
        font-weight: 500 !important;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Success & Alert Boxes */
    .success-box {
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: none;
        color: #155724;
        box-shadow: 0 8px 25px rgba(212, 237, 218, 0.3);
        animation: slideInLeft 0.5s ease-out;
    }
    
    .warning-box {
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: none;
        color: #856404;
        box-shadow: 0 8px 25px rgba(255, 243, 205, 0.3);
        animation: slideInRight 0.5s ease-out;
    }
    
    .info-box {
        padding: 1.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #cce7ff 0%, #b3d9ff 100%);
        border: none;
        color: #0056b3;
        box-shadow: 0 8px 25px rgba(204, 231, 255, 0.3);
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #667eea;
        font-weight: 600;
        transition: all 0.3s ease;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);
        color: #764ba2;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* File Uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        border: 2px dashed #667eea;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.05);
        transform: scale(1.02);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Selectbox & Input Elements */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Checkbox & Radio */
    .stCheckbox > label {
        color: #333;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        font-weight: 600;
        color: #667eea;
    }
    
    /* Sidebar Elements */
    .css-1d391kg .stSelectbox > div > div,
    .css-1d391kg .stFileUploader,
    .css-1d391kg .stButton > button {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .css-1d391kg .stSelectbox > div > div:hover {
        background: rgba(255, 255, 255, 0.25);
    }
    
    /* Custom Success Message */
    .custom-success {
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
        font-weight: 500;
        margin: 1rem 0;
        animation: slideInLeft 0.5s ease-out;
    }
    
    /* Custom Info Message */
    .custom-info {
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        color: white;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(33, 150, 243, 0.3);
        font-weight: 500;
        margin: 1rem 0;
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Plotly Charts */
    .js-plotly-plot .plotly .modebar {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .main .block-container {
            padding: 1rem;
            margin: 0.5rem;
        }
        
        .stButton > button {
            padding: 0.5rem 1.5rem;
            font-size: 0.9rem;
        }
    }
    
    /* Hover Effects for Cards */
    .hover-lift {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .hover-lift:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    /* Glass Morphism Effect - Enhanced Readability */
    .glass {
        background: rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    }
    
    /* Improved Text Visibility */
    .glass h1, .glass h2, .glass h3, .glass h4, .glass h5, .glass h6 {
        color: #2c3e50 !important;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.9) !important;
        font-weight: 700 !important;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.2em 0.5em;
        border-radius: 8px;
        display: inline-block;
    }
    
    .glass p, .glass span, .glass div, .glass li {
        color: #34495e !important;
        text-shadow: 0 1px 3px rgba(255, 255, 255, 0.8) !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
    }
    
    .glass ul, .glass ol {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Mobile Responsiveness */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 2rem !important;
            padding: 1rem !important;
        }
        
        .subtitle {
            font-size: 1rem !important;
            padding: 0.5rem !important;
        }
        
        .metric-card {
            margin: 0.5rem 0 !important;
            padding: 1rem !important;
            height: auto !important;
            min-height: 250px !important;
        }
        
        .glass {
            margin: 0.5rem !important;
            padding: 1rem !important;
        }
        
        .main .block-container {
            padding: 1rem !important;
            margin: 0.5rem !important;
        }
        
        /* Mobile Sidebar Improvements */
        .css-1d391kg {
            width: 100% !important;
            position: relative !important;
            transform: none !important;
            border-radius: 0 !important;
        }
        
        /* Make sidebar toggle more visible on mobile */
        .css-1rs6os {
            display: block !important;
            position: fixed !important;
            top: 1rem !important;
            left: 1rem !important;
            z-index: 999999 !important;
            background: rgba(102, 126, 234, 0.9) !important;
            border-radius: 50% !important;
            padding: 0.5rem !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Adjust button sizes for mobile */
        .stButton > button {
            width: 100% !important;
            padding: 0.8rem !important;
            font-size: 0.9rem !important;
        }
        
        /* Mobile-friendly charts */
        .js-plotly-plot {
            width: 100% !important;
            height: 300px !important;
        }
        
        /* Responsive text sizes */
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.6rem !important; }
        h3 { font-size: 1.4rem !important; }
        h4 { font-size: 1.2rem !important; }
        
        /* Demo dataset cards responsive */
        .metric-card ul {
            font-size: 0.85rem !important;
        }
        
        /* Getting started section mobile */
        .hover-lift {
            margin: 0.5rem 0 !important;
            min-width: auto !important;
        }
        
        /* Mobile demo loading section */
        .mobile-demo-loader {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        
        /* Mobile responsive welcome and header */
        @media screen and (max-width: 768px) {
            .welcome-text {
                font-size: 1.1rem !important;
                margin-bottom: 0.3rem !important;
                padding: 0.3rem 0.8rem !important;
            }
            
            .main-header {
                font-size: 2rem !important;
                padding: 0.8rem 1rem !important;
                margin: 0.5rem auto !important;
            }
            
            .subtitle {
                font-size: 1rem !important;
                padding: 0.5rem 1rem !important;
                margin-bottom: 1rem !important;
            }
        }
        
        @media screen and (max-width: 480px) {
            .main-header {
                font-size: 1.6rem !important;
                padding: 0.6rem 0.8rem !important;
            }
            
            .subtitle {
                font-size: 0.9rem !important;
                padding: 0.4rem 0.8rem !important;
            }
            
            .welcome-text {
                font-size: 1rem !important;
                padding: 0.2rem 0.6rem !important;
            }
        }
        
        /* Upload info styling */
        .upload-info {
            text-align: center;
            background: rgba(102, 126, 234, 0.1);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 2px dashed rgba(102, 126, 234, 0.3);
        }
        
        .upload-info p {
            margin: 0.5rem 0 !important;
            color: #667eea !important;
            font-weight: 500 !important;
        }
        
        /* Hide default Streamlit file uploader limit message */
        .uploadedFile small {
            display: none !important;
        }
        
        /* Custom file uploader styling */
        .stFileUploader > div {
            border: 2px dashed rgba(102, 126, 234, 0.4) !important;
            border-radius: 15px !important;
            padding: 2rem !important;
            text-align: center !important;
            background: rgba(255, 255, 255, 0.8) !important;
            transition: all 0.3s ease !important;
        }
        
        .stFileUploader > div:hover {
            border-color: rgba(102, 126, 234, 0.6) !important;
            background: rgba(102, 126, 234, 0.05) !important;
        }
        
        /* Mobile file uploader responsive */
        @media screen and (max-width: 768px) {
            .stFileUploader > div {
                padding: 1.5rem 1rem !important;
                margin: 0.5rem 0 !important;
            }
            
            .upload-info {
                padding: 0.8rem !important;
                margin: 0.5rem 0 !important;
            }
            
            .upload-info p {
                font-size: 0.9rem !important;
            }
        }
        
        /* Override Streamlit upload limit text */
        div[data-testid="stFileUploadDropzone"] small,
        div[data-testid="stFileUploader"] small {
            display: none !important;
        }
        
        /* Add custom 2GB limit message */
        div[data-testid="stFileUploadDropzone"]::after {
            content: "Limit: 2GB per file";
            display: block;
            font-size: 0.8rem;
            color: #667eea;
            margin-top: 0.5rem;
            font-weight: 500;
        }
    }
    
    @media screen and (max-width: 480px) {
        .main-header {
            font-size: 1.5rem !important;
            padding: 0.8rem !important;
        }
        
        .subtitle {
            font-size: 0.9rem !important;
        }
        
        .metric-card {
            height: auto !important;
            min-height: 200px !important;
            margin: 0.3rem 0 !important;
        }
        
        /* Ultra-mobile adjustments */
        .glass {
            padding: 0.8rem !important;
            margin: 0.3rem !important;
        }
        
        .main .block-container {
            padding: 0.8rem !important;
        }
        
        /* Mobile demo cards */
        .hover-lift {
            padding: 0.8rem !important;
        }
        
        /* Smaller text for very small screens */
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.2rem !important; }
        h4 { font-size: 1.1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Welcome message and Enhanced Header
    st.markdown('''
        <div style="text-align: center; margin: 2rem 0;">
            <div class="welcome-text">Welcome to the</div>
            <div class="main-header">🤖 <strong>AI Data Cleaning & EDA Agent</strong></div>
            <div class="subtitle">📊 Now Supporting Datasets up to 2GB 🚀</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Initialize session state first (moved up)
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'demo_loaded' not in st.session_state:
        st.session_state.demo_loaded = False
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None
    
    # Back to Home button (show when data is loaded)
    if st.session_state.data is not None or st.session_state.demo_loaded:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🏠 Back to Home", key="back_home", use_container_width=True, help="Return to welcome screen"):
                # Clear all data and return to home
                st.session_state.data = None
                st.session_state.demo_loaded = False
                st.session_state.file_name = None
                if hasattr(st.session_state, 'cleaning_agent'):
                    st.session_state.cleaning_agent.data = None
                if hasattr(st.session_state, 'eda_agent'):
                    st.session_state.eda_agent.data = None
                st.rerun()
    
    # Add some CSS for the back button
    st.markdown('''
        <style>
        /* Back button styling */
        button[data-testid="baseButton-secondary"] {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 0.8rem 2rem !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }
        
        button[data-testid="baseButton-secondary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(108, 117, 125, 0.4) !important;
            background: linear-gradient(135deg, #495057 0%, #343a40 100%) !important;
        }
        
        /* Mobile back button */
        @media screen and (max-width: 768px) {
            button[data-testid="baseButton-secondary"] {
                padding: 1rem 1.5rem !important;
                font-size: 1rem !important;
                margin: 0.5rem 0 !important;
            }
        }
        
        /* Improved mobile sidebar button visibility */
        @media screen and (max-width: 768px) {
            .css-1rs6os .css-1cpxqw2 {
                background-color: rgba(102, 126, 234, 0.9) !important;
                color: white !important;
                border-radius: 50% !important;
                padding: 0.8rem !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.5) !important;
                position: fixed !important;
                top: 1rem !important;
                left: 1rem !important;
                z-index: 999999 !important;
            }
        }
        </style>
    ''', unsafe_allow_html=True)
    
    st.markdown('''
        <div style="text-align: center; margin: 2rem 0;">
            <div class="glass hover-lift" style="display: inline-block; padding: 1rem 2rem; margin: 0.5rem;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #667eea;">
                    ⚡ Intelligent Analysis • 🔧 Auto-Optimization • 📈 ML Pipeline
                </span>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown('''
            <div style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
                <h2 style="color: white; font-weight: 700; margin: 0;">🛠️ Controls</h2>
            </div>
        ''', unsafe_allow_html=True)
        
        # Capacity highlight with custom styling
        st.markdown('''
            <div class="custom-success">
                <div style="text-align: center;">
                    <span style="font-size: 1.1rem; font-weight: 600;">🚀 NEW: 2GB Dataset Support!</span>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        # File upload section with enhanced styling
        st.markdown('''
            <div style="margin: 1.5rem 0;">
                <h3 style="color: white; font-weight: 600; margin-bottom: 1rem;">📁 Data Upload</h3>
            </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced capacity indicator with custom styling
        st.markdown('''
            <div class="metric-card hover-lift" style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: #667eea; font-weight: 600;">📊 Maximum Dataset Size</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;">Memory optimization included</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2rem; font-weight: 700; color: #764ba2;">2GB</div>
                        <div style="font-size: 0.8rem; color: #4CAF50; font-weight: 600;">↗️ 10x increase</div>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Unified upload message
        st.info("📁 **File Upload**: Use the main area below for uploading datasets (supports up to 2GB)")
        
        # Sidebar file uploader (works like main area uploader)
        uploaded_file = st.file_uploader(
            "Upload dataset (sidebar)",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Upload CSV, Excel, JSON, or Parquet files up to 2GB",
            key="sidebar_uploader",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Try to load the uploaded file using the shared loader
            with st.spinner("📊 Loading uploaded file..."):
                data = load_uploaded_file(uploaded_file)

            if data is not None:
                st.session_state.data = data
                st.session_state.file_name = getattr(uploaded_file, 'name', 'uploaded_dataset')
                st.session_state.demo_loaded = False
                # Initialize agents with the loaded data
                if hasattr(st.session_state, 'cleaning_agent'):
                    st.session_state.cleaning_agent.load_data(data)
                else:
                    st.session_state.cleaning_agent = DataCleaningAgent(use_large_dataset_optimization=True, memory_limit_gb=1.5)
                    st.session_state.cleaning_agent.load_data(data)

                if hasattr(st.session_state, 'eda_agent'):
                    st.session_state.eda_agent.set_data(data)
                else:
                    st.session_state.eda_agent = EDAAgent()
                    st.session_state.eda_agent.set_data(data)

                st.success("✅ Dataset uploaded successfully from sidebar!")
                st.balloons()
                st.rerun()
        
        # Demo data option
        use_demo_data = st.checkbox("Use Demo Dataset", help="Load a sample dataset for testing")
        
        if use_demo_data:
            demo_option = st.selectbox(
                "Select Demo Dataset",
                ["Titanic", "Wine Quality", "Boston Housing"]
            )
            
            # Show current status
            if st.session_state.demo_loaded and st.session_state.data is not None:
                st.success(f"✅ Dataset loaded: {st.session_state.file_name}")
                if st.button("Clear Demo Data", key="clear_demo", use_container_width=True):
                    st.session_state.data = None
                    st.session_state.demo_loaded = False
                    st.session_state.file_name = None
                    st.rerun()
            else:
                # Add load button in sidebar
                if st.button("Load Selected Dataset", key="sidebar_load_demo", use_container_width=True):
                    loaded_data = load_demo_dataset(demo_option)
                    if loaded_data is not None:
                        st.success(f"✅ {demo_option} dataset loaded!")
                        st.balloons()
                        st.rerun()
    
    # Initialize session state with large dataset support
    if 'cleaning_agent' not in st.session_state:
        # Initialize with large dataset optimization
        st.session_state.cleaning_agent = DataCleaningAgent(
            use_large_dataset_optimization=True,
            memory_limit_gb=1.5
        )
    if 'eda_agent' not in st.session_state:
        st.session_state.eda_agent = EDAAgent()
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = {}
    
    # Load data (from session state - main uploader handles the loading)
    data = st.session_state.data
    
    # Show analysis interface if ANY data is loaded (demo or uploaded)
    if st.session_state.data is not None:
        # Handle previously loaded data (demo or uploaded)
        data = st.session_state.data
        if not hasattr(st.session_state.cleaning_agent, 'data') or st.session_state.cleaning_agent.data is None:
            st.session_state.cleaning_agent.load_data(data)
            st.session_state.eda_agent.set_data(data)
        
        # Back to Home button in main analysis area
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🏠 Back to Home", key="analysis_back_home", use_container_width=True):
                # Clear all data and return to home
                st.session_state.data = None
                st.session_state.demo_loaded = False
                st.session_state.file_name = None
                if hasattr(st.session_state, 'cleaning_agent'):
                    st.session_state.cleaning_agent.data = None
                if hasattr(st.session_state, 'eda_agent'):
                    st.session_state.eda_agent.data = None
                st.rerun()
        
        st.markdown("---")
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Data Overview", 
            "🧹 Data Cleaning", 
            "🔍 EDA Analysis", 
            "📈 Visualizations", 
            "🤖 ML Pipeline",
            "📋 Reports"
        ])
        
        with tab1:
            show_data_overview(data)
        
        with tab2:
            show_data_cleaning_interface()
        
        with tab3:
            show_eda_analysis()
        
        with tab4:
            show_visualizations(data)
        
        with tab5:
            show_ml_pipeline(data)
        
        with tab6:
            show_reports()
    
    else:
        # Unified data loading section (works for all devices)
        # Show welcome/upload screen only if no data is loaded
        if st.session_state.data is None:
            
            # Universal file uploader with custom styling
            st.markdown("### 📁 Upload Your Dataset")
            
            st.markdown('''
                <div class="upload-info">
                    <p>📊 <strong>Drag & Drop or Browse</strong> your data file</p>
                    <p>✅ Supports up to <strong>2GB</strong> • CSV, Excel, JSON, Parquet</p>
                </div>
            ''', unsafe_allow_html=True)
            
            uploaded_file_main = st.file_uploader(
                "Choose file",
                type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
                help="Upload CSV, Excel, JSON, or Parquet files up to 2GB",
                key="main_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file_main is not None:
                data = load_uploaded_file(uploaded_file_main)
                if data is not None:
                    st.session_state.data = data
                    st.session_state.file_name = uploaded_file_main.name
                    st.session_state.demo_loaded = False  # Mark as uploaded (not demo)
                    st.session_state.cleaning_agent.load_data(data)
                    st.session_state.eda_agent.set_data(data)
                    st.success("✅ Dataset uploaded successfully!")
                    st.balloons()
                    st.rerun()
            
            st.markdown("---")
            
            # Demo datasets section
            st.markdown("### 📊 Or Try Demo Datasets")
            
            # Mobile demo selector
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚢 Titanic", key="mobile_titanic", use_container_width=True, help="Load Titanic survival dataset"):
                    loaded_data = load_demo_dataset("Titanic")
                    if loaded_data is not None:
                        st.success("✅ Titanic dataset loaded!")
                        st.balloons()
                        st.rerun()
                        
            with col2:
                if st.button("🍷 Wine Quality", key="mobile_wine", use_container_width=True, help="Load Wine Quality dataset"):
                    loaded_data = load_demo_dataset("Wine Quality")
                    if loaded_data is not None:
                        st.success("✅ Wine Quality dataset loaded!")
                        st.balloons()
                        st.rerun()
                        
            with col3:
                if st.button("🏠 Boston Housing", key="mobile_housing", use_container_width=True, help="Load Boston Housing dataset"):
                    loaded_data = load_demo_dataset("Boston Housing")
                    if loaded_data is not None:
                        st.success("✅ Boston Housing dataset loaded!")
                        st.balloons()
                        st.rerun()
                        
            st.markdown("---")
        
        # Simple demo status (sidebar method)
        if use_demo_data and not st.session_state.demo_loaded:
            st.info(f"📋 **{demo_option}** selected - Click 'Load Selected Dataset' in sidebar")
            
            # Show preview of what will be loaded
            st.markdown("### 🎯 Dataset Preview")
            
            col1, col2, col3 = st.columns(3)
            
            if demo_option == "Titanic":
                with col1:
                    st.metric("Dataset", "Titanic")
                    st.metric("Rows", "100")
                with col2:
                    st.metric("Columns", "9")
                    st.metric("Type", "Classification")
                with col3:
                    st.metric("Target", "Survival")
                    st.metric("Features", "Passenger Info")
                    
                st.markdown("**Features**: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked")
                
            elif demo_option == "Wine Quality":
                with col1:
                    st.metric("Dataset", "Wine Quality")
                    st.metric("Rows", "100")
                with col2:
                    st.metric("Columns", "8")
                    st.metric("Type", "Regression")
                with col3:
                    st.metric("Target", "Quality Score")
                    st.metric("Features", "Chemical Properties")
                    
                st.markdown("**Features**: fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, pH, alcohol, quality")
                
            elif demo_option == "Boston Housing":
                with col1:
                    st.metric("Dataset", "Boston Housing")
                    st.metric("Rows", "100")
                with col2:
                    st.metric("Columns", "12")
                    st.metric("Type", "Regression")
                with col3:
                    st.metric("Target", "House Price")
                    st.metric("Features", "Property Info")
                    
                st.markdown("**Features**: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, TAX, PTRATIO, LSTAT, MEDV")
            
            st.markdown("---")
            
        # Welcome screen
        show_welcome_screen()

def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load uploaded file into DataFrame with large file support"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
        uploaded_file.seek(0)  # Reset file pointer
        
        # Show loading progress for large files
        if file_size_mb > 100:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text(f"Loading large file ({file_size_mb:.1f} MB)...")
            progress_bar.progress(25)
        
        # Use optimized loading for large files
        if file_size_mb > 200 and hasattr(st.session_state, 'cleaning_agent'):
            # Use the large dataset handler for files > 200MB
            import tempfile
            import os
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            if file_size_mb > 100:
                progress_bar.progress(50)
                status_text.text("Applying memory optimizations...")
            
            try:
                # Use DataCleaningAgent's optimized loading
                st.session_state.cleaning_agent.load_data(tmp_file_path)
                data = st.session_state.cleaning_agent.data
                
                if file_size_mb > 100:
                    progress_bar.progress(100)
                    status_text.text("Loading complete!")
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        else:
            # Standard loading for smaller files
            if file_extension == 'csv':
                data = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                data = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                data = pd.read_json(uploaded_file)
            elif file_extension == 'parquet':
                data = pd.read_parquet(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return None
        
        # Clear progress indicators
        if file_size_mb > 100:
            progress_bar.empty()
            status_text.empty()
        
        # Show success message with file info
        memory_usage = data.memory_usage(deep=True).sum() / (1024 * 1024)
        st.success(f"✅ Successfully loaded {data.shape[0]:,} rows and {data.shape[1]} columns")
        st.info(f"📊 Memory usage: {memory_usage:.1f} MB | File size: {file_size_mb:.1f} MB")
        
        # Show optimization info for large files
        if file_size_mb > 200:
            memory_reduction = (1 - memory_usage / file_size_mb) * 100
            if memory_reduction > 0:
                st.success(f"🚀 Memory optimization achieved: {memory_reduction:.1f}% reduction!")
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.error("💡 Tip: For very large files, try converting to Parquet format first")
        return None

def load_demo_dataset(dataset_name: str) -> Optional[pd.DataFrame]:
    """Load demo dataset"""
    try:
        if dataset_name == "Iris Dataset":
            from sklearn.datasets import load_iris
            iris = load_iris()
            data = pd.DataFrame(iris.data, columns=iris.feature_names)
            data['species'] = iris.target
            
        elif dataset_name == "Titanic Dataset":
            # Create a simple titanic-like dataset
            np.random.seed(42)
            n_samples = 800
            data = pd.DataFrame({
                'age': np.random.normal(30, 12, n_samples),
                'fare': np.random.lognormal(3, 1, n_samples),
                'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.3, 0.3, 0.4]),
                'sex': np.random.choice(['male', 'female'], n_samples),
                'embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
                'survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
            })
            # Add some missing values
            missing_indices = np.random.choice(n_samples, int(0.1 * n_samples), replace=False)
            data.loc[missing_indices, 'age'] = np.nan
            
        elif dataset_name == "Boston Housing":
            from sklearn.datasets import load_boston
            boston = load_boston()
            data = pd.DataFrame(boston.data, columns=boston.feature_names)
            data['price'] = boston.target
            
        elif dataset_name == "Wine Quality":
            # Create synthetic wine quality data
            np.random.seed(42)
            n_samples = 1000
            data = pd.DataFrame({
                'alcohol': np.random.normal(10.5, 1.5, n_samples),
                'acidity': np.random.normal(0.5, 0.2, n_samples),
                'residual_sugar': np.random.lognormal(1, 0.5, n_samples),
                'pH': np.random.normal(3.2, 0.3, n_samples),
                'density': np.random.normal(0.995, 0.003, n_samples),
                'quality': np.random.choice(range(3, 9), n_samples)
            })
        
        st.info(f"📊 Loaded demo dataset: {dataset_name}")
        return data
        
    except Exception as e:
        st.error(f"❌ Error loading demo dataset: {str(e)}")
        return None

def load_demo_dataset(dataset_name):
    """Load demo dataset based on name"""
    try:
        # Create demo data directly in memory to avoid file system issues
        if dataset_name == "Titanic":
            # Create consistent 100-row Titanic dataset
            demo_data = pd.DataFrame({
                'PassengerId': list(range(1, 101)),
                'Survived': ([0, 1, 1, 1, 0, 0, 0, 0, 1, 1] * 10),
                'Pclass': ([3, 1, 3, 1, 3, 3, 1, 3, 3, 2] * 10),
                'Name': ([
                    'Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina',
                    'Futrelle, Mrs. Jacques Heath', 'Allen, Mr. William Henry', 'Moran, Mr. James',
                    'McCarthy, Mr. Timothy J', 'Palsson, Master. Gosta Leonard', 'Johnson, Mrs. Oscar W',
                    'Nasser, Mrs. Nicholas'
                ] * 10),
                'Sex': (['male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female'] * 10),
                'Age': ([22, 38, 26, 35, 35, 28, 54, 2, 27, 14] * 10),
                'SibSp': ([1, 1, 0, 1, 0, 0, 0, 3, 0, 1] * 10),
                'Parch': ([0, 0, 0, 0, 0, 0, 0, 1, 2, 0] * 10),
                'Fare': ([7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07, 11.13, 30.07] * 10),
                'Embarked': (['S', 'C', 'S', 'S', 'S', 'Q', 'S', 'S', 'S', 'C'] * 10)
            })
        elif dataset_name == "Wine Quality":
            demo_data = pd.DataFrame({
                'fixed_acidity': [7.4, 7.8, 7.8, 11.2, 7.4, 7.4, 7.9, 7.3, 7.8, 7.5] * 10,
                'volatile_acidity': [0.7, 0.88, 0.76, 0.28, 0.7, 0.66, 0.6, 0.65, 0.58, 0.5] * 10,
                'citric_acid': [0.0, 0.0, 0.04, 0.56, 0.0, 0.0, 0.06, 0.0, 0.02, 0.36] * 10,
                'residual_sugar': [1.9, 2.6, 2.3, 1.9, 1.9, 1.8, 1.6, 1.2, 2.0, 6.1] * 10,
                'chlorides': [0.076, 0.098, 0.092, 0.075, 0.076, 0.075, 0.069, 0.065, 0.073, 0.071] * 10,
                'pH': [3.51, 3.2, 3.26, 3.16, 3.51, 3.51, 3.3, 3.39, 3.36, 3.35] * 10,
                'alcohol': [9.4, 9.8, 9.8, 9.8, 9.4, 9.4, 9.4, 10.0, 9.5, 10.5] * 10,
                'quality': [5, 5, 5, 6, 5, 5, 5, 7, 7, 5] * 10
            })
        elif dataset_name == "Boston Housing":
            demo_data = pd.DataFrame({
                'CRIM': [0.00632, 0.02731, 0.02729, 0.03237, 0.06905, 0.02985, 0.08829, 0.14455, 0.21124, 0.17004] * 10,
                'ZN': [18.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.5, 12.5, 12.5, 12.5] * 10,
                'INDUS': [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87] * 10,
                'CHAS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 10,
                'NOX': [0.538, 0.469, 0.469, 0.458, 0.458, 0.458, 0.524, 0.524, 0.524, 0.524] * 10,
                'RM': [6.575, 6.421, 7.185, 6.998, 7.147, 6.430, 6.012, 6.172, 5.631, 6.004] * 10,
                'AGE': [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 66.6, 96.1, 100.0, 85.9] * 10,
                'DIS': [4.0900, 4.9671, 4.9671, 6.0622, 6.0622, 6.0622, 5.5605, 5.9505, 6.0821, 6.5921] * 10,
                'TAX': [296, 242, 242, 222, 222, 222, 311, 311, 311, 311] * 10,
                'PTRATIO': [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2] * 10,
                'LSTAT': [4.98, 9.14, 4.03, 2.94, 5.33, 5.21, 12.43, 19.15, 29.93, 17.10] * 10,
                'MEDV': [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 22.9, 27.1, 16.5, 18.9] * 10
            })
        else:
            return None
            
        # Store in session state
        st.session_state.data = demo_data
        st.session_state.file_name = f"{dataset_name.lower().replace(' ', '_')}_demo.csv"
        st.session_state.demo_loaded = True
        
        return demo_data
        
    except Exception as e:
        st.error(f"Error creating demo dataset: {str(e)}")
        return None

def show_welcome_screen():
    """Show enhanced welcome screen with beautiful styling"""
    
    # Hero Section
    st.markdown('''
        <div style="text-align: center; padding: 2rem 0;">
            <div class="glass hover-lift" style="padding: 2rem; margin: 1rem 0; background: rgba(255, 255, 255, 0.9) !important;">
                <h2 style="color: #2c3e50 !important; font-weight: 700; margin-bottom: 1rem; text-shadow: none !important;">
                    👋 Welcome to the AI Data Cleaning & EDA Agent!
                </h2>
                <div class="custom-success" style="margin: 1rem 0; display: inline-block;">
                    <span style="font-size: 1.2rem; font-weight: 600; color: white !important;">
                        🚀 NEW: Now Supporting Datasets up to 2GB! 📊
                    </span>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Feature Cards - Responsive layout
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    with col1:
        st.markdown('''
            <div class="metric-card hover-lift" style="height: 280px; background: rgba(255, 255, 255, 0.95) !important;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🧹</div>
                    <h3 style="color: #2c3e50 !important; font-weight: 600; margin-bottom: 1rem;">Automated Data Cleaning</h3>
                    <ul style="text-align: left; color: #34495e !important; line-height: 1.6; font-weight: 500;">
                        <li>Smart missing value detection</li>
                        <li>Outlier identification & handling</li>
                        <li>Data type optimization</li>
                        <li>Duplicate detection</li>
                        <li>90% memory reduction</li>
                    </ul>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
            <div class="metric-card hover-lift" style="height: 280px; background: rgba(255, 255, 255, 0.95) !important;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
                    <h3 style="color: #2c3e50 !important; font-weight: 600; margin-bottom: 1rem;">AI-Powered EDA</h3>
                    <ul style="text-align: left; color: #34495e !important; line-height: 1.6; font-weight: 500;">
                        <li>Comprehensive statistical analysis</li>
                        <li>Intelligent visualizations</li>
                        <li>Pattern & anomaly detection</li>
                        <li>Feature importance analysis</li>
                        <li>Automated insights</li>
                    </ul>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
            <div class="metric-card hover-lift" style="height: 280px; background: rgba(255, 255, 255, 0.95) !important;">
                <div style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🤖</div>
                    <h3 style="color: #2c3e50 !important; font-weight: 600; margin-bottom: 1rem;">ML Pipeline</h3>
                    <ul style="text-align: left; color: #34495e !important; line-height: 1.6; font-weight: 500;">
                        <li>Automated preprocessing</li>
                        <li>Multi-model training</li>
                        <li>Performance comparison</li>
                        <li>Auto-ML workflow</li>
                        <li>Production-ready models</li>
                    </ul>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown('''
        <div style="margin: 3rem 0;">
            <div class="glass" style="padding: 2rem; text-align: center; background: rgba(255, 255, 255, 0.9) !important;">
                <h3 style="color: #2c3e50 !important; font-weight: 600; margin-bottom: 2rem; text-shadow: none !important;">🚀 Get Started in 3 Easy Steps</h3>
                <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                    <div class="hover-lift" style="margin: 1rem; padding: 1rem; background: rgba(102, 126, 234, 0.15); border-radius: 15px; min-width: 200px;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📤</div>
                        <h4 style="color: #2c3e50 !important; margin: 0.5rem 0; font-weight: 600;">Upload Dataset</h4>
                        <p style="color: #34495e !important; font-size: 0.9rem; margin: 0; font-weight: 500;">Up to 2GB supported</p>
                    </div>
                    <div class="hover-lift" style="margin: 1rem; padding: 1rem; background: rgba(118, 75, 162, 0.15); border-radius: 15px; min-width: 200px;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">⚡</div>
                        <h4 style="color: #2c3e50 !important; margin: 0.5rem 0; font-weight: 600;">Auto Analysis</h4>
                        <p style="color: #34495e !important; font-size: 0.9rem; margin: 0; font-weight: 500;">Intelligent processing</p>
                    </div>
                    <div class="hover-lift" style="margin: 1rem; padding: 1rem; background: rgba(76, 175, 80, 0.15); border-radius: 15px; min-width: 200px;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>
                        <h4 style="color: #2c3e50 !important; margin: 0.5rem 0; font-weight: 600;">Explore Results</h4>
                        <p style="color: #34495e !important; font-size: 0.9rem; margin: 0; font-weight: 500;">Interactive insights</p>
                    </div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Performance Highlights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('''
            <div class="custom-info">
                <h4 style="margin: 0 0 0.5rem 0; color: white;">� Pro Tip</h4>
                <p style="margin: 0; font-size: 0.9rem;">Large datasets (>200MB) are automatically optimized for better performance with up to 90% memory reduction!</p>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
            <div class="custom-success">
                <h4 style="margin: 0 0 0.5rem 0; color: white;">🎯 Perfect For</h4>
                <p style="margin: 0; font-size: 0.9rem;">Amazon ML Challenges • Kaggle Competitions • Enterprise Data • Research Projects</p>
            </div>
        ''', unsafe_allow_html=True)
    
    # Demo datasets section
    st.markdown('''
        <div style="margin: 2rem 0;">
            <h3 style="text-align: center; color: #667eea; font-weight: 600; margin-bottom: 1.5rem;">🎮 Try Demo Datasets</h3>
        </div>
    ''', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    demo_datasets = {
        "Titanic": {
            "description": "Classic ML dataset for survival prediction",
            "size": "60KB",
            "features": "12 columns, 891 rows",
            "emoji": "🚢"
        },
        "Wine Quality": {
            "description": "Wine quality prediction dataset", 
            "size": "240KB",
            "features": "12 columns, 6497 rows",
            "emoji": "🍷"
        },
        "Boston Housing": {
            "description": "Real estate price prediction",
            "size": "33KB", 
            "features": "14 columns, 506 rows",
            "emoji": "🏠"
        }
    }
    
    for i, (name, info) in enumerate(demo_datasets.items()):
        col = [col1, col2, col3][i]
        with col:
            st.markdown(f'''
                <div class="metric-card hover-lift" style="height: 200px; text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.95) !important;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{info['emoji']}</div>
                    <h4 style="color: #2c3e50 !important; margin: 0.5rem 0; font-weight: 600;">{name}</h4>
                    <p style="color: #34495e !important; font-size: 0.85rem; line-height: 1.4; margin: 0.5rem 0; font-weight: 500;">{info['description']}</p>
                    <div style="margin-top: 1rem;">
                        <small style="color: #5a6c7d !important; font-weight: 500;">📊 {info['features']}</small><br>
                        <small style="color: #5a6c7d !important; font-weight: 500;">💾 Size: {info['size']}</small>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
            
            if st.button(f"Load {name}", key=f"demo_{name.lower()}", help=f"Load the {name} dataset", use_container_width=True):
                loaded_data = load_demo_dataset(name)
                if loaded_data is not None:
                    st.success(f"✅ Demo dataset '{name}' loaded successfully!")
                    st.balloons()  # Add celebration effect
                    # Force page refresh to show the loaded data
                    st.rerun()
    
    # Footer
    st.markdown('''
        <div style="margin-top: 3rem; text-align: center;">
            <div class="glass" style="padding: 1.5rem; background: rgba(255, 255, 255, 0.9) !important;">
                <div style="color: #2c3e50 !important; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">
                    🚀 Ready to Process Your Data?
                </div>
                <p style="color: #34495e !important; margin: 0; line-height: 1.6; font-weight: 500;">
                    Upload your dataset using the sidebar to get started with AI-powered data cleaning and analysis!<br>
                    <small style="color: #5a6c7d !important;">✨ Now supporting datasets up to 2GB with intelligent optimization</small>
                </p>
            </div>
        </div>
    ''', unsafe_allow_html=True)

def show_data_overview(data: pd.DataFrame):
    """Show data overview tab"""
    st.header("📊 Dataset Overview")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{data.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{data.shape[1]:,}")
    with col3:
        missing_pct = (data.isnull().sum().sum() / data.size) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")
    with col4:
        memory_mb = data.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory (MB)", f"{memory_mb:.1f}")
    
    st.markdown("---")
    
    # Data preview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Column Types")
        
        # Column type distribution
        numeric_cols = data.select_dtypes(include=[np.number]).shape[1]
        categorical_cols = data.select_dtypes(include=['object', 'category']).shape[1]
        datetime_cols = data.select_dtypes(include=['datetime64']).shape[1]
        
        type_data = {
            'Type': ['Numeric', 'Categorical', 'DateTime'],
            'Count': [numeric_cols, categorical_cols, datetime_cols]
        }
        
        fig = px.pie(
            values=type_data['Count'],
            names=type_data['Type'],
            title="Column Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Missing data heatmap
    if data.isnull().any().any():
        st.subheader("Missing Data Pattern")
        
        # Create missing data heatmap
        missing_data = data.isnull()
        
        fig = px.imshow(
            missing_data.T,
            title="Missing Data Heatmap (White = Missing)",
            color_continuous_scale=['lightblue', 'white'],
            aspect='auto'
        )
        fig.update_layout(
            xaxis_title="Row Index",
            yaxis_title="Columns"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_data_cleaning_interface():
    """Show data cleaning interface"""
    st.header("🧹 Data Cleaning")
    
    # Assessment button
    if st.button("🔍 Assess Data Quality"):
        with st.spinner("Analyzing data quality..."):
            quality_report = st.session_state.cleaning_agent.assess_data_quality()
            st.session_state.quality_report = quality_report
    
    # Show quality report if available
    if hasattr(st.session_state, 'quality_report'):
        show_quality_report(st.session_state.quality_report)
        
        # Auto-cleaning options
        st.markdown("---")
        st.subheader("🤖 Automated Cleaning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Auto Clean (Conservative)", help="Apply high-priority fixes only"):
                with st.spinner("Applying conservative cleaning..."):
                    cleaned_data = st.session_state.cleaning_agent.auto_clean(aggressive=False)
                    st.session_state.data = cleaned_data
                    st.success("✅ Conservative cleaning completed!")
                    st.rerun()
        
        with col2:
            if st.button("⚡ Auto Clean (Aggressive)", help="Apply all recommended fixes"):
                with st.spinner("Applying aggressive cleaning..."):
                    cleaned_data = st.session_state.cleaning_agent.auto_clean(aggressive=True)
                    st.session_state.data = cleaned_data
                    st.success("✅ Aggressive cleaning completed!")
                    st.rerun()

def show_quality_report(quality_report: dict):
    """Display the quality assessment report"""
    st.subheader("📋 Data Quality Assessment")
    
    # Missing values
    if quality_report['missing_values']['columns_with_missing']:
        st.markdown("### ❌ Missing Values")
        missing_df = pd.DataFrame([
            {
                'Column': col,
                'Missing Count': info['count'],
                'Missing %': f"{info['percentage']:.1f}%",
                'Pattern': info['pattern']
            }
            for col, info in quality_report['missing_values']['columns_with_missing'].items()
        ])
        st.dataframe(missing_df, use_container_width=True)
    
    # Duplicates
    if quality_report['duplicates']['exact_duplicates'] > 0:
        st.markdown("### 🔄 Duplicate Records")
        st.warning(f"Found {quality_report['duplicates']['exact_duplicates']} duplicate rows ({quality_report['duplicates']['duplicate_percentage']:.1f}%)")
    
    # Data type optimizations
    if quality_report['data_types']['optimization_suggestions']:
        st.markdown("### 🔧 Data Type Optimizations")
        dtype_df = pd.DataFrame([
            {
                'Column': col,
                'Current Type': info['current'],
                'Suggested Type': info['suggested'],
                'Memory Savings (bytes)': info['memory_savings']
            }
            for col, info in quality_report['data_types']['optimization_suggestions'].items()
        ])
        st.dataframe(dtype_df, use_container_width=True)
    
    # Recommendations
    if quality_report['recommendations']:
        st.markdown("### 💡 AI Recommendations")
        for i, rec in enumerate(quality_report['recommendations']):
            priority_color = {
                'high': '🔴',
                'medium': '🟡', 
                'low': '🟢'
            }.get(rec['priority'], '⚪')
            
            st.markdown(f"{priority_color} **{rec['priority'].upper()}**: {rec['description']}")

def show_eda_analysis():
    """Show EDA analysis tab"""
    st.header("🔍 Exploratory Data Analysis")
    
    # Generate EDA report button
    if st.button("📊 Generate EDA Report"):
        with st.spinner("Performing comprehensive EDA analysis..."):
            eda_report = st.session_state.eda_agent.generate_comprehensive_report()
            st.session_state.eda_report = eda_report
    
    # Show EDA report if available
    if hasattr(st.session_state, 'eda_report'):
        show_eda_report(st.session_state.eda_report)

def show_eda_report(eda_report: dict):
    """Display the EDA report"""
    
    # AI Insights
    if eda_report['ai_insights']:
        st.markdown("### 🤖 AI-Generated Insights")
        for insight in eda_report['ai_insights']:
            st.info(f"💡 {insight}")
    
    # Statistical Summary
    if 'basic_statistics' in eda_report['statistical_summary']:
        st.markdown("### 📈 Statistical Summary")
        stats_df = pd.DataFrame(eda_report['statistical_summary']['basic_statistics'])
        st.dataframe(stats_df, use_container_width=True)
    
    # Correlation Analysis
    if 'high_correlations' in eda_report['correlation_analysis']:
        high_corrs = eda_report['correlation_analysis']['high_correlations']
        if high_corrs:
            st.markdown("### 🔗 High Correlations")
            corr_df = pd.DataFrame(high_corrs)
            st.dataframe(corr_df, use_container_width=True)
    
    # Recommendations
    if eda_report['recommendations']:
        st.markdown("### 📋 EDA Recommendations")
        for rec in eda_report['recommendations']:
            st.markdown(f"**{rec['category']}** ({rec['priority']} Priority): {rec['recommendation']}")
            st.caption(f"Reason: {rec['reason']}")

def show_visualizations(data: pd.DataFrame):
    """Show visualizations tab"""
    st.header("📈 Interactive Visualizations")
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Distribution Plots", "Correlation Matrix", "Box Plots", "Scatter Plots", "Time Series"]
    )
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if viz_type == "Distribution Plots":
        show_distribution_plots(data, numeric_columns)
    elif viz_type == "Correlation Matrix":
        show_correlation_matrix(data, numeric_columns)
    elif viz_type == "Box Plots":
        show_box_plots(data, numeric_columns, categorical_columns)
    elif viz_type == "Scatter Plots":
        show_scatter_plots(data, numeric_columns, categorical_columns)
    elif viz_type == "Time Series":
        show_time_series_plots(data)

def show_distribution_plots(data: pd.DataFrame, numeric_columns: list):
    """Show distribution plots"""
    if not numeric_columns:
        st.warning("No numeric columns found for distribution plots.")
        return
    
    selected_columns = st.multiselect(
        "Select columns for distribution analysis",
        numeric_columns,
        default=numeric_columns[:4]
    )
    
    if selected_columns:
        n_cols = min(len(selected_columns), 2)
        n_rows = (len(selected_columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=selected_columns,
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(selected_columns):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=data[col], name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=300 * n_rows, title="Distribution Analysis")
        st.plotly_chart(fig, use_container_width=True)

def show_correlation_matrix(data: pd.DataFrame, numeric_columns: list):
    """Show correlation matrix"""
    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
    
    corr_method = st.selectbox("Correlation Method", ["pearson", "spearman"])
    
    numeric_data = data[numeric_columns]
    correlation_matrix = numeric_data.corr(method=corr_method)
    
    fig = px.imshow(
        correlation_matrix,
        title=f"{corr_method.title()} Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_box_plots(data: pd.DataFrame, numeric_columns: list, categorical_columns: list):
    """Show box plots"""
    if not numeric_columns:
        st.warning("No numeric columns found for box plots.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        y_column = st.selectbox("Select Y-axis (numeric)", numeric_columns)
    
    with col2:
        x_column = st.selectbox("Select X-axis (categorical)", ['None'] + categorical_columns)
    
    if x_column == 'None':
        fig = px.box(data, y=y_column, title=f"Box Plot of {y_column}")
    else:
        fig = px.box(data, x=x_column, y=y_column, title=f"Box Plot of {y_column} by {x_column}")
    
    st.plotly_chart(fig, use_container_width=True)

def show_scatter_plots(data: pd.DataFrame, numeric_columns: list, categorical_columns: list):
    """Show scatter plots"""
    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns for scatter plots.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_column = st.selectbox("X-axis", numeric_columns)
    
    with col2:
        y_column = st.selectbox("Y-axis", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
    
    with col3:
        color_column = st.selectbox("Color by", ['None'] + categorical_columns + numeric_columns)
    
    if color_column == 'None':
        fig = px.scatter(data, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")
    else:
        fig = px.scatter(
            data, x=x_column, y=y_column, color=color_column,
            title=f"{y_column} vs {x_column} (colored by {color_column})"
        )
    
    st.plotly_chart(fig, use_container_width=True)

def show_time_series_plots(data: pd.DataFrame):
    """Show time series plots"""
    datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not datetime_columns and not any(col.lower() in ['date', 'time', 'timestamp'] for col in data.columns):
        st.warning("No datetime columns detected. Try converting a column to datetime first.")
        return
    
    # If no datetime columns but date-like column names exist
    potential_date_cols = [col for col in data.columns if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp'])]
    
    if potential_date_cols:
        date_column = st.selectbox("Select date column", potential_date_cols)
        value_column = st.selectbox("Select value column", numeric_columns)
        
        # Try to convert to datetime
        try:
            data_copy = data.copy()
            data_copy[date_column] = pd.to_datetime(data_copy[date_column])
            
            fig = px.line(
                data_copy, x=date_column, y=value_column,
                title=f"Time Series: {value_column} over {date_column}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Could not convert {date_column} to datetime: {str(e)}")

def show_reports():
    """Show reports tab"""
    st.header("📋 Generated Reports")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Data Quality Report"):
            if hasattr(st.session_state, 'quality_report'):
                report_html = generate_quality_report_html(st.session_state.quality_report)
                st.download_button(
                    "💾 Download Quality Report",
                    report_html,
                    "data_quality_report.html",
                    "text/html"
                )
            else:
                st.warning("Please run data quality assessment first.")
    
    with col2:
        if st.button("🔍 Export EDA Report"):
            if hasattr(st.session_state, 'eda_report'):
                report_html = generate_eda_report_html(st.session_state.eda_report)
                st.download_button(
                    "💾 Download EDA Report", 
                    report_html,
                    "eda_report.html",
                    "text/html"
                )
            else:
                st.warning("Please run EDA analysis first.")
    
    with col3:
        if st.button("📈 Export Cleaned Data"):
            if st.session_state.data is not None:
                csv_data = st.session_state.data.to_csv(index=False)
                st.download_button(
                    "💾 Download Cleaned Data",
                    csv_data,
                    "cleaned_data.csv",
                    "text/csv"
                )
            else:
                st.warning("No data available to export.")

def generate_quality_report_html(quality_report: dict) -> str:
    """Generate HTML report for data quality assessment"""
    html = f"""
    <html>
    <head><title>Data Quality Report</title></head>
    <body>
        <h1>Data Quality Assessment Report</h1>
        <h2>Missing Values</h2>
        <p>Total missing values: {quality_report['missing_values']['total_missing']}</p>
        
        <h2>Duplicates</h2>
        <p>Exact duplicates: {quality_report['duplicates']['exact_duplicates']}</p>
        
        <h2>Recommendations</h2>
        <ul>
    """
    
    for rec in quality_report['recommendations']:
        html += f"<li><strong>{rec['priority']}</strong>: {rec['description']}</li>"
    
    html += """
        </ul>
    </body>
    </html>
    """
    return html

def generate_eda_report_html(eda_report: dict) -> str:
    """Generate HTML report for EDA analysis"""
    html = f"""
    <html>
    <head><title>EDA Report</title></head>
    <body>
        <h1>Exploratory Data Analysis Report</h1>
        
        <h2>AI Insights</h2>
        <ul>
    """
    
    for insight in eda_report['ai_insights']:
        html += f"<li>{insight}</li>"
    
    html += """
        </ul>
        
        <h2>Recommendations</h2>
        <ul>
    """
    
    for rec in eda_report['recommendations']:
        html += f"<li><strong>{rec['category']}</strong>: {rec['recommendation']}</li>"
    
    html += """
        </ul>
    </body>
    </html>
    """
    return html

def show_ml_pipeline(data):
    """Show ML pipeline interface"""
    st.header("🤖 Machine Learning Pipeline")
    
    if data is None:
        st.info("Please upload a dataset to use the ML pipeline.")
        return
    
    # Initialize agents if not exists
    if 'preprocessing_agent' not in st.session_state:
        st.session_state.preprocessing_agent = PreprocessingAgent()
    if 'ml_agent' not in st.session_state:
        st.session_state.ml_agent = MLModelAgent()
    
    # ML Pipeline tabs
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        "🔧 Preprocessing", 
        "🎯 Model Training", 
        "📊 Model Comparison", 
        "🚀 Auto-ML"
    ])
    
    with ml_tab1:
        st.subheader("Data Preprocessing")
        
        # Target selection
        target_column = st.selectbox(
            "Select Target Column (for supervised learning)", 
            ['None'] + list(data.columns),
            help="Choose the column you want to predict"
        )
        
        if target_column != 'None':
            st.session_state.preprocessing_agent.set_data(data, target_column)
            
            # Analyze preprocessing needs
            if st.button("Analyze Preprocessing Requirements"):
                with st.spinner("Analyzing data for preprocessing..."):
                    analysis = st.session_state.preprocessing_agent.analyze_preprocessing_needs()

                    # Display analysis
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Missing Values Analysis:**")
                        if analysis['missing_values']:
                            missing_df = pd.DataFrame.from_dict(analysis['missing_values'], orient='index')
                            st.dataframe(missing_df)
                        else:
                            st.success("No missing values found!")

                    with col2:
                        st.write("**Categorical Variables:**")
                        if analysis['categorical_variables']:
                            cat_df = pd.DataFrame.from_dict(analysis['categorical_variables'], orient='index')
                            st.dataframe(cat_df[['unique_count', 'recommended_encoding']])
                        else:
                            st.info("No categorical variables found.")

                    # Recommendations
                    st.write("**Preprocessing Recommendations:**")
                    for rec in analysis['recommendations']:
                        if rec['priority'] == 'high':
                            st.error(f"🔴 {rec['action']}: {rec['details']}")
                        elif rec['priority'] == 'medium':
                            st.warning(f"🟡 {rec['action']}: {rec['details']}")
                        else:
                            st.info(f"🔵 {rec['action']}: {rec['details']}")
            
            # Auto preprocessing
            if st.button("Auto-Preprocess Data"):
                with st.spinner("Preprocessing data..."):
                    result = st.session_state.preprocessing_agent.auto_preprocess(target_column)
                    st.session_state.preprocessed_data = result
                    
                    st.success("✅ Data preprocessing completed!")
                    
                    # Show preprocessing summary
                    summary = st.session_state.preprocessing_agent.get_preprocessing_summary()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Features", summary['original_shape'][1])
                    with col2:
                        st.metric("Final Features", summary['final_shape'][1])
                    with col3:
                        st.metric("Preprocessing Steps", len(summary['preprocessing_steps']))
                    
                    # Show steps taken
                    with st.expander("Preprocessing Steps Performed"):
                        for step in summary['preprocessing_steps']:
                            st.write(f"• {step}")
        else:
            st.info("Select a target column to start preprocessing analysis.")
    
    with ml_tab2:
        st.subheader("Model Training")
        
        if 'preprocessed_data' in st.session_state:
            processed_result = st.session_state.preprocessed_data
            
            if 'X_train' in processed_result:
                # Set up ML agent
                st.session_state.ml_agent.set_data(
                    processed_result['X_train'],
                    processed_result['X_test'],
                    processed_result['y_train'],
                    processed_result['y_test']
                )
                
                # Model selection
                st.write("**Select Models to Train:**")
                analysis = st.session_state.ml_agent.analyze_ml_requirements()
                recommended_models = analysis['recommended_models']
                
                selected_models = st.multiselect(
                    "Choose models", 
                    recommended_models,
                    default=recommended_models[:3]
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    quick_mode = st.checkbox("Quick Mode", value=True, help="Faster training with default parameters")
                with col2:
                    if st.button("Train Selected Models"):
                        if selected_models:
                            results = {}
                            progress_bar = st.progress(0)
                            
                            for i, model_name in enumerate(selected_models):
                                st.write(f"Training {model_name}...")
                                try:
                                    result = st.session_state.ml_agent._train_single_model(model_name, quick_mode)
                                    results[model_name] = result
                                    st.success(f"✅ {model_name} completed")
                                except Exception as e:
                                    st.error(f"❌ {model_name} failed: {str(e)}")
                                
                                progress_bar.progress((i + 1) / len(selected_models))
                            
                            st.session_state.ml_results = results
                            st.success("🎉 Model training completed!")
                        else:
                            st.warning("Please select at least one model to train.")
            else:
                st.info("Preprocessed data doesn't include train-test split. Using full dataset.")
        else:
            st.info("Please preprocess the data first in the Preprocessing tab.")
    
    with ml_tab3:
        st.subheader("Model Comparison")
        
        if 'ml_results' in st.session_state:
            results = st.session_state.ml_results
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, result in results.items():
                row = {
                    'Model': model_name,
                    'CV Score': f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f}"
                }
                
                # Add test metrics
                for metric_name, value in result['metrics']['test'].items():
                    row[f'Test {metric_name.title()}'] = f"{value:.4f}"
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance visualization
            if len(results) > 1:
                # CV scores comparison
                cv_data = {
                    'Model': list(results.keys()),
                    'CV_Score': [result['cv_mean'] for result in results.values()],
                    'CV_Std': [result['cv_std'] for result in results.values()]
                }
                
                fig = px.bar(
                    cv_data, 
                    x='Model', 
                    y='CV_Score',
                    error_y='CV_Std',
                    title="Cross-Validation Performance Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance comparison (if available)
                importance_data = {}
                for model_name, result in results.items():
                    if result['feature_importance']:
                        importance_data[model_name] = result['feature_importance']
                
                if importance_data:
                    st.write("**Feature Importance Comparison:**")
                    importance_df = pd.DataFrame(importance_data).fillna(0)
                    
                    # Show top 10 features
                    top_features = importance_df.mean(axis=1).sort_values(ascending=False).head(10)
                    
                    fig = px.bar(
                        x=top_features.values,
                        y=top_features.index,
                        orientation='h',
                        title="Top 10 Most Important Features (Average)"
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train some models first to see the comparison.")
    
    with ml_tab4:
        st.subheader("Auto-ML Pipeline")
        st.write("Run a complete automated machine learning pipeline with minimal configuration.")
        
        if 'preprocessed_data' in st.session_state:
            processed_result = st.session_state.preprocessed_data
            
            if 'X_train' in processed_result:
                col1, col2 = st.columns(2)
                
                with col1:
                    auto_quick_mode = st.checkbox("Quick Auto-ML", value=True, 
                                                help="Faster execution with fewer models")
                
                with col2:
                    if st.button("🚀 Run Auto-ML Pipeline"):
                        # Set up ML agent
                        st.session_state.ml_agent.set_data(
                            processed_result['X_train'],
                            processed_result['X_test'],
                            processed_result['y_train'],
                            processed_result['y_test']
                        )
                        
                        # Run Auto-ML
                        with st.spinner("Running Auto-ML pipeline... This may take a few minutes."):
                            auto_ml_report = st.session_state.ml_agent.auto_ml_pipeline(quick_mode=auto_quick_mode)
                            st.session_state.auto_ml_report = auto_ml_report
                        
                        st.balloons()
                        st.success("🎉 Auto-ML Pipeline completed successfully!")
                
                # Display Auto-ML results
                if 'auto_ml_report' in st.session_state:
                    report = st.session_state.auto_ml_report
                    
                    # Best model summary
                    st.write("## 🏆 Best Model")
                    best_model = report['best_model']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Model", best_model['model_name'])
                    with col2:
                        primary_metric = list(best_model['performance']['test'].keys())[0]
                        st.metric(f"Test {primary_metric.title()}", 
                                f"{best_model['performance']['test'][primary_metric]:.4f}")
                    with col3:
                        st.metric("CV Score", 
                                f"{best_model['cv_performance']['mean']:.4f}")
                    
                    # Model comparison
                    st.write("## 📊 All Models Performance")
                    if 'model_comparison' in report:
                        st.dataframe(report['model_comparison'], use_container_width=True)
                    
                    # Recommendations
                    st.write("## 💡 Recommendations")
                    if 'recommendations' in report:
                        for rec in report['recommendations']:
                            if rec['priority'] == 'high':
                                st.error(f"🔴 **{rec['title']}**: {rec['description']}")
                            elif rec['priority'] == 'medium':
                                st.warning(f"🟡 **{rec['title']}**: {rec['description']}")
                            else:
                                st.info(f"🔵 **{rec['title']}**: {rec['description']}")
                    
                    # Next steps
                    if 'next_steps' in report:
                        st.write("## 🎯 Suggested Next Steps")
                        for step in report['next_steps']:
                            st.write(f"• {step}")
            else:
                st.info("Preprocessed data doesn't include train-test split. Please ensure you have a target column selected.")
        else:
            st.info("Please preprocess the data first to run Auto-ML.")

if __name__ == "__main__":
    main()