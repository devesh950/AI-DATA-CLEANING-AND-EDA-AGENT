"""
Simple test for large dataset capabilities
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.large_dataset_handler import LargeDatasetHandler
    print("✅ LargeDatasetHandler imported successfully")
    
    # Test basic functionality
    handler = LargeDatasetHandler()
    print("✅ LargeDatasetHandler initialized")
    
    # Create small test dataset
    test_data = pd.DataFrame({
        'A': np.random.randn(10000),
        'B': np.random.choice(['X', 'Y', 'Z'], 10000),
        'C': np.random.randint(0, 100, 10000)
    })
    
    print(f"✅ Test dataset created: {test_data.shape}")
    
    # Test memory optimization
    original_memory = test_data.memory_usage(deep=True).sum() / 1024**2
    optimized_data = handler.reduce_memory_usage(test_data.copy())
    optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024**2
    
    print(f"✅ Memory optimization test:")
    print(f"   Original: {original_memory:.2f} MB")
    print(f"   Optimized: {optimized_memory:.2f} MB")
    print(f"   Reduction: {(1-optimized_memory/original_memory)*100:.1f}%")
    
    # Test dataset info
    info = handler.get_dataset_info(test_data)
    print(f"✅ Dataset info: {info['type']}, Shape: {info['shape']}")
    
    print("\n🎉 Large dataset handler basic functionality working!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")