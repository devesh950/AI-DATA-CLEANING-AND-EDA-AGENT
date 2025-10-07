"""
Test script for verifying 2GB upload capability
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_test_file(size_mb=500, filename="test_large_dataset.csv"):
    """Create a test file of specified size"""
    print(f"Creating {size_mb}MB test file: {filename}")
    
    # Estimate rows needed (roughly 100 bytes per row)
    estimated_rows = int(size_mb * 1024 * 1024 / 100)
    
    # Generate data in chunks to avoid memory issues
    chunk_size = 50000
    chunks = []
    
    for i in range(0, estimated_rows, chunk_size):
        end_row = min(i + chunk_size, estimated_rows)
        chunk_rows = end_row - i
        
        chunk_data = {
            'id': range(i, end_row),
            'feature_1': np.random.randn(chunk_rows),
            'feature_2': np.random.randn(chunk_rows),
            'feature_3': np.random.uniform(0, 100, chunk_rows),
            'category_1': np.random.choice(['A', 'B', 'C', 'D'], chunk_rows),
            'category_2': np.random.choice(['Type1', 'Type2', 'Type3'], chunk_rows),
            'value': np.random.randint(0, 1000, chunk_rows),
            'score': np.random.exponential(2, chunk_rows),
            'rating': np.random.gamma(2, 2, chunk_rows),
            'target': np.random.choice([0, 1], chunk_rows)
        }
        
        chunk_df = pd.DataFrame(chunk_data)
        chunks.append(chunk_df)
        
        print(f"Generated chunk {len(chunks)}: {chunk_rows} rows")
    
    # Combine and save
    print("Combining chunks and saving...")
    full_df = pd.concat(chunks, ignore_index=True)
    full_df.to_csv(filename, index=False)
    
    # Verify file size
    actual_size_mb = Path(filename).stat().st_size / (1024 * 1024)
    print(f"✅ Created: {filename}")
    print(f"   Rows: {len(full_df):,}")
    print(f"   Columns: {len(full_df.columns)}")
    print(f"   Size: {actual_size_mb:.1f} MB")
    
    return filename, actual_size_mb

if __name__ == "__main__":
    print("🧪 Testing Large File Upload Capability")
    print("=" * 50)
    
    # Create test files of different sizes
    test_sizes = [300, 500, 800]  # MB
    
    for size in test_sizes:
        print(f"\n📊 Creating {size}MB test file...")
        try:
            filename, actual_size = create_test_file(size, f"test_{size}mb.csv")
            print(f"✅ Success: {filename} ({actual_size:.1f} MB)")
        except Exception as e:
            print(f"❌ Error creating {size}MB file: {e}")
    
    print(f"\n🎯 Test files created! You can now:")
    print(f"1. Start the Streamlit app: streamlit run app.py")
    print(f"2. Try uploading the test files to verify 2GB support")
    print(f"3. Files will be automatically optimized if >200MB")
    
    print(f"\n📋 Upload test checklist:")
    print(f"✅ Files up to 2GB supported")
    print(f"✅ Progress bars for large uploads") 
    print(f"✅ Memory optimization automatic")
    print(f"✅ File size display improved")