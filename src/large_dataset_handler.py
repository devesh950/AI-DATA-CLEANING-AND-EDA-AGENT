"""
Large Dataset Handler for processing datasets up to 2GB efficiently
"""

import pandas as pd
import numpy as np
import polars as pl
import dask.dataframe as dd
from typing import Dict, List, Any, Optional, Union, Iterator
import gc
import psutil
import os
from pathlib import Path
import warnings
from contextlib import contextmanager

warnings.filterwarnings('ignore')

class LargeDatasetHandler:
    """
    Optimized handler for processing large datasets (up to 2GB) with memory efficiency
    """
    
    def __init__(self, 
                 chunk_size: int = 100000,
                 memory_limit_gb: float = 1.5,
                 use_polars: bool = True,
                 use_dask: bool = False):
        """
        Initialize Large Dataset Handler
        
        Args:
            chunk_size: Number of rows to process in each chunk
            memory_limit_gb: Memory limit in GB for processing
            use_polars: Use Polars for faster processing
            use_dask: Use Dask for distributed computing
        """
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        self.use_polars = use_polars
        self.use_dask = use_dask
        self.data = None
        self.data_info = {}
        
    def load_large_dataset(self, 
                          file_path: Union[str, Path],
                          file_type: str = 'auto',
                          **kwargs) -> Union[pd.DataFrame, pl.DataFrame, dd.DataFrame]:
        """
        Load large datasets efficiently with automatic optimization
        
        Args:
            file_path: Path to the dataset file
            file_type: File type ('csv', 'parquet', 'json', 'excel', 'auto')
            **kwargs: Additional arguments for file reading
            
        Returns:
            Loaded dataset (optimized format)
        """
        file_path = Path(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Auto-detect file type
        if file_type == 'auto':
            file_type = file_path.suffix.lower().lstrip('.')
        
        print(f"Loading dataset: {file_path.name}")
        print(f"File size: {file_size_mb:.1f} MB")
        
        # Choose optimal loading strategy based on file size
        if file_size_mb > 500 and self.use_dask:
            return self._load_with_dask(file_path, file_type, **kwargs)
        elif file_size_mb > 100 and self.use_polars:
            return self._load_with_polars(file_path, file_type, **kwargs)
        else:
            return self._load_optimized_pandas(file_path, file_type, **kwargs)
    
    def _load_with_dask(self, file_path: Path, file_type: str, **kwargs) -> dd.DataFrame:
        """Load large files using Dask for distributed processing"""
        print("🚀 Loading with Dask for distributed processing...")
        
        try:
            if file_type == 'csv':
                # Optimize CSV reading with Dask
                return dd.read_csv(
                    str(file_path),
                    blocksize="64MB",  # 64MB blocks for optimal performance
                    assume_missing=True,
                    **kwargs
                )
            elif file_type == 'parquet':
                return dd.read_parquet(str(file_path), **kwargs)
            else:
                # Fallback to pandas for unsupported types
                return dd.from_pandas(
                    self._load_optimized_pandas(file_path, file_type, **kwargs),
                    npartitions=4
                )
        except Exception as e:
            print(f"Dask loading failed: {e}")
            return self._load_with_polars(file_path, file_type, **kwargs)
    
    def _load_with_polars(self, file_path: Path, file_type: str, **kwargs) -> pl.DataFrame:
        """Load large files using Polars for fast processing"""
        print("⚡ Loading with Polars for fast processing...")
        
        try:
            if file_type == 'csv':
                return pl.read_csv(
                    str(file_path),
                    infer_schema_length=10000,  # Infer schema from first 10k rows
                    **kwargs
                )
            elif file_type == 'parquet':
                return pl.read_parquet(str(file_path), **kwargs)
            elif file_type == 'json':
                return pl.read_json(str(file_path), **kwargs)
            else:
                # Fallback to pandas
                return pl.from_pandas(self._load_optimized_pandas(file_path, file_type, **kwargs))
        except Exception as e:
            print(f"Polars loading failed: {e}")
            return self._load_optimized_pandas(file_path, file_type, **kwargs)
    
    def _load_optimized_pandas(self, file_path: Path, file_type: str, **kwargs) -> pd.DataFrame:
        """Load files with optimized pandas settings"""
        print("🐼 Loading with optimized Pandas...")
        
        # Memory optimization settings
        pandas_kwargs = {
            'low_memory': False,
            'engine': 'c',  # Use C engine for speed
        }
        pandas_kwargs.update(kwargs)
        
        if file_type == 'csv':
            # For large CSV files, read in chunks and optimize dtypes
            return self._load_csv_optimized(file_path, **pandas_kwargs)
        elif file_type == 'parquet':
            return pd.read_parquet(str(file_path), **pandas_kwargs)
        elif file_type == 'json':
            return pd.read_json(str(file_path), **pandas_kwargs)
        elif file_type in ['xlsx', 'xls']:
            return pd.read_excel(str(file_path), **pandas_kwargs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _load_csv_optimized(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV with automatic dtype optimization"""
        # First, sample the data to determine optimal dtypes
        sample_df = pd.read_csv(str(file_path), nrows=10000, **kwargs)
        
        # Optimize dtypes
        optimized_dtypes = self._optimize_dtypes(sample_df)
        
        # Load full dataset with optimized dtypes
        return pd.read_csv(str(file_path), dtype=optimized_dtypes, **kwargs)
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> Dict[str, str]:
        """Optimize column dtypes to reduce memory usage"""
        optimized_dtypes = {}
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                # Numeric optimization
                if pd.api.types.is_integer_dtype(df[col]):
                    # Check if we can use smaller integer types
                    c_min = df[col].min()
                    c_max = df[col].max()
                    
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_dtypes[col] = 'int8'
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_dtypes[col] = 'int16'
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_dtypes[col] = 'int32'
                    else:
                        optimized_dtypes[col] = 'int64'
                        
                elif pd.api.types.is_float_dtype(df[col]):
                    # Use float32 if precision allows
                    if df[col].between(-3.4e38, 3.4e38).all():
                        optimized_dtypes[col] = 'float32'
                    else:
                        optimized_dtypes[col] = 'float64'
            else:
                # String optimization
                num_unique_values = df[col].nunique()
                num_total_values = len(df[col])
                
                # Use category for low cardinality strings
                if num_unique_values / num_total_values < 0.5:
                    optimized_dtypes[col] = 'category'
        
        return optimized_dtypes
    
    def reduce_memory_usage(self, df: Union[pd.DataFrame, pl.DataFrame]) -> Union[pd.DataFrame, pl.DataFrame]:
        """Reduce memory usage of dataframe"""
        if isinstance(df, pl.DataFrame):
            # Polars is already memory efficient
            return df
        
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                        
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                # Convert to category if beneficial
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
        
        return df
    
    def process_in_chunks(self, 
                         df: Union[pd.DataFrame, str], 
                         processing_func,
                         chunk_size: Optional[int] = None,
                         **kwargs) -> List[Any]:
        """Process large dataframe in chunks to manage memory"""
        chunk_size = chunk_size or self.chunk_size
        results = []
        
        if isinstance(df, str):
            # Process file in chunks
            for chunk_df in pd.read_csv(df, chunksize=chunk_size):
                chunk_df = self.reduce_memory_usage(chunk_df)
                result = processing_func(chunk_df, **kwargs)
                results.append(result)
                gc.collect()  # Force garbage collection
        else:
            # Process dataframe in chunks
            for i in range(0, len(df), chunk_size):
                chunk_df = df.iloc[i:i + chunk_size]
                result = processing_func(chunk_df, **kwargs)
                results.append(result)
                gc.collect()
        
        return results
    
    @contextmanager
    def memory_monitor(self):
        """Context manager to monitor memory usage"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        try:
            yield
        finally:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"Final memory usage: {final_memory:.1f} MB")
            print(f"Memory change: {final_memory - initial_memory:+.1f} MB")
            
            # Warning if memory usage is too high
            if final_memory > self.memory_limit_gb * 1024:
                print(f"⚠️ Warning: Memory usage ({final_memory:.1f} MB) exceeds limit ({self.memory_limit_gb * 1024:.1f} MB)")
    
    def get_dataset_info(self, df: Union[pd.DataFrame, pl.DataFrame, dd.DataFrame]) -> Dict[str, Any]:
        """Get comprehensive information about the dataset"""
        info = {}
        
        if isinstance(df, dd.DataFrame):
            # Dask DataFrame
            info.update({
                'type': 'dask',
                'shape': (len(df), len(df.columns)),
                'columns': list(df.columns),
                'dtypes': dict(df.dtypes),
                'partitions': df.npartitions,
                'memory_usage_per_partition': 'Unknown (lazy evaluation)'
            })
        elif isinstance(df, pl.DataFrame):
            # Polars DataFrame
            info.update({
                'type': 'polars',
                'shape': df.shape,
                'columns': df.columns,
                'dtypes': dict(zip(df.columns, [str(dtype) for dtype in df.dtypes])),
                'memory_usage_mb': df.estimated_size("mb"),
            })
        else:
            # Pandas DataFrame
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            info.update({
                'type': 'pandas',
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': dict(df.dtypes),
                'memory_usage_mb': memory_usage,
                'null_counts': dict(df.isnull().sum()),
            })
        
        return info
    
    def convert_to_parquet(self, 
                          input_file: Union[str, Path],
                          output_file: Union[str, Path],
                          compression: str = 'snappy') -> None:
        """Convert large datasets to Parquet format for better performance"""
        input_file = Path(input_file)
        output_file = Path(output_file)
        
        print(f"Converting {input_file.name} to Parquet format...")
        
        if input_file.suffix.lower() == '.csv':
            # Process CSV in chunks and save as Parquet
            chunk_list = []
            for chunk in pd.read_csv(str(input_file), chunksize=self.chunk_size):
                chunk = self.reduce_memory_usage(chunk)
                chunk_list.append(chunk)
            
            # Combine chunks and save
            full_df = pd.concat(chunk_list, ignore_index=True)
            full_df.to_parquet(str(output_file), compression=compression, index=False)
            
            print(f"✅ Converted to Parquet: {output_file}")
            
            # Show size comparison
            original_size = input_file.stat().st_size / 1024**2
            new_size = output_file.stat().st_size / 1024**2
            print(f"Size reduction: {original_size:.1f} MB → {new_size:.1f} MB ({100*(1-new_size/original_size):.1f}% smaller)")
        
        else:
            raise ValueError("Currently only CSV to Parquet conversion is supported")
    
    def sample_large_dataset(self, 
                           df: Union[pd.DataFrame, str],
                           sample_size: int = 100000,
                           method: str = 'random') -> pd.DataFrame:
        """Create a representative sample from large dataset"""
        if isinstance(df, str):
            # Sample from file
            total_lines = sum(1 for _ in open(df)) - 1  # Exclude header
            
            if total_lines <= sample_size:
                return pd.read_csv(df)
            
            if method == 'random':
                skip_idx = np.random.choice(range(1, total_lines), 
                                          size=total_lines - sample_size, 
                                          replace=False)
                return pd.read_csv(df, skiprows=skip_idx)
            else:
                # Systematic sampling
                step = total_lines // sample_size
                return pd.read_csv(df, skiprows=lambda x: x % step != 0 and x != 0)
        else:
            # Sample from DataFrame
            if len(df) <= sample_size:
                return df
            
            if method == 'random':
                return df.sample(n=sample_size, random_state=42)
            else:
                # Systematic sampling
                step = len(df) // sample_size
                return df.iloc[::step].head(sample_size)