#!/usr/bin/env python3
"""
Memory Optimization and Monitoring for SSD Pipeline
Author: Research Assistant
Date: 2025-06-29
"""

import os
import sys
import gc
import psutil
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('logs/memory_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor and optimize memory usage during pipeline execution"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = self.get_memory_usage()
        logger.info(f"Memory monitor initialized. Initial memory: {self.start_memory:.2f} MB")
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_system_memory(self):
        """Get system memory statistics"""
        mem = psutil.virtual_memory()
        return {
            'total': mem.total / 1024 / 1024 / 1024,  # GB
            'available': mem.available / 1024 / 1024 / 1024,  # GB
            'percent': mem.percent,
            'used': mem.used / 1024 / 1024 / 1024  # GB
        }
    
    def log_memory_status(self, step_name=""):
        """Log current memory status"""
        current_memory = self.get_memory_usage()
        system_mem = self.get_system_memory()
        
        logger.info(f"=== MEMORY STATUS {step_name} ===")
        logger.info(f"Process Memory: {current_memory:.2f} MB")
        logger.info(f"Memory Change: {current_memory - self.start_memory:+.2f} MB")
        logger.info(f"System Memory: {system_mem['used']:.2f}/{system_mem['total']:.2f} GB ({system_mem['percent']:.1f}%)")
        logger.info(f"Available Memory: {system_mem['available']:.2f} GB")
        
        # Warning thresholds
        if system_mem['percent'] > 85:
            logger.warning(f"HIGH MEMORY USAGE: {system_mem['percent']:.1f}%")
        if system_mem['available'] < 2.0:
            logger.warning(f"LOW AVAILABLE MEMORY: {system_mem['available']:.2f} GB")
            
        return current_memory, system_mem
    
    def optimize_pandas(self):
        """Optimize pandas memory usage"""
        logger.info("Applying pandas memory optimizations...")
        
        # Set pandas options for memory efficiency
        pd.set_option('mode.copy_on_write', True)
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
        
        logger.info("Pandas optimizations applied")
    
    def force_garbage_collection(self):
        """Force garbage collection and log memory freed"""
        before_memory = self.get_memory_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        after_memory = self.get_memory_usage()
        freed_memory = before_memory - after_memory
        
        logger.info(f"Garbage collection: {collected} objects collected, {freed_memory:.2f} MB freed")
        return freed_memory
    
    def check_memory_threshold(self, threshold_percent=80):
        """Check if memory usage exceeds threshold"""
        system_mem = self.get_system_memory()
        if system_mem['percent'] > threshold_percent:
            logger.warning(f"Memory threshold exceeded: {system_mem['percent']:.1f}% > {threshold_percent}%")
            return True
        return False
    
    def optimize_dataframe(self, df, name="DataFrame"):
        """Optimize a pandas DataFrame memory usage"""
        if not isinstance(df, pd.DataFrame):
            return df
            
        logger.info(f"Optimizing {name} memory usage...")
        
        # Memory usage before optimization
        memory_before = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if df[col].min() > -128 and df[col].max() < 127:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Optimize object columns (convert to category if beneficial)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Memory usage after optimization
        memory_after = df.memory_usage(deep=True).sum() / 1024 / 1024
        memory_saved = memory_before - memory_after
        
        logger.info(f"{name} memory optimization: {memory_before:.2f} MB -> {memory_after:.2f} MB (saved {memory_saved:.2f} MB)")
        
        return df

def setup_memory_monitoring():
    """Setup memory monitoring for the current process"""
    monitor = MemoryMonitor()
    monitor.optimize_pandas()
    monitor.log_memory_status("SETUP")
    return monitor

def main():
    """Main function for testing memory monitoring"""
    monitor = setup_memory_monitoring()
    
    # Test with a sample DataFrame
    logger.info("Creating test DataFrame...")
    n_rows = 100000
    test_df = pd.DataFrame({
        'id': range(n_rows),
        'category': (['A', 'B', 'C'] * (n_rows // 3 + 1))[:n_rows],
        'value': ([1.0, 2.0, 3.0] * (n_rows // 3 + 1))[:n_rows],
        'flag': ([True, False] * (n_rows // 2 + 1))[:n_rows]
    })
    
    monitor.log_memory_status("AFTER_DATAFRAME_CREATION")
    
    # Optimize the DataFrame
    optimized_df = monitor.optimize_dataframe(test_df, "test_df")
    
    monitor.log_memory_status("AFTER_OPTIMIZATION")
    
    # Force garbage collection
    monitor.force_garbage_collection()
    
    monitor.log_memory_status("AFTER_GC")
    
    logger.info("Memory monitoring test completed")

if __name__ == "__main__":
    main() 