"""
Utility functions for GPU acceleration
"""

import streamlit as st
import numpy as np
import time
import sys

# Check if GPU libraries are available
HAS_GPU = False
HAS_CUPY = False

try:
    from numba import cuda
    HAS_GPU = cuda.is_available()
except ImportError:
    pass

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    pass


def check_gpu():
    """Check for GPU availability and show status in UI"""
    if HAS_GPU:
        # Get basic GPU info
        gpu_info = cuda.get_current_device().name
        st.sidebar.success(f"GPU acceleration enabled: {gpu_info}")
    else:
        st.sidebar.info("GPU acceleration not available. Using CPU.")
        if 'cuda' in sys.modules:
            st.sidebar.warning("CUDA detected but no GPU device found.")


def to_gpu(array):
    """
    Transfer NumPy array to GPU
    
    Parameters:
    -----------
    array : np.ndarray
        NumPy array to transfer
        
    Returns:
    --------
    cupy.ndarray or numba.cuda.devicearray.DeviceNDArray
        Array on GPU
    """
    if not HAS_GPU:
        return array
    
    # Handle different data types
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    
    # Ensure dtype is compatible with GPU
    if array.dtype.kind in 'US':  # String types not compatible
        # Convert string to numeric representation if needed
        array = array.astype(np.float64)
    
    try:
        if HAS_CUPY:
            return cp.array(array)
        else:
            return cuda.to_device(array)
    except Exception as e:
        print(f"Error transferring to GPU: {e}")
        return array


def to_cpu(gpu_array):
    """
    Transfer array from GPU to CPU
    
    Parameters:
    -----------
    gpu_array : cupy.ndarray or numba.cuda.devicearray.DeviceNDArray
        Array on GPU
        
    Returns:
    --------
    np.ndarray
        NumPy array on CPU
    """
    if not HAS_GPU:
        return gpu_array
    
    try:
        if HAS_CUPY and hasattr(gpu_array, 'get'):
            return gpu_array.get()
        elif hasattr(gpu_array, 'copy_to_host'):
            return gpu_array.copy_to_host()
        return np.array(gpu_array)
    except Exception as e:
        print(f"Error transferring from GPU: {e}")
        return np.array(gpu_array) 