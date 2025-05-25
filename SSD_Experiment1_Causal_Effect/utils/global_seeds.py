"""
Global Seeds Utility for Reproducibility

This module provides centralized random seed management to ensure
reproducibility across all pipeline components.
"""

import os
import random
import numpy as np
import warnings
from typing import Optional

# Global seed value
GLOBAL_SEED = 42

def set_global_seeds(seed: Optional[int] = None) -> int:
    """
    Set random seeds for Python, NumPy, and environment variables.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value. If None, uses GLOBAL_SEED (42).
    
    Returns
    -------
    int
        The seed value that was set.
    
    Examples
    --------
    >>> set_global_seeds()
    42
    >>> set_global_seeds(123)
    123
    """
    if seed is None:
        seed = GLOBAL_SEED
    
    # Python's built-in random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # Python hash seed for dictionary ordering
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # TensorFlow (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    # Scikit-learn (doesn't have global seed, but we can set numpy)
    # Already handled above with np.random.seed()
    
    return seed

def get_random_state(base_seed: Optional[int] = None, offset: int = 0) -> int:
    """
    Get a deterministic random state based on base seed and offset.
    
    Useful for creating different but reproducible random states
    for different components of the pipeline.
    
    Parameters
    ----------
    base_seed : int, optional
        Base seed value. If None, uses GLOBAL_SEED.
    offset : int
        Offset to add to base seed.
    
    Returns
    -------
    int
        Random state value.
    
    Examples
    --------
    >>> get_random_state()
    42
    >>> get_random_state(offset=10)
    52
    """
    if base_seed is None:
        base_seed = GLOBAL_SEED
    return base_seed + offset

def check_reproducibility() -> bool:
    """
    Check if environment is set up for reproducibility.
    
    Returns
    -------
    bool
        True if PYTHONHASHSEED is set, False otherwise.
    
    Warns
    -----
    UserWarning
        If PYTHONHASHSEED is not set.
    """
    if os.environ.get("PYTHONHASHSEED") is None:
        warnings.warn(
            "PYTHONHASHSEED is not set. Dictionary ordering may not be reproducible. "
            "Call set_global_seeds() to ensure reproducibility.",
            UserWarning
        )
        return False
    return True

# Set seeds on module import for convenience
set_global_seeds()