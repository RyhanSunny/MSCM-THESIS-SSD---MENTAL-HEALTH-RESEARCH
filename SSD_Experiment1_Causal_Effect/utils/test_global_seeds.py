"""
Unit tests for global_seeds module.
"""

import unittest
import os
import random
import numpy as np
from utils.global_seeds import (
    set_global_seeds, 
    get_random_state, 
    check_reproducibility,
    GLOBAL_SEED
)

class TestGlobalSeeds(unittest.TestCase):
    
    def test_set_global_seeds_default(self):
        """Test setting seeds with default value."""
        seed = set_global_seeds()
        self.assertEqual(seed, GLOBAL_SEED)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), str(GLOBAL_SEED))
    
    def test_set_global_seeds_custom(self):
        """Test setting seeds with custom value."""
        custom_seed = 123
        seed = set_global_seeds(custom_seed)
        self.assertEqual(seed, custom_seed)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), str(custom_seed))
    
    def test_python_random_reproducibility(self):
        """Test Python random reproducibility."""
        set_global_seeds(100)
        values1 = [random.random() for _ in range(5)]
        
        set_global_seeds(100)
        values2 = [random.random() for _ in range(5)]
        
        self.assertEqual(values1, values2)
    
    def test_numpy_random_reproducibility(self):
        """Test NumPy random reproducibility."""
        set_global_seeds(200)
        arr1 = np.random.rand(5)
        
        set_global_seeds(200)
        arr2 = np.random.rand(5)
        
        np.testing.assert_array_equal(arr1, arr2)
    
    def test_get_random_state_default(self):
        """Test get_random_state with default parameters."""
        state = get_random_state()
        self.assertEqual(state, GLOBAL_SEED)
    
    def test_get_random_state_with_offset(self):
        """Test get_random_state with offset."""
        offset = 10
        state = get_random_state(offset=offset)
        self.assertEqual(state, GLOBAL_SEED + offset)
    
    def test_get_random_state_custom_base(self):
        """Test get_random_state with custom base seed."""
        base = 500
        offset = 25
        state = get_random_state(base_seed=base, offset=offset)
        self.assertEqual(state, base + offset)
    
    def test_check_reproducibility_true(self):
        """Test check_reproducibility when PYTHONHASHSEED is set."""
        set_global_seeds()
        self.assertTrue(check_reproducibility())
    
    def test_check_reproducibility_false(self):
        """Test check_reproducibility when PYTHONHASHSEED is not set."""
        # Save current value
        old_value = os.environ.get("PYTHONHASHSEED")
        
        # Remove PYTHONHASHSEED
        if "PYTHONHASHSEED" in os.environ:
            del os.environ["PYTHONHASHSEED"]
        
        # Test
        with self.assertWarns(UserWarning):
            result = check_reproducibility()
        self.assertFalse(result)
        
        # Restore
        if old_value is not None:
            os.environ["PYTHONHASHSEED"] = old_value
    
    def test_different_offsets_produce_different_states(self):
        """Test that different offsets produce different random states."""
        states = [get_random_state(offset=i) for i in range(10)]
        # All states should be unique
        self.assertEqual(len(states), len(set(states)))

if __name__ == '__main__':
    unittest.main()