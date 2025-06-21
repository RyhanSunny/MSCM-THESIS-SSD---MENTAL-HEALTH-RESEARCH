"""
Test for MC-SIMEX flag integration with patient master table.

Following TDD principles as per CLAUDE.md requirements.
"""

import pandas as pd
import pytest
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_simex_flag_merger import merge_bias_corrected_flag


class TestMCSimexFlagIntegration:
    """Test MC-SIMEX flag integration with patient master table."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create mock patient_master.parquet
        self.master_data = pd.DataFrame({
            'patient_id': [1, 2, 3, 4, 5],
            'ssd_flag': [1, 0, 1, 0, 1],
            'age': [45, 32, 67, 28, 55],
            'sex': ['F', 'M', 'F', 'M', 'F']
        })
        
        # Create mock cohort_bias_corrected.parquet
        self.corrected_data = pd.DataFrame({
            'patient_id': [1, 2, 3, 4, 5],
            'ssd_flag': [1, 0, 1, 0, 1],
            'ssd_flag_adj': [0, 0, 1, 1, 1],  # Different from original
            'age': [45, 32, 67, 28, 55],
            'sex': ['F', 'M', 'F', 'M', 'F']
        })
        
        self.master_path = self.test_dir / "patient_master.parquet"
        self.corrected_path = self.test_dir / "cohort_bias_corrected.parquet"
        
        self.master_data.to_parquet(self.master_path, index=False)
        self.corrected_data.to_parquet(self.corrected_path, index=False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_merge_bias_corrected_flag_function_exists(self):
        """Test that merge_bias_corrected_flag function exists."""
        # This should fail initially (TDD)
        from mc_simex_flag_merger import merge_bias_corrected_flag
        assert callable(merge_bias_corrected_flag)
    
    def test_merge_adds_ssd_flag_adj_to_master(self):
        """Test that merge adds ssd_flag_adj column to patient_master.parquet."""
        # Load original master
        master_before = pd.read_parquet(self.master_path)
        assert 'ssd_flag_adj' not in master_before.columns
        
        # Perform merge
        merge_bias_corrected_flag(
            master_path=self.master_path,
            corrected_path=self.corrected_path
        )
        
        # Check result
        master_after = pd.read_parquet(self.master_path)
        assert 'ssd_flag_adj' in master_after.columns
        
        # Verify the corrected flags are properly merged
        expected_adj_flags = self.corrected_data['ssd_flag_adj'].tolist()
        actual_adj_flags = master_after['ssd_flag_adj'].tolist()
        assert actual_adj_flags == expected_adj_flags
    
    def test_merge_preserves_all_original_columns(self):
        """Test that merge preserves all original columns in patient_master."""
        master_before = pd.read_parquet(self.master_path)
        original_columns = set(master_before.columns)
        
        merge_bias_corrected_flag(
            master_path=self.master_path,
            corrected_path=self.corrected_path
        )
        
        master_after = pd.read_parquet(self.master_path)
        after_columns = set(master_after.columns)
        
        # All original columns should be preserved
        assert original_columns.issubset(after_columns)
        
        # Should have one additional column: ssd_flag_adj
        assert len(after_columns) == len(original_columns) + 1
        assert 'ssd_flag_adj' in after_columns
    
    def test_merge_preserves_row_count(self):
        """Test that merge preserves the same number of rows."""
        master_before = pd.read_parquet(self.master_path)
        original_rows = len(master_before)
        
        merge_bias_corrected_flag(
            master_path=self.master_path,
            corrected_path=self.corrected_path
        )
        
        master_after = pd.read_parquet(self.master_path)
        assert len(master_after) == original_rows
    
    def test_merge_handles_missing_corrected_file(self):
        """Test that merge handles missing bias-corrected file gracefully."""
        non_existent_path = self.test_dir / "nonexistent.parquet"
        
        with pytest.raises(FileNotFoundError):
            merge_bias_corrected_flag(
                master_path=self.master_path,
                corrected_path=non_existent_path
            )
    
    def test_merge_validates_patient_id_alignment(self):
        """Test that merge validates patient IDs match between files."""
        # Create misaligned corrected data
        misaligned_data = self.corrected_data.copy()
        misaligned_data['patient_id'] = [1, 2, 3, 4, 99]  # Different patient_id
        
        misaligned_path = self.test_dir / "misaligned.parquet"
        misaligned_data.to_parquet(misaligned_path, index=False)
        
        with pytest.raises(ValueError, match="Patient IDs do not match"):
            merge_bias_corrected_flag(
                master_path=self.master_path,
                corrected_path=misaligned_path
            )


# Run the failing test to ensure TDD compliance
if __name__ == "__main__":
    test = TestMCSimexFlagIntegration()
    test.setup_method()
    try:
        test.test_merge_bias_corrected_flag_function_exists()
        print("❌ TEST SHOULD FAIL (TDD requirement)")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"✓ Test fails as expected: {e}")
        print("Now implementing mc_simex_flag_merger module...")
    finally:
        test.teardown_method()