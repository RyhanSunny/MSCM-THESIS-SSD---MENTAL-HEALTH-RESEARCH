"""
Test for removal of synthetic socioeconomic status (SES) data.

Following TDD principles as per CLAUDE.md requirements.
"""

import pandas as pd
import pytest
from pathlib import Path
import sys
import tempfile
import shutil
import csv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ses_data_cleaner import (
    remove_synthetic_ices_marginals, 
    remove_ses_references_from_code,
    validate_no_synthetic_ses
)


class TestSESDataRemoval:
    """Test removal of synthetic SES data from pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create mock ices_marginals.csv with synthetic data
        self.ices_path = self.test_dir / "ices_marginals.csv"
        with open(self.ices_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['variable', 'category', 'proportion'])
            writer.writerow(['age_group', '18-34', '0.25'])
            writer.writerow(['sex', 'female', '0.52'])
            writer.writerow(['socioeconomic_quintile', 'q1_lowest', '0.20'])
            writer.writerow(['socioeconomic_quintile', 'q2', '0.20'])
            writer.writerow(['socioeconomic_quintile', 'q3', '0.20'])
            writer.writerow(['socioeconomic_quintile', 'q4', '0.20'])
            writer.writerow(['socioeconomic_quintile', 'q5_highest', '0.20'])
        
        # Create mock source file with SES references
        self.src_file = self.test_dir / "test_source.py"
        with open(self.src_file, 'w') as f:
            f.write("""
def analyze_data(df):
    # This function has SES references that should be removed
    ses_flag = df.get('deprivation_quintile', 3) >= 4
    high_ses = df['socioeconomic_quintile'] == 5
    return ses_flag and high_ses

def clean_function(df):
    # This function should remain unchanged
    age_flag = df['age'] > 65
    return age_flag
""")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_remove_synthetic_ices_marginals_function_exists(self):
        """Test that remove_synthetic_ices_marginals function exists."""
        # This should fail initially (TDD)
        from ses_data_cleaner import remove_synthetic_ices_marginals
        assert callable(remove_synthetic_ices_marginals)
    
    def test_remove_synthetic_ices_marginals_removes_ses_rows(self):
        """Test that function removes socioeconomic_quintile rows from CSV."""
        # Check original file has SES data
        df_before = pd.read_csv(self.ices_path)
        ses_rows_before = df_before[df_before['variable'] == 'socioeconomic_quintile']
        assert len(ses_rows_before) == 5
        
        # Remove synthetic SES data
        remove_synthetic_ices_marginals(self.ices_path)
        
        # Check SES data is removed
        df_after = pd.read_csv(self.ices_path)
        ses_rows_after = df_after[df_after['variable'] == 'socioeconomic_quintile']
        assert len(ses_rows_after) == 0
        
        # Check other data is preserved
        age_rows = df_after[df_after['variable'] == 'age_group']
        sex_rows = df_after[df_after['variable'] == 'sex']
        assert len(age_rows) == 1  # Only one age row in our test data
        assert len(sex_rows) == 1  # Only one sex row in our test data
    
    def test_remove_ses_references_from_code_function_exists(self):
        """Test that remove_ses_references_from_code function exists."""
        from ses_data_cleaner import remove_ses_references_from_code
        assert callable(remove_ses_references_from_code)
    
    def test_remove_ses_references_removes_deprivation_quintile(self):
        """Test that function removes deprivation_quintile references."""
        # Check original file has SES references
        with open(self.src_file, 'r') as f:
            content_before = f.read()
        assert 'deprivation_quintile' in content_before
        assert 'socioeconomic_quintile' in content_before
        
        # Remove SES references
        remove_ses_references_from_code(self.src_file)
        
        # Check SES references are removed/commented
        with open(self.src_file, 'r') as f:
            content_after = f.read()
        
        # Should comment out lines with SES references
        assert '# REMOVED SES:' in content_after
        assert 'deprivation_quintile' not in content_after or '# REMOVED SES:' in content_after
    
    def test_validate_no_synthetic_ses_function_exists(self):
        """Test that validate_no_synthetic_ses function exists."""
        from ses_data_cleaner import validate_no_synthetic_ses
        assert callable(validate_no_synthetic_ses)
    
    def test_validate_no_synthetic_ses_passes_when_clean(self):
        """Test validation passes when no synthetic SES data present."""
        # First remove SES data
        remove_synthetic_ices_marginals(self.ices_path)
        remove_ses_references_from_code(self.src_file)
        
        # Validation should pass
        result = validate_no_synthetic_ses(
            ices_path=self.ices_path,
            source_files=[self.src_file]
        )
        assert result == True
    
    def test_validate_no_synthetic_ses_fails_when_ses_present(self):
        """Test validation fails when synthetic SES data still present."""
        # Don't remove SES data - validation should fail
        result = validate_no_synthetic_ses(
            ices_path=self.ices_path,
            source_files=[self.src_file]
        )
        assert result == False
    
    def test_backup_created_before_modification(self):
        """Test that backup files are created before modification."""
        # Remove SES data
        remove_synthetic_ices_marginals(self.ices_path, backup=True)
        
        # Check backup was created
        backup_path = self.ices_path.with_suffix('.csv.backup')
        assert backup_path.exists()
        
        # Check backup contains original data
        df_backup = pd.read_csv(backup_path)
        ses_rows = df_backup[df_backup['variable'] == 'socioeconomic_quintile']
        assert len(ses_rows) == 5


# Run the failing test to ensure TDD compliance
if __name__ == "__main__":
    test = TestSESDataRemoval()
    test.setup_method()
    try:
        test.test_remove_synthetic_ices_marginals_function_exists()
        print("❌ TEST SHOULD FAIL (TDD requirement)")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"✓ Test fails as expected: {e}")
        print("Now implementing ses_data_cleaner module...")
    finally:
        test.teardown_method()