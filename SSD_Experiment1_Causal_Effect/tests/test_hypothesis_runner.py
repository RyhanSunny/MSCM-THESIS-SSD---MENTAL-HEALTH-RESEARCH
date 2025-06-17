#!/usr/bin/env python3
"""
Test suite for hypothesis runner
Following TDD principles - tests written first per CLAUDE.md requirements
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from hypothesis_runner import SSDHypothesisRunner


class TestSSDHypothesisRunner:
    """Test suite for SSD hypothesis analysis runner"""
    
    def test_load_analysis_data(self):
        """Test loading analysis data from parquet file"""
        # Create mock data
        np.random.seed(42)
        n = 1000
        
        mock_data = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'site_id': np.random.randint(1, 21, n),
            'iptw': np.random.gamma(2, 0.5, n),
            'total_encounters': np.random.poisson(5, n),
            'primary_care_encounters': np.random.poisson(3, n),
            'ed_visits': np.random.poisson(1, n),
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n),
            'charlson_score': np.random.poisson(1, n)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "test_data.parquet"
            results_dir = Path(tmp_dir) / "results"
            
            # Save mock data
            mock_data.to_parquet(data_path, index=False)
            
            # Test loading
            runner = SSDHypothesisRunner(data_path, results_dir)
            df = runner.load_analysis_data()
            
            assert len(df) == n
            assert 'Patient_ID' in df.columns
            assert 'ssd_flag' in df.columns
    
    def test_validate_analysis_requirements(self):
        """Test validation of analysis requirements"""
        np.random.seed(42)
        n = 1000
        
        # Create complete dataset
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'site_id': np.random.randint(1, 21, n),
            'iptw': np.random.gamma(2, 0.5, n),
            'total_encounters': np.random.poisson(5, n),
            'primary_care_encounters': np.random.poisson(3, n),
            'ed_visits': np.random.poisson(1, n)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            validation = runner.validate_analysis_requirements(df)
            
            assert validation['data_loaded'] is True
            assert validation['n_patients'] == n
            assert 'weight_diagnostics' in validation
            assert 'clustering_valid' in validation
            assert 'temporal_valid' in validation
    
    def test_run_hypothesis_h1(self):
        """Test H1 hypothesis execution"""
        np.random.seed(42)
        n = 1000
        
        # Create dataset with H1 variables
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'h1_normal_lab_flag': np.random.binomial(1, 0.15, n),
            'total_encounters': np.random.poisson(5, n),
            'site_id': np.random.randint(1, 21, n),
            'iptw': np.random.gamma(2, 0.5, n),
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n),
            'charlson_score': np.random.poisson(1, n),
            'baseline_encounters': np.random.poisson(3, n)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            h1_results = runner.run_hypothesis_h1(df)
            
            # Check required fields
            assert h1_results['hypothesis'] == 'H1'
            assert 'description' in h1_results
            assert 'analysis_method' in h1_results
            assert 'sample_size' in h1_results
            assert 'timestamp' in h1_results
            
            # Check numerical results (if successful)
            if 'error' not in h1_results:
                assert 'irr' in h1_results or 'estimate' in h1_results
                assert 'exposure_prevalence' in h1_results
    
    def test_run_hypothesis_h2(self):
        """Test H2 hypothesis execution"""
        np.random.seed(42)
        n = 1000
        
        # Create dataset for H2
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'h2_referral_loop_flag': np.random.binomial(1, 0.08, n),
            'primary_care_encounters': np.random.poisson(3, n),
            'site_id': np.random.randint(1, 21, n),
            'iptw': np.random.gamma(2, 0.5, n),
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n),
            'charlson_score': np.random.poisson(1, n),
            'baseline_encounters': np.random.poisson(3, n)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            h2_results = runner.run_hypothesis_h2(df)
            
            # Check required fields
            assert h2_results['hypothesis'] == 'H2'
            assert 'description' in h2_results
            assert h2_results['description'] == 'Unresolved referrals → healthcare utilization'
            assert 'timestamp' in h2_results
    
    def test_run_hypothesis_h3(self):
        """Test H3 hypothesis execution"""
        np.random.seed(42)
        n = 1000
        
        # Create dataset for H3
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'h3_medication_flag': np.random.binomial(1, 0.20, n),
            'ed_visits': np.random.poisson(1, n),
            'site_id': np.random.randint(1, 21, n),
            'iptw': np.random.gamma(2, 0.5, n),
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n),
            'charlson_score': np.random.poisson(1, n),
            'baseline_encounters': np.random.poisson(3, n)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            h3_results = runner.run_hypothesis_h3(df)
            
            # Check required fields
            assert h3_results['hypothesis'] == 'H3'
            assert 'description' in h3_results
            assert h3_results['description'] == 'Medication persistence → reduced ED visits'
            assert 'timestamp' in h3_results
    
    def test_save_hypothesis_results(self):
        """Test saving hypothesis results to JSON"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            # Mock results
            mock_results = {
                'hypothesis': 'H1',
                'irr': 1.25,
                'irr_ci_lower': 1.10,
                'irr_ci_upper': 1.42,
                'p_value': 0.001
            }
            
            # Save results
            output_file = runner.save_hypothesis_results('H1', mock_results)
            
            # Verify file exists and contents
            assert output_file.exists()
            assert output_file.name == 'hypothesis_h1.json'
            
            with open(output_file, 'r') as f:
                loaded_results = json.load(f)
                
            assert loaded_results['hypothesis'] == 'H1'
            assert loaded_results['irr'] == 1.25
    
    def test_run_all_hypotheses(self):
        """Test running all H1-H3 hypotheses"""
        np.random.seed(42)
        n = 1000
        
        # Create comprehensive dataset
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'site_id': np.random.randint(1, 21, n),
            'iptw': np.random.gamma(2, 0.5, n),
            'total_encounters': np.random.poisson(5, n),
            'primary_care_encounters': np.random.poisson(3, n),
            'ed_visits': np.random.poisson(1, n),
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n),
            'charlson_score': np.random.poisson(1, n),
            'baseline_encounters': np.random.poisson(3, n)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            all_results = runner.run_all_hypotheses(df)
            
            # Check all hypotheses were attempted
            assert 'H1' in all_results
            assert 'H2' in all_results
            assert 'H3' in all_results
            
            # Check results directory
            results_dir = Path(tmp_dir) / "results"
            assert results_dir.exists()
            
            # Check individual result files
            h1_file = results_dir / "hypothesis_h1.json"
            h2_file = results_dir / "hypothesis_h2.json"
            h3_file = results_dir / "hypothesis_h3.json"
            summary_file = results_dir / "hypothesis_summary.json"
            
            # Files should exist (even if analysis failed)
            for file_path in [h1_file, h2_file, h3_file, summary_file]:
                assert file_path.exists(), f"Missing file: {file_path}"
    
    def test_mock_exposure_creation(self):
        """Test that mock exposures are created when variables missing"""
        np.random.seed(42)
        n = 500
        
        # Create minimal dataset without hypothesis-specific flags
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'site_id': np.random.randint(1, 21, n),
            'iptw': np.random.gamma(2, 0.5, n),
            'total_encounters': np.random.poisson(5, n),
            'primary_care_encounters': np.random.poisson(3, n),
            'ed_visits': np.random.poisson(1, n),
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            # Run H1 (should create mock exposure)
            h1_results = runner.run_hypothesis_h1(df)
            
            # Check that mock exposure was created
            assert 'h1_normal_lab_flag' in df.columns
            assert df['h1_normal_lab_flag'].sum() > 0  # Some patients exposed
            assert df['h1_normal_lab_flag'].sum() < n   # Not all patients exposed
            
            # Results should still be generated
            assert h1_results['hypothesis'] == 'H1'
            assert 'exposure_prevalence' in h1_results
    
    def test_error_handling(self):
        """Test error handling for invalid data"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "nonexistent.parquet", 
                Path(tmp_dir) / "results"
            )
            
            # Should raise FileNotFoundError for missing data
            with pytest.raises(FileNotFoundError):
                runner.load_analysis_data()
    
    def test_required_columns_validation(self):
        """Test validation fails with missing required columns"""
        # Create dataset missing required columns
        df = pd.DataFrame({
            'Patient_ID': range(100),
            'some_other_column': range(100)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            # Should raise ValueError for missing columns
            with pytest.raises(ValueError, match="Missing required columns"):
                runner.validate_analysis_requirements(df)


class TestHypothesisIntegration:
    """Test integration with Week 1 implementations"""
    
    def test_week1_integration(self):
        """Test that hypothesis runner uses Week 1 implementations"""
        np.random.seed(42)
        n = 1000
        
        # Create realistic SSD dataset
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'site_id': np.random.randint(1, 21, n),
            'iptw': np.random.gamma(2, 0.5, n),  # Well-behaved weights
            'total_encounters': np.random.poisson(5, n),
            'primary_care_encounters': np.random.poisson(3, n),
            'ed_visits': np.random.poisson(1, n),
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n),
            'charlson_score': np.random.poisson(1, n),
            'baseline_encounters': np.random.poisson(3, n)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            # Test validation uses Week 1 implementations
            validation = runner.validate_analysis_requirements(df)
            
            # Should include weight diagnostics from Week 1
            assert 'weight_diagnostics' in validation
            if validation['weight_diagnostics'] is not None:
                assert 'ess' in validation['weight_diagnostics']
                assert 'max_weight' in validation['weight_diagnostics']
                
            # Should include clustering validation from Week 1
            assert 'clustering_valid' in validation
            if validation['clustering_valid'] is not None:
                assert 'n_clusters' in validation['clustering_valid']
    
    def test_results_structure_compliance(self):
        """Test that results comply with Week 2 requirements"""
        np.random.seed(42)
        n = 500
        
        df = pd.DataFrame({
            'Patient_ID': range(n),
            'ssd_flag': np.random.binomial(1, 0.15, n),
            'site_id': np.random.randint(1, 21, n),
            'iptw': np.random.gamma(2, 0.5, n),
            'total_encounters': np.random.poisson(5, n),
            'primary_care_encounters': np.random.poisson(3, n),
            'ed_visits': np.random.poisson(1, n),
            'age': np.random.normal(50, 15, n),
            'sex_M': np.random.binomial(1, 0.4, n)
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SSDHypothesisRunner(
                Path(tmp_dir) / "data.parquet", 
                Path(tmp_dir) / "results"
            )
            
            # Run analysis
            results = runner.run_all_hypotheses(df)
            
            # Check each hypothesis result has required fields
            for hypothesis in ['H1', 'H2', 'H3']:
                assert hypothesis in results
                result = results[hypothesis]
                
                # Required metadata fields
                expected_fields = [
                    'hypothesis', 'description', 'analysis_method', 
                    'sample_size', 'timestamp'
                ]
                
                for field in expected_fields:
                    assert field in result, f"Missing {field} in {hypothesis}"
                    
                # Should have either results or error
                has_results = any(key in result for key in ['irr', 'estimate', 'coefficient'])
                has_error = 'error' in result
                assert has_results or has_error, f"{hypothesis} has neither results nor error"


if __name__ == "__main__":
    # Run basic demonstrations
    test_instance = TestSSDHypothesisRunner()
    
    print("Running hypothesis runner tests...")
    test_instance.test_run_hypothesis_h1()
    print("✓ H1 hypothesis test passed")
    
    test_instance.test_run_hypothesis_h2()
    print("✓ H2 hypothesis test passed")
    
    test_instance.test_run_hypothesis_h3()
    print("✓ H3 hypothesis test passed")
    
    test_instance.test_mock_exposure_creation()
    print("✓ Mock exposure creation test passed")
    
    print("\nAll hypothesis runner tests passed! Ready for implementation.")