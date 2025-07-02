# CORRECTED Step 14.1: Placebo Tests for Causal Validity
# This version uses the correct arguments and treatment column

print("\n14.1: Running Placebo Tests...")
print("This performs falsification tests to validate causal assumptions")
print("Expected outcomes: No significant effects for placebo outcomes")

# Check if required data exists
import os
if os.path.exists("data_derived/ps_weighted.parquet"):
    print("✓ Required ps_weighted.parquet file exists")
    
    # Run placebo tests with MC-SIMEX adjusted treatment
    result = run_pipeline_script("14_placebo_tests.py",
                               args="--n-iterations 100 --treatment-col ssd_flag_adj",
                               description="Placebo Tests for Causal Validity (Adjusted)")
    
    if result['status'] != 'success':
        print(f"WARNING: Placebo tests failed: {result.get('error', 'Unknown error')}")
        
        # Try with original flag as fallback
        print("\nTrying with original ssd_flag...")
        result = run_pipeline_script("14_placebo_tests.py",
                                   args="--n-iterations 100 --treatment-col ssd_flag",
                                   description="Placebo Tests for Causal Validity (Original)")
    
    if result['status'] == 'success':
        print("✓ Placebo tests completed - check results/placebo_test_results.json")
else:
    print("ERROR: Required file data_derived/ps_weighted.parquet not found")
    print("The placebo tests require propensity score weighted data")
    print("Please ensure Step 13 (PS Matching) completed successfully")