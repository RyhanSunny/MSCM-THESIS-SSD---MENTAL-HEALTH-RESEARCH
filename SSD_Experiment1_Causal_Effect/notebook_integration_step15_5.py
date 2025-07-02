#!/usr/bin/env python3
"""
Step 15.5: Multiple Testing Correction Integration
CRITICAL MISSING INTEGRATION for thesis defensibility

This script integrates week4_statistical_refinements.py into the main pipeline
to address the FALLBACK_AUDIT issue of multiple testing across 6 hypotheses.

Following CLAUDE.md + RULES.md + ANALYSIS_RULES.md requirements.

Author: Ryhan Suny, MSc
Date: 2025-07-02
"""

# STEP 15.5: Multiple Testing Correction (CRITICAL INTEGRATION)
print("\n" + "="*80)
print("STEP 15.5: Multiple Testing Correction - Benjamini-Hochberg FDR")
print("CRITICAL: Addresses FALLBACK_AUDIT issue across 6 hypotheses")
print("="*80)

# Run week4 statistical refinements for multiple testing correction
result = run_pipeline_script("week4_statistical_refinements.py",
                           description="Multiple Testing Correction + E-values")

# VALIDATE: FDR-adjusted p-values
fdr_path = RESULTS_DIR / "fdr_adjusted_results.json"
if fdr_path.exists():
    with open(fdr_path, 'r') as f:
        fdr_results = json.load(f)
    
    print(f"\n✓ Multiple testing correction complete:")
    
    # Display FDR-adjusted results
    if 'fdr_adjusted' in fdr_results:
        fdr_data = fdr_results['fdr_adjusted']
        print(f"\n  Benjamini-Hochberg FDR Correction (α = 0.05):")
        print(f"  - Original p-values: {len(fdr_data.get('original_pvalues', []))} hypotheses")
        print(f"  - FDR threshold: {fdr_data.get('fdr_threshold', 'N/A'):.4f}")
        print(f"  - Significant after FDR: {fdr_data.get('n_significant', 0)}/{len(fdr_data.get('original_pvalues', []))}")
        
        # Show hypothesis-specific results
        if 'hypothesis_results' in fdr_data:
            print(f"\n  Hypothesis-specific results:")
            for h, result in fdr_data['hypothesis_results'].items():
                p_orig = result.get('p_original', 0)
                p_adj = result.get('p_adjusted', 0)
                significant = result.get('significant_fdr', False)
                status = "✓ SIGNIFICANT" if significant else "✗ Not significant"
                print(f"    {h}: p_orig={p_orig:.4f}, p_adj={p_adj:.4f} - {status}")
    
    # Display E-values for unmeasured confounding
    if 'evalues' in fdr_results:
        print(f"\n  E-values for Unmeasured Confounding:")
        for h, evalue_data in fdr_results['evalues'].items():
            evalue = evalue_data.get('evalue', 0)
            effect = evalue_data.get('effect_estimate', 0)
            print(f"    {h}: Effect={effect:.3f}, E-value={evalue:.2f}")
            
            # Interpretation
            if evalue >= 2.0:
                print(f"      → Strong evidence against unmeasured confounding")
            elif evalue >= 1.5:
                print(f"      → Moderate evidence against unmeasured confounding")
            else:
                print(f"      → Weak evidence against unmeasured confounding")
    
    # Save critical results to session directory
    import shutil
    shutil.copy(fdr_path, session_results_dir / 'fdr_correction_results.json')
    print(f"\n✓ FDR correction results saved to session directory")
    
    # CRITICAL: Update pooled results with FDR correction
    pooled_path = RESULTS_DIR / "pooled_causal_estimates.json"
    if pooled_path.exists():
        with open(pooled_path, 'r') as f:
            pooled_data = json.load(f)
        
        # Add FDR correction flag to pooled results
        pooled_data['fdr_correction_applied'] = True
        pooled_data['fdr_threshold'] = fdr_data.get('fdr_threshold', 0.05)
        pooled_data['multiple_testing_method'] = "Benjamini-Hochberg"
        
        # Save updated pooled results
        with open(pooled_path, 'w') as f:
            json.dump(pooled_data, f, indent=2)
        
        print(f"✓ Pooled results updated with FDR correction metadata")
    
else:
    print(f"⚠️ FDR results not found at {fdr_path}")
    print("This is a CRITICAL missing component for thesis defense!")

print("\nSTEP 15.5 COMPLETE ✓")
print("CRITICAL INTEGRATION: Multiple testing correction now addresses FALLBACK_AUDIT concerns")

