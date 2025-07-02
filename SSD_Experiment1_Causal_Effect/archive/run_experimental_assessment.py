#!/usr/bin/env python3
"""
Assessment Script for Experimental Modules
Author: Ryhan Suny
Date: June 21, 2025

This script runs the experimental modules and generates a recommendation
for whether they should be integrated into the main pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
DATA_DERIVED = ROOT / "data_derived"
SRC_DIR = ROOT / "src"

# Add src to path
sys.path.insert(0, str(SRC_DIR))

def analyze_exposure_differences():
    """Compare base vs enhanced exposure flags"""
    logger.info("\n=== EXPOSURE ANALYSIS ===")
    
    try:
        # Load base exposure
        base_exposure = pd.read_parquet(DATA_DERIVED / "exposure.parquet")
        
        # Check if enhanced exposure exists
        enhanced_path = DATA_DERIVED / "exposure_enhanced.parquet"
        if enhanced_path.exists():
            enhanced_exposure = pd.read_parquet(enhanced_path)
            
            # Compare H3 drug persistence
            base_h3 = base_exposure['H3_drug_persistence'].sum()
            enhanced_h3 = enhanced_exposure['H3_drug_persistence_enhanced'].sum()
            change_pct = (enhanced_h3 / base_h3 - 1) * 100 if base_h3 > 0 else 0
            
            logger.info(f"H3 Drug Persistence Comparison:")
            logger.info(f"  Base (90 days): {base_h3:,} patients")
            logger.info(f"  Enhanced (180 days): {enhanced_h3:,} patients")
            logger.info(f"  Change: {enhanced_h3 - base_h3:,} ({change_pct:+.1f}%)")
            
            return {
                'base_h3': base_h3,
                'enhanced_h3': enhanced_h3,
                'change_pct': change_pct,
                'significant_change': abs(change_pct) > 10
            }
        else:
            logger.warning("Enhanced exposure file not found")
            return None
    except Exception as e:
        logger.error(f"Error analyzing exposure: {e}")
        return None

def analyze_referral_patterns():
    """Analyze referral sequence enhancements"""
    logger.info("\n=== REFERRAL ANALYSIS ===")
    
    try:
        # Check if enhanced referral exists
        enhanced_path = DATA_DERIVED / "referral_enhanced.parquet"
        pathway_path = DATA_DERIVED / "referral_pathway_detail.parquet"
        
        if enhanced_path.exists():
            enhanced_ref = pd.read_parquet(enhanced_path)
            
            total = len(enhanced_ref)
            dual_pathway = enhanced_ref['dual_pathway'].sum() if 'dual_pathway' in enhanced_ref else 0
            psych_ref = enhanced_ref['has_psychiatric_referral'].sum() if 'has_psychiatric_referral' in enhanced_ref else 0
            
            logger.info(f"Referral Pattern Analysis:")
            logger.info(f"  Total patients: {total:,}")
            logger.info(f"  Psychiatric referrals: {psych_ref:,} ({psych_ref/total*100:.1f}%)")
            logger.info(f"  Dual pathway (medical+psych): {dual_pathway:,} ({dual_pathway/total*100:.1f}%)")
            
            return {
                'total': total,
                'psychiatric': psych_ref,
                'dual_pathway': dual_pathway,
                'clinically_relevant': dual_pathway > total * 0.05  # >5% is clinically relevant
            }
        else:
            logger.warning("Enhanced referral file not found")
            return None
    except Exception as e:
        logger.error(f"Error analyzing referrals: {e}")
        return None

def generate_recommendation():
    """Generate recommendation for experimental module integration"""
    logger.info("\n=== INTEGRATION RECOMMENDATION ===")
    
    exposure_results = analyze_exposure_differences()
    referral_results = analyze_referral_patterns()
    
    recommendations = []
    
    # Exposure enhancement recommendation
    if exposure_results and exposure_results['significant_change']:
        recommendations.append({
            'module': 'Enhanced Exposure (180-day threshold)',
            'recommendation': 'INTEGRATE',
            'reason': f"Significant change ({exposure_results['change_pct']:+.1f}%) in H3 criterion",
            'action': 'Use as sensitivity analysis or primary analysis after clinical validation'
        })
    else:
        recommendations.append({
            'module': 'Enhanced Exposure',
            'recommendation': 'KEEP EXPERIMENTAL',
            'reason': 'Minimal impact on cohort size',
            'action': 'Monitor for future use'
        })
    
    # Referral enhancement recommendation
    if referral_results and referral_results['clinically_relevant']:
        recommendations.append({
            'module': 'Enhanced Referral (Psychiatric pathways)',
            'recommendation': 'INTEGRATE',
            'reason': f"Clinically relevant dual pathway pattern ({referral_results['dual_pathway']:,} patients)",
            'action': 'Add as additional analysis in main pipeline'
        })
    else:
        recommendations.append({
            'module': 'Enhanced Referral',
            'recommendation': 'KEEP EXPERIMENTAL',
            'reason': 'Limited dual pathway prevalence',
            'action': 'Use for exploratory analyses only'
        })
    
    # NYD enhancement is always experimental until validated
    recommendations.append({
        'module': 'Enhanced Cohort (NYD flags)',
        'recommendation': 'KEEP EXPERIMENTAL',
        'reason': 'Requires clinical validation of body part mapping',
        'action': 'Validate mapping with clinical experts before integration'
    })
    
    return recommendations

def create_integration_report(recommendations):
    """Create markdown report with recommendations"""
    report = f"""# Experimental Module Assessment Report
Date: June 21, 2025
Author: SSD Pipeline Assessment

## Executive Summary

This report assesses three experimental modules for potential integration into the main SSD pipeline.

## Current Architecture

**Base Pipeline**: Sequential data flow
- cohort → exposure → master → analysis

**Experimental Modules**: Independent from base data
- Each reads from base parquet files
- No sequential dependencies between experimental modules

## Module Assessments

"""
    
    for rec in recommendations:
        report += f"### {rec['module']}\n"
        report += f"**Recommendation**: {rec['recommendation']}\n"
        report += f"**Reason**: {rec['reason']}\n"
        report += f"**Action**: {rec['action']}\n\n"
    
    report += """## Integration Strategy

Based on the assessment, we recommend:

1. **Keep Current Architecture**: Experimental modules remain independent
   - Allows A/B testing of enhancements
   - Maintains pipeline stability
   - Easy rollback if issues arise

2. **Selective Integration**: Only integrate modules that show:
   - Significant clinical impact (>10% change in cohort)
   - Validated clinical relevance
   - Stable implementation

3. **Future Sequential Pipeline**: Consider after:
   - Clinical validation of all enhancements
   - Proven value in real analyses
   - Consensus from research team

## Next Steps

1. Run full pipeline with base modules: `make all`
2. Run experimental modules separately: `make all_enhanced`
3. Compare results using validation scripts
4. Present findings to clinical team for validation
"""
    
    return report

if __name__ == "__main__":
    logger.info("Starting experimental module assessment...")
    
    # Generate recommendations
    recommendations = generate_recommendation()
    
    # Create report
    report = create_integration_report(recommendations)
    
    # Save report
    report_path = ROOT / "reports" / "experimental_module_assessment.md"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"\nAssessment complete! Report saved to: {report_path}")
    
    # Print summary
    logger.info("\n=== SUMMARY ===")
    for rec in recommendations:
        logger.info(f"{rec['module']}: {rec['recommendation']}") 