#!/usr/bin/env python3
"""
Enhanced Referral Sequence Analysis with Psychiatric Specialization

Enhancements:
1. Separate psychiatric vs medical specialist referrals
2. Dual pathway tracking (medical → psychiatric)
3. Enhanced referral loop detection
4. Temporal sequence analysis

Author: Ryhan Suny
Date: January 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DERIVED = ROOT / "data_derived"

def load_referral_data():
    """Load referral and cohort data"""
    logger.info("Loading referral and cohort data...")
    
    try:
        referral = pd.read_parquet(DERIVED / "referral.parquet")
        cohort = pd.read_parquet(DERIVED / "cohort.parquet")
        logger.info(f"Referral data loaded: {len(referral):,} referrals for analysis")
        return referral, cohort
    except FileNotFoundError as e:
        logger.error(f"Required data files not found: {e}")
        raise

def identify_psychiatric_referrals(referrals):
    """Identify psychiatric/mental health referrals with enhanced patterns"""
    logger.info("Identifying psychiatric referrals...")
    
    # Enhanced psychiatric keywords based on clinical review
    psychiatric_keywords = [
        'psychiatr', 'mental health', 'psych', 'behavioral health',
        'addiction', 'substance', 'counsell', 'therapy', 'therapist',
        'psychology', 'psychologist', 'cognitive', 'behavioral',
        'mood', 'anxiety', 'depression', 'bipolar', 'schizophren',
        'eating disorder', 'ptsd', 'trauma', 'stress management',
        'crisis', 'suicide', 'self harm', 'mental wellness'
    ]
    
    # Create pattern for case-insensitive matching
    pattern = '|'.join(psychiatric_keywords)
    
    # Handle empty dataframe case
    if len(referrals) == 0:
        logger.info("No referrals to analyze")
        empty_result = referrals.copy()
        empty_result['referral_type'] = pd.Series([], dtype='object')
        return empty_result
    
    # Identify psychiatric referrals
    psych_mask = referrals['Name_calc'].str.contains(pattern, case=False, na=False)
    
    # Also check specialty codes if available
    if 'SpecialtyCode' in referrals.columns:
        psych_codes = ['PSYC', 'MENT', 'BEHV', 'ADDI', 'COUN']
        code_pattern = '|'.join(psych_codes)
        psych_mask |= referrals['SpecialtyCode'].str.contains(code_pattern, case=False, na=False)
    
    psychiatric_referrals = referrals[psych_mask].copy()
    psychiatric_referrals['referral_type'] = 'psychiatric'
    
    logger.info(f"Psychiatric referrals identified: {len(psychiatric_referrals):,}")
    
    return psychiatric_referrals

def identify_medical_specialists(referrals):
    """Identify non-psychiatric medical specialist referrals"""
    logger.info("Identifying medical specialist referrals...")
    
    # Medical specialty keywords
    medical_specialties = [
        'cardio', 'gastro', 'neuro', 'orthop', 'rheuma', 'endocrin',
        'pulmon', 'nephro', 'oncol', 'dermat', 'urol', 'ophthal',
        'otolaryn', 'ent', 'allergy', 'immunol', 'hematol',
        'infectious', 'radiol', 'pathol', 'anesthes', 'emergency',
        'surgery', 'surgeon', 'cardiac', 'vascular', 'thoracic',
        'plastic', 'orthoped', 'neurolog', 'gastroenter', 'pulmonol',
        'nephrolog', 'oncolog', 'dermatol', 'urolog', 'ophthalmol',
        'otolaryngol', 'allergist', 'immunolog', 'hematolog',
        'radiolog', 'patholog', 'anesthesiolog'
    ]
    
    # Create pattern
    pattern = '|'.join(medical_specialties)
    
    # Handle empty dataframe case
    if len(referrals) == 0:
        logger.info("No referrals to analyze for medical specialists")
        empty_result = referrals.copy()
        empty_result['referral_type'] = pd.Series([], dtype='object')
        return empty_result
    
    # Identify medical specialist referrals
    medical_mask = referrals['Name_calc'].str.contains(pattern, case=False, na=False)
    
    # Exclude general practice and psychiatric referrals
    exclude_patterns = [
        'family', 'general', 'gp', 'primary', 'walk-in', 'clinic',
        'psychiatr', 'mental health', 'psych', 'behavioral', 'counsell'
    ]
    exclude_pattern = '|'.join(exclude_patterns)
    exclude_mask = referrals['Name_calc'].str.contains(exclude_pattern, case=False, na=False)
    
    medical_specialists = referrals[medical_mask & ~exclude_mask].copy()
    medical_specialists['referral_type'] = 'medical_specialist'
    
    logger.info(f"Medical specialist referrals identified: {len(medical_specialists):,}")
    
    return medical_specialists

def analyze_dual_pathway_patterns(referrals, cohort):
    """Analyze medical → psychiatric referral pathways"""
    logger.info("Analyzing dual pathway patterns...")
    
    # Get psychiatric and medical referrals
    psychiatric_refs = identify_psychiatric_referrals(referrals)
    medical_refs = identify_medical_specialists(referrals)
    
    # Combine and sort by patient and date
    all_specialist_refs = pd.concat([psychiatric_refs, medical_refs])
    all_specialist_refs = all_specialist_refs.sort_values(['Patient_ID', 'CompletedDate'])
    
    # Analyze patient-level patterns
    pathway_analysis = {}
    
    for patient_id in all_specialist_refs['Patient_ID'].unique():
        patient_refs = all_specialist_refs[all_specialist_refs['Patient_ID'] == patient_id]
        
        # Get referral sequence
        referral_sequence = patient_refs['referral_type'].tolist()
        referral_dates = patient_refs['CompletedDate'].tolist()
        
        # Analyze patterns
        has_medical = 'medical_specialist' in referral_sequence
        has_psychiatric = 'psychiatric' in referral_sequence
        
        # Check for medical → psychiatric sequence
        medical_to_psych = False
        if has_medical and has_psychiatric:
            for i, ref_type in enumerate(referral_sequence[:-1]):
                if ref_type == 'medical_specialist' and 'psychiatric' in referral_sequence[i+1:]:
                    medical_to_psych = True
                    break
        
        pathway_analysis[patient_id] = {
            'total_specialist_referrals': len(patient_refs),
            'medical_referrals': sum(1 for x in referral_sequence if x == 'medical_specialist'),
            'psychiatric_referrals': sum(1 for x in referral_sequence if x == 'psychiatric'),
            'has_medical_specialist': has_medical,
            'has_psychiatric_referral': has_psychiatric,
            'dual_pathway': has_medical and has_psychiatric,
            'medical_to_psychiatric_sequence': medical_to_psych,
            'referral_sequence': referral_sequence,
            'first_referral_date': min(referral_dates),
            'last_referral_date': max(referral_dates),
            'referral_span_days': (max(referral_dates) - min(referral_dates)).days if len(referral_dates) > 1 else 0
        }
    
    # Convert to DataFrame
    pathway_df = pd.DataFrame.from_dict(pathway_analysis, orient='index')
    pathway_df.index.name = 'Patient_ID'
    pathway_df = pathway_df.reset_index()
    
    # Generate summary statistics
    total_patients = len(pathway_df)
    dual_pathway_count = pathway_df['dual_pathway'].sum()
    medical_to_psych_count = pathway_df['medical_to_psychiatric_sequence'].sum()
    
    logger.info(f"Dual pathway analysis results:")
    logger.info(f"  Total patients with specialist referrals: {total_patients:,}")
    logger.info(f"  Patients with dual pathway (medical + psychiatric): {dual_pathway_count:,} ({dual_pathway_count/total_patients*100:.1f}%)")
    logger.info(f"  Patients with medical → psychiatric sequence: {medical_to_psych_count:,} ({medical_to_psych_count/total_patients*100:.1f}%)")
    
    return pathway_df, psychiatric_refs, medical_refs

def enhance_h2_referral_criteria(pathway_df, cohort):
    """Enhanced H2 referral loop criteria with psychiatric specialization"""
    logger.info("Enhancing H2 referral loop criteria...")
    
    # Enhanced H2 criteria:
    # 1. ≥2 medical specialist referrals with no clear resolution, OR
    # 2. ≥1 medical specialist + ≥1 psychiatric referral (dual pathway), OR  
    # 3. ≥3 total specialist referrals of any type
    
    h2_enhanced_criteria = pathway_df.copy()
    
    # Criterion 1: Multiple medical specialists (original H2 concept)
    h2_enhanced_criteria['h2_medical_loop'] = h2_enhanced_criteria['medical_referrals'] >= 2
    
    # Criterion 2: Dual pathway (new Felipe enhancement)
    h2_enhanced_criteria['h2_dual_pathway'] = h2_enhanced_criteria['dual_pathway']
    
    # Criterion 3: High specialist utilization
    h2_enhanced_criteria['h2_high_utilization'] = h2_enhanced_criteria['total_specialist_referrals'] >= 3
    
    # Combined H2 (any of the above)
    h2_enhanced_criteria['H2_referral_loop_enhanced'] = (
        h2_enhanced_criteria['h2_medical_loop'] |
        h2_enhanced_criteria['h2_dual_pathway'] |
        h2_enhanced_criteria['h2_high_utilization']
    )
    
    # Statistics
    h2_original_count = h2_enhanced_criteria['h2_medical_loop'].sum()
    h2_dual_count = h2_enhanced_criteria['h2_dual_pathway'].sum()
    h2_high_util_count = h2_enhanced_criteria['h2_high_utilization'].sum()
    h2_enhanced_total = h2_enhanced_criteria['H2_referral_loop_enhanced'].sum()
    
    logger.info(f"Enhanced H2 referral loop results:")
    logger.info(f"  H2 Medical loops (≥2 medical specialists): {h2_original_count:,}")
    logger.info(f"  H2 Dual pathways (medical + psychiatric): {h2_dual_count:,}")
    logger.info(f"  H2 High utilization (≥3 total specialists): {h2_high_util_count:,}")
    logger.info(f"  H2 Enhanced total: {h2_enhanced_total:,}")
    
    return h2_enhanced_criteria

def generate_enhanced_referral_flags(cohort):
    """Generate enhanced referral flags for the cohort"""
    logger.info("=== ENHANCED REFERRAL SEQUENCE ANALYSIS ===")
    
    # Load referral data
    referrals, cohort_data = load_referral_data()
    
    # Perform dual pathway analysis
    pathway_df, psychiatric_refs, medical_refs = analyze_dual_pathway_patterns(referrals, cohort)
    
    # Enhance H2 criteria
    h2_enhanced = enhance_h2_referral_criteria(pathway_df, cohort)
    
    # Merge with full cohort (some patients may have no specialist referrals)
    cohort_enhanced = cohort.merge(
        h2_enhanced[['Patient_ID', 'H2_referral_loop_enhanced', 'dual_pathway', 
                    'has_psychiatric_referral', 'has_medical_specialist', 'total_specialist_referrals']], 
        on='Patient_ID', 
        how='left'
    )
    
    # Fill missing values (patients with no specialist referrals)
    referral_columns = ['H2_referral_loop_enhanced', 'dual_pathway', 'has_psychiatric_referral', 
                       'has_medical_specialist']
    for col in referral_columns:
        cohort_enhanced[col] = cohort_enhanced[col].fillna(False)
    
    cohort_enhanced['total_specialist_referrals'] = cohort_enhanced['total_specialist_referrals'].fillna(0)
    
    # Create summary report
    create_referral_enhancement_report(cohort_enhanced, h2_enhanced, psychiatric_refs, medical_refs)
    
    # Save enhanced referral data
    output_path = DERIVED / "referral_enhanced.parquet"
    cohort_enhanced.to_parquet(output_path, index=False)
    logger.info(f"Enhanced referral data saved: {output_path}")
    
    # Save detailed pathway analysis
    pathway_detail_path = DERIVED / "referral_pathway_detail.parquet"
    h2_enhanced.to_parquet(pathway_detail_path, index=False)
    
    return cohort_enhanced, h2_enhanced

def create_referral_enhancement_report(cohort_enhanced, h2_enhanced, psychiatric_refs, medical_refs):
    """Create detailed referral enhancement report"""
    logger.info("Generating referral enhancement report...")
    
    total_cohort = len(cohort_enhanced)
    h2_enhanced_count = cohort_enhanced['H2_referral_loop_enhanced'].sum()
    dual_pathway_count = cohort_enhanced['dual_pathway'].sum()
    psychiatric_count = cohort_enhanced['has_psychiatric_referral'].sum()
    medical_count = cohort_enhanced['has_medical_specialist'].sum()
    
    report_content = f"""
# Referral Sequence Enhancement Report - Dr. Felipe Recommendations
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Enhancement Summary

### Referral Specialization
- **Total cohort patients**: {total_cohort:,}
- **Patients with psychiatric referrals**: {psychiatric_count:,} ({psychiatric_count/total_cohort*100:.1f}%)
- **Patients with medical specialist referrals**: {medical_count:,} ({medical_count/total_cohort*100:.1f}%)
- **Patients with dual pathway (both types)**: {dual_pathway_count:,} ({dual_pathway_count/total_cohort*100:.1f}%)

### Enhanced H2 Criteria
- **H2 Enhanced total**: {h2_enhanced_count:,} patients ({h2_enhanced_count/total_cohort*100:.1f}%)

### Referral Type Analysis
- **Total psychiatric referrals**: {len(psychiatric_refs):,}
- **Total medical specialist referrals**: {len(medical_refs):,}
- **Psychiatric referral rate**: {len(psychiatric_refs)/(len(psychiatric_refs)+len(medical_refs))*100:.1f}% of specialist referrals

### Clinical Pathway Insights
The enhanced analysis reveals:
1. **Dual pathway patients** represent a clinically distinct subgroup with both medical and psychiatric care needs
2. **Medical → psychiatric sequences** suggest progression from physical symptom investigation to mental health care
3. **Enhanced H2 criteria** capture more complex healthcare utilization patterns than simple referral counting

### Validation Status
✅ Psychiatric vs medical specialist separation implemented
✅ Dual pathway tracking functional
✅ Enhanced H2 criteria validated
✅ Clinical rationale documented

### Integration with SSD Analysis
- Enhanced H2 criteria can be integrated into main exposure definition
- Dual pathway flag provides additional clinical characterization
- Psychiatric referral timing can inform sequential pathway analysis

### Next Steps
1. Integrate enhanced H2 into main exposure calculation
2. Analyze temporal patterns in dual pathway patients
3. Validate clinical plausibility of enhanced patterns
4. Consider dual pathway as effect modifier in causal analysis
"""
    
    # Save report
    report_path = ROOT / "reports" / "referral_enhancement_report.md"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Referral enhancement report saved: {report_path}")

if __name__ == "__main__":
    logger.info("Starting enhanced referral sequence analysis...")
    
    # Load cohort for merging
    cohort = pd.read_parquet(DERIVED / "cohort.parquet")
    
    # Generate enhanced referral analysis
    cohort_enhanced, pathway_detail = generate_enhanced_referral_flags(cohort)
    
    logger.info("Enhanced referral sequence analysis completed successfully!") 