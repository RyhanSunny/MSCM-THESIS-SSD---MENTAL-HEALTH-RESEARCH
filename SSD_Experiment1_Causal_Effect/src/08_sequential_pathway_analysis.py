#!/usr/bin/env python3
"""
Optimized Sequential Pathway Analysis for Somatic Symptom Disorder (SSD)

Implements Dr. Felipe's sequential causal chain:
NYD → Labs → Specialist → Anxiety → Psychiatrist → SSD

Memory-optimized version for processing 250K patients with 8M lab records
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OptimizedSSDSequentialAnalyzer:
    """Memory-optimized detection of NYD→SSD diagnostic pathway."""

    def __init__(self):
        # Temporal windows in months
        self.pathway_window = 24  # complete pathway
        self.lab_window = 12      # normal labs after NYD
        self.referral_window = 18 # specialist referrals
        
        # Data containers (will be populated)
        self.patient_data = {}
        self.exposure_patients = set()

    def prepare_data(self, cohort_df, health_condition_df, lab_df, referral_df, exposure_df):
        """Pre-process and index data for efficient lookups."""
        logger.info("Preparing data structures for efficient processing")
        
        # Get cohort patient IDs
        cohort_patients = set(cohort_df['Patient_ID'].values)
        logger.info(f"Processing {len(cohort_patients)} cohort patients")
        
        # Pre-filter all dataframes to cohort patients only
        health_condition_df = health_condition_df[health_condition_df['Patient_ID'].isin(cohort_patients)]
        lab_df = lab_df[lab_df['Patient_ID'].isin(cohort_patients)]
        referral_df = referral_df[referral_df['Patient_ID'].isin(cohort_patients)]
        
        # Store exposure patients for quick lookup
        self.exposure_patients = set(exposure_df['Patient_ID'].values)
        
        # Group data by patient for efficient access
        logger.info("Creating patient data index")
        
        # Process health conditions
        hc_grouped = health_condition_df.groupby('Patient_ID')
        lab_grouped = lab_df.groupby('Patient_ID')
        ref_grouped = referral_df.groupby('Patient_ID')
        
        # Build patient data dictionary
        for pid in cohort_patients:
            self.patient_data[pid] = {
                'conditions': hc_grouped.get_group(pid) if pid in hc_grouped.groups else pd.DataFrame(),
                'labs': lab_grouped.get_group(pid) if pid in lab_grouped.groups else pd.DataFrame(),
                'referrals': ref_grouped.get_group(pid) if pid in ref_grouped.groups else pd.DataFrame(),
                'has_exposure': pid in self.exposure_patients
            }
            
        logger.info(f"Data preparation complete. Indexed {len(self.patient_data)} patients")
        
        # Clear original dataframes to free memory
        del health_condition_df, lab_df, referral_df
        gc.collect()

    def detect_complete_pathway(self, patient_id: int):
        """Evaluate whether a patient completes the NYD→SSD pathway."""
        patient_info = self.patient_data.get(patient_id, {})
        if not patient_info:
            return self._create_result(patient_id, 0, False, None, [])
            
        stage_dates = {}
        
        # Step 1: NYD diagnosis
        conditions = patient_info.get('conditions', pd.DataFrame())
        if conditions.empty:
            return self._create_result(patient_id, 0, False, None, stage_dates)
            
        nyd_mask = conditions['DiagnosisText_calc'].str.contains("NYD", case=False, na=False)
        if not nyd_mask.any():
            return self._create_result(patient_id, 0, False, None, stage_dates)
            
        nyd_dates = pd.to_datetime(conditions[nyd_mask]['DateOfOnset']).sort_values()
        stage_dates["nyd"] = nyd_dates.iloc[0]
        
        # Step 2: Normal labs within window (≥3)
        labs = patient_info.get('labs', pd.DataFrame())
        if not labs.empty and 'Result' in labs.columns:
            end_date = stage_dates["nyd"] + timedelta(days=30 * self.lab_window)
            normal_labs = labs[
                (labs['Date'] >= stage_dates["nyd"]) & 
                (labs['Date'] <= end_date) & 
                (labs['Result'] == 'normal')
            ]
            
            if len(normal_labs) < 3:
                return self._create_result(patient_id, 1, False, None, stage_dates)
            stage_dates["normal_labs"] = normal_labs['Date'].min()
        else:
            return self._create_result(patient_id, 1, False, None, stage_dates)
        
        # Step 3: Medical specialist referrals
        referrals = patient_info.get('referrals', pd.DataFrame())
        if referrals.empty:
            return self._create_result(patient_id, 2, False, None, stage_dates)
            
        # Check for body system referrals
        body_system_cols = ['to_cardio', 'to_gastro', 'to_neuro', 'to_musculo', 
                           'to_respiratory', 'to_endo', 'to_derm', 'to_gyn']
        
        has_medical_ref = False
        if all(col in referrals.columns for col in body_system_cols):
            medical_refs = referrals[referrals[body_system_cols].any(axis=1)]
            if not medical_refs.empty:
                has_medical_ref = True
                stage_dates["specialist"] = pd.to_datetime(medical_refs['EffectiveDate']).min()
        
        if not has_medical_ref:
            return self._create_result(patient_id, 2, False, None, stage_dates)
        
        # Step 4: Anxiety/depression emergence
        anxiety_mask = conditions['DiagnosisText_calc'].str.contains(
            "anxiety|depression", case=False, na=False
        )
        if not anxiety_mask.any():
            return self._create_result(patient_id, 3, False, None, stage_dates)
            
        anxiety_dates = pd.to_datetime(conditions[anxiety_mask]['DateOfOnset'])
        # Must be after specialist referral
        anxiety_after = anxiety_dates[anxiety_dates > stage_dates["specialist"]]
        if anxiety_after.empty:
            return self._create_result(patient_id, 3, False, None, stage_dates)
            
        stage_dates["anxiety"] = anxiety_after.min()
        
        # Step 5: Psychiatrist referral
        if 'to_psychiatrist' in referrals.columns:
            psych_refs = referrals[
                (referrals['to_psychiatrist'] == True) & 
                (pd.to_datetime(referrals['EffectiveDate']) > stage_dates["anxiety"])
            ]
            if psych_refs.empty:
                return self._create_result(patient_id, 4, False, None, stage_dates)
            stage_dates["psychiatrist"] = pd.to_datetime(psych_refs['EffectiveDate']).min()
        else:
            return self._create_result(patient_id, 4, False, None, stage_dates)
        
        # Step 6: SSD diagnosis (exposure flag)
        if not patient_info.get('has_exposure', False):
            return self._create_result(patient_id, 5, False, None, stage_dates)
            
        # Estimate SSD date
        ssd_date = stage_dates["psychiatrist"] + timedelta(days=30)
        stage_dates["ssd"] = ssd_date
        
        nyd_to_ssd_days = (ssd_date - stage_dates["nyd"]).days
        
        return self._create_result(patient_id, 6, True, nyd_to_ssd_days, stage_dates)

    def _create_result(self, patient_id, stage, complete, days, stage_dates):
        """Create standardized result dictionary."""
        return {
            "patient_id": patient_id,
            "pathway_stage": stage,
            "complete_pathway": complete,
            "nyd_to_ssd_days": days,
            "bottleneck_stage": None if complete else stage,
            "stages_completed": list(range(stage + 1)) if stage > 0 else []
            # Removed stage_dates as it causes parquet serialization issues
        }

    def analyze_cohort_batch(self, patient_ids):
        """Analyze a batch of patients."""
        results = []
        for pid in patient_ids:
            results.append(self.detect_complete_pathway(pid))
        return results


def main():
    """Main execution function."""
    ROOT = Path(__file__).resolve().parents[1]
    DERIVED = ROOT / "data_derived"
    CHECKPOINT = ROOT / "Notebooks/data/interim/checkpoint_1_20250318_024427"
    RESULTS_DIR = ROOT / "results"
    
    logger.info("Starting optimized sequential pathway analysis")
    
    # Load cohort
    cohort = pd.read_parquet(DERIVED / "cohort.parquet")
    patient_ids = cohort['Patient_ID'].values
    logger.info(f"Loaded {len(patient_ids)} cohort patients")
    
    # Load other data
    logger.info("Loading health condition data")
    health_condition = pd.read_parquet(CHECKPOINT / "health_condition.parquet")
    health_condition['DateOfOnset'] = pd.to_datetime(health_condition['DateOfOnset'])
    
    logger.info("Loading referral data")
    referral = pd.read_parquet(CHECKPOINT / "referral.parquet")
    referral['EffectiveDate'] = pd.to_datetime(referral['EffectiveDate'])
    
    logger.info("Loading exposure data")
    exposure = pd.read_parquet(DERIVED / "exposure.parquet")
    
    # Load and prepare lab data
    logger.info("Loading lab data (this may take a few minutes)")
    lab_csv_path = CHECKPOINT / "lab.csv"
    
    # Get cohort patient IDs for filtering
    cohort_patients = set(patient_ids)
    
    # Read lab data in chunks
    lab_chunks = []
    chunk_size = 500000  # Larger chunks for faster processing
    
    for i, chunk in enumerate(pd.read_csv(lab_csv_path, chunksize=chunk_size, 
                                          usecols=['Patient_ID', 'PerformedDate', 
                                                  'TestResult_calc', 'is_normal'])):
        if i % 5 == 0:
            logger.info(f"  Processing lab chunk {i+1}")
        
        # Filter to cohort patients
        chunk_filtered = chunk[chunk['Patient_ID'].isin(cohort_patients)]
        if not chunk_filtered.empty:
            # Rename columns
            chunk_filtered = chunk_filtered.rename(columns={
                'PerformedDate': 'Date',
                'TestResult_calc': 'Result'
            })
            # Convert date
            chunk_filtered['Date'] = pd.to_datetime(chunk_filtered['Date'], errors='coerce')
            # Map is_normal to Result
            if 'is_normal' in chunk_filtered.columns:
                # Convert Result column to string type first to avoid dtype warning
                chunk_filtered['Result'] = chunk_filtered['Result'].astype(str)
                chunk_filtered.loc[chunk_filtered['is_normal'] == 1, 'Result'] = 'normal'
            lab_chunks.append(chunk_filtered)
    
    lab = pd.concat(lab_chunks, ignore_index=True)
    logger.info(f"Loaded {len(lab)} lab records for cohort patients")
    
    # Clear chunks to free memory
    del lab_chunks
    gc.collect()
    
    # Create analyzer and prepare data
    analyzer = OptimizedSSDSequentialAnalyzer()
    analyzer.prepare_data(cohort, health_condition, lab, referral, exposure)
    
    # Clear original dataframes
    del health_condition, lab, referral, exposure
    gc.collect()
    
    # Process patients in batches (without multiprocessing for now due to pickle issues)
    logger.info("Starting patient pathway analysis")
    
    # Process in batches for progress tracking
    batch_size = 10000
    all_results = []
    
    for i in range(0, len(patient_ids), batch_size):
        batch = patient_ids[i:i+batch_size]
        batch_results = analyzer.analyze_cohort_batch(batch)
        all_results.extend(batch_results)
        
        progress = len(all_results) / len(patient_ids) * 100
        logger.info(f"Progress: {len(all_results)}/{len(patient_ids)} patients ({progress:.1f}%)")
    
    # Create results dataframe
    result_df = pd.DataFrame(all_results)
    
    # Save parquet output
    output = DERIVED / "sequential_pathways.parquet"
    result_df.to_parquet(output, index=False)
    logger.info(f"Sequential pathway results saved to {output}")
    
    # Create summary statistics
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Count bottlenecks by stage
    bottleneck_counts = result_df['bottleneck_stage'].value_counts().to_dict()
    bottleneck_counts = {int(k) if pd.notna(k) else None: int(v) for k, v in bottleneck_counts.items()}
    
    summary = {
        "analysis_type": "Sequential Pathway Analysis",
        "total_patients": len(result_df),
        "complete_pathways": int(result_df['complete_pathway'].sum()),
        "pathway_completion_rate": float(result_df['complete_pathway'].mean()),
        "stage_bottlenecks": bottleneck_counts,
        "avg_nyd_to_ssd_days": float(result_df['nyd_to_ssd_days'].mean()) if result_df['nyd_to_ssd_days'].notna().any() else None,
        "stage_names": {
            "0": "No NYD diagnosis",
            "1": "NYD but <3 normal labs",
            "2": "Normal labs but no specialist referral",
            "3": "Specialist referral but no anxiety/depression",
            "4": "Anxiety/depression but no psychiatrist referral",
            "5": "Psychiatrist referral but no SSD diagnosis",
            "6": "Complete pathway to SSD"
        }
    }
    
    with open(RESULTS_DIR / "sequential_analysis_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Sequential analysis summary saved to {RESULTS_DIR / 'sequential_analysis_results.json'}")
    logger.info(f"Analysis complete: {summary['complete_pathways']}/{summary['total_patients']} patients completed full pathway ({summary['pathway_completion_rate']:.1%})")


if __name__ == "__main__":
    main()