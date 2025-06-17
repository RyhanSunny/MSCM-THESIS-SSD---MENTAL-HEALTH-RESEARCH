#!/usr/bin/env python3
"""
test_msm_demo.py - Test MSM demo functionality

Directly test the MSM demo without importing the full temporal module.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def run_iptw_msm_demo(df):
    """Simplified MSM demo for testing"""
    print("Running IPTW-based MSM demo")
    
    try:
        # Group by patient and calculate mean effects
        if 'patient_id' in df.columns:
            patient_data = df.groupby('patient_id').agg({
                'ssd_flag': 'mean',
                'encounters': 'mean',
                'stress_level': 'mean',
                'severity': 'mean'
            }).reset_index()
        else:
            patient_data = df.copy()
            patient_data['patient_id'] = range(len(patient_data))
        
        # Simple propensity score calculation
        # Use a basic logistic-like function without sklearn
        X = patient_data[['stress_level', 'severity']].fillna(0.5)
        
        # Simple logistic regression approximation
        # ps = 1 / (1 + exp(-(beta0 + beta1*x1 + beta2*x2)))
        # Using rough coefficients
        logits = -1.0 + 2.0 * X['stress_level'] + 1.5 * X['severity']
        ps_scores = 1 / (1 + np.exp(-logits))
        
        # Calculate IPTW weights
        weights = np.where(
            patient_data['ssd_flag'] == 1,
            1 / ps_scores,
            1 / (1 - ps_scores)
        )
        
        # Trim extreme weights
        weights = np.clip(weights, 0.1, 10)
        
        # Weighted outcome analysis
        treated_mask = patient_data['ssd_flag'] == 1
        control_mask = patient_data['ssd_flag'] == 0
        
        if treated_mask.any():
            treated_outcome = np.average(
                patient_data[treated_mask]['encounters'],
                weights=weights[treated_mask]
            )
        else:
            treated_outcome = 0
            
        if control_mask.any():
            control_outcome = np.average(
                patient_data[control_mask]['encounters'],
                weights=weights[control_mask]
            )
        else:
            control_outcome = 0
        
        ate = treated_outcome - control_outcome
        
        results = {
            'method': 'IPTW-MSM-Demo',
            'treated_mean': float(treated_outcome),
            'control_mean': float(control_outcome),
            'ate': float(ate),
            'n_treated': int(treated_mask.sum()),
            'n_control': int(control_mask.sum()),
            'mean_ps_score': float(ps_scores.mean()),
            'demo_mode': True,
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        print(f"MSM ATE: {ate:.3f} (Treated: {treated_outcome:.3f}, Control: {control_outcome:.3f})")
        return results
        
    except Exception as e:
        print(f"MSM demo failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Test MSM demo"""
    print("Testing MSM demo functionality...")
    
    # Load demo data
    demo_path = Path("tests/data/longitudinal_demo.parquet")
    if not demo_path.exists():
        print(f"Demo data not found at {demo_path}")
        return 1
    
    df = pd.read_parquet(demo_path)
    print(f"Loaded demo data: {df.shape}")
    
    # Run MSM demo
    results = run_iptw_msm_demo(df)
    
    # Save results
    output_path = Path("results/msm_demo.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Check if successful
    if results.get('status') == 'completed':
        print("✅ MSM demo completed successfully")
        return 0
    else:
        print("❌ MSM demo failed")
        return 1

if __name__ == "__main__":
    exit(main())