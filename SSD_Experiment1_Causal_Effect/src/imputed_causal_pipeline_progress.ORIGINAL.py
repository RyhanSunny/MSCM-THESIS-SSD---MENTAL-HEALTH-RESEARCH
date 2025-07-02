#!/usr/bin/env python3
"""
Enhanced imputed causal pipeline with progress tracking
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def run_with_progress():
    import pandas as pd
    import numpy as np
    from src.config_loader import load_config
    
    # Load configuration
    config = load_config()
    
    # Paths
    IMPUTED_DIR = Path("data_derived/imputed_master")
    RESULTS_DIR = Path("results/imputed_causal_results")
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Get all imputed datasets
    imputed_files = sorted(IMPUTED_DIR.glob("master_imputed_*.parquet"))
    n_imputations = len(imputed_files)
    
    print(f"\nFound {n_imputations} imputed datasets")
    print("="*60)
    
    # Progress tracking
    start_time = time.time()
    successful_runs = 0
    failed_runs = 0
    
    for i, imputed_file in enumerate(imputed_files):
        imp_num = int(imputed_file.stem.split('_')[-1])
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing imputation {imp_num}/{n_imputations}")
        print(f"Progress: {i/n_imputations*100:.1f}% complete")
        
        # Estimate time remaining
        if i > 0:
            elapsed = time.time() - start_time
            avg_time_per_imp = elapsed / i
            remaining_time = avg_time_per_imp * (n_imputations - i)
            eta = datetime.now() + timedelta(seconds=remaining_time)
            print(f"Estimated completion: {eta.strftime('%H:%M:%S')} ({remaining_time/60:.1f} minutes remaining)")
        
        try:
            # Run causal estimation
            print(f"  Loading data from {imputed_file.name}...")
            df = pd.read_parquet(imputed_file)
            print(f"  Data shape: {df.shape}")
            
            # Import and run causal estimators with the fixed age handling
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from src.imputed_causal_wrapper import run_causal_estimation_on_imputation
            
            print(f"  Running causal estimation (TMLE, DML, Causal Forest)...")
            results = run_causal_estimation_on_imputation(df, imp_num)
            
            # Save results
            output_file = RESULTS_DIR / f"causal_results_imp{imp_num}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"  ✓ Saved results to {output_file.name}")
            successful_runs += 1
            
            # Show brief summary of results
            if 'estimates' in results:
                print(f"  Results summary:")
                for est in results['estimates']:
                    method = est.get('method', 'Unknown')
                    estimate = est.get('estimate', 'N/A')
                    print(f"    - {method}: {estimate:.4f if isinstance(estimate, (int, float)) else estimate}")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            failed_runs += 1
            
            # Save error information
            error_file = RESULTS_DIR / f"causal_error_imp{imp_num}.txt"
            with open(error_file, 'w') as f:
                f.write(f"Error processing imputation {imp_num}:\n")
                f.write(f"{str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
    
    # Final summary
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"COMPLETED: Causal estimation on {n_imputations} imputations")
    print(f"Time taken: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"Successful: {successful_runs}/{n_imputations}")
    print(f"Failed: {failed_runs}/{n_imputations}")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"{'='*60}")
    
    return successful_runs, failed_runs

if __name__ == "__main__":
    run_with_progress()
