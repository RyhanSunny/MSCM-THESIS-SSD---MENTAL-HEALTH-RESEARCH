#!/usr/bin/env python3
"""Master validation runner that executes all validation analyses"""

import subprocess
import sys
from pathlib import Path


def run_validation(script_path: Path, name: str) -> bool:
    """Run a validation script and capture results"""
    print(f"\n{'='*60}")
    print(f"Running {name} validation...")
    print('='*60)

    try:
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\u2713 {name} validation completed successfully")
        else:
            print(f"\u2717 {name} validation failed")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"\u2717 Error running {name}: {e}")
        return False


def main() -> int:
    """Run all validation analyses in sequence"""
    base_dir = Path(__file__).resolve().parent.parent
    analysis_dir = base_dir / 'analysis'

    validations = [
        (analysis_dir / 'charlson_validation/charlson_validation.py', 'Charlson'),
        (analysis_dir / 'exposure_validation/exposure_validation.py', 'Exposure'),
        (analysis_dir / 'autoencoder_validation/autoencoder_validation.py', 'Autoencoder'),
        (analysis_dir / 'utilization_validation/utilization_validation.py', 'Utilization'),
        (analysis_dir / 'run_combined_validation.py', 'Combined Analysis'),
        (analysis_dir / 'combined_validation_summary.py', 'Summary Report'),
    ]

    results = []
    for script_path, name in validations:
        if script_path.exists():
            success = run_validation(script_path, name)
            results.append((name, success))
        else:
            print(f"\u26A0 {name} validation script not found: {script_path}")
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for name, success in results:
        status = "\u2713 PASSED" if success else "\u2717 FAILED"
        print(f"{name:20} {status}")

    if all(success for _, success in results):
        print("\n\u2713 All validations completed successfully")
        print("Reports available in analysis/*/")
        return 0
    else:
        print("\n\u2717 Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
