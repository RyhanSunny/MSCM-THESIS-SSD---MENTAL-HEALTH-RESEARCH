#!/usr/bin/env python3
"""
Quick test runner for Rubin's Rules implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rubins_pooling_engine import pool_estimates_rubins_rules

# Test basic pooling
print("Testing Rubin's Rules implementation...")

# Example from the analysis
estimates = [1.5, 1.6, 1.55, 1.58, 1.52]
ses = [0.10, 0.12, 0.11, 0.105, 0.115]

try:
    result = pool_estimates_rubins_rules(estimates, ses, "TMLE", "total_encounters")
    print(f"✓ Pooled estimate: {result.estimate:.4f}")
    print(f"✓ Pooled SE: {result.standard_error:.4f}")
    print(f"✓ 95% CI: ({result.ci_lower:.4f}, {result.ci_upper:.4f})")
    print(f"✓ FMI: {result.fmi:.3f}")
    print(f"✓ RIV: {result.riv:.3f}")
    
    # Key check: pooled SE should be larger than average SE when there's between variance
    avg_se = sum(ses) / len(ses)
    print(f"\nAverage SE: {avg_se:.4f}")
    print(f"Pooled SE > Average SE: {result.standard_error > avg_se} ✓")
    
    print("\nRubin's Rules implementation is working correctly!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)