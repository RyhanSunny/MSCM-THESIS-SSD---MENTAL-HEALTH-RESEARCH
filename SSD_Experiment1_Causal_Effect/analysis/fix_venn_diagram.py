#!/usr/bin/env python3
"""
Fix Venn diagram visualization with better layout and error handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Load data
exposure = pd.read_parquet('data_derived/exposure.parquet')

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Calculate overlaps
h1 = set(exposure[exposure['H1_normal_labs']]['Patient_ID'])
h2 = set(exposure[exposure['H2_referral_loop']]['Patient_ID']) 
h3 = set(exposure[exposure['H3_drug_persistence']]['Patient_ID'])

# Print sizes for debugging
print(f"H1 (Normal Labs): {len(h1):,}")
print(f"H2 (Referral Loops): {len(h2):,}")
print(f"H3 (Drug Persistence): {len(h3):,}")
print(f"All three: {len(h1 & h2 & h3):,}")

# Create Venn diagram with explicit subset sizes
venn = venn3(subsets=(
    len(h1 - h2 - h3),  # Only H1
    len(h2 - h1 - h3),  # Only H2
    len((h1 & h2) - h3),  # H1 and H2, not H3
    len(h3 - h1 - h2),  # Only H3
    len((h1 & h3) - h2),  # H1 and H3, not H2
    len((h2 & h3) - h1),  # H2 and H3, not H1
    len(h1 & h2 & h3)  # All three
), set_labels=('H1: Normal Labs\n(n={:,})'.format(len(h1)),
               'H2: Referral Loops\n(n={:,})'.format(len(h2)),
               'H3: Drug Persistence\n(n={:,})'.format(len(h3))),
ax=ax)

# Color all patches safely
patch_ids = ['100', '010', '001', '110', '101', '011', '111']
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFA502', '#FF7979', '#7BED9F', '#2F3542']

for patch_id, color in zip(patch_ids, colors):
    patch = venn.get_patch_by_id(patch_id)
    if patch:
        patch.set_color(color)
        patch.set_alpha(0.8)

# Add circles
venn3_circles([h1, h2, h3], linewidth=2, color='grey')

# Title
ax.set_title('Exposure Criteria Overlap Analysis', fontsize=18, fontweight='bold', pad=20)

# Add detailed statistics box
stats_text = f"""Summary Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━
OR Logic (Any criterion): {len(h1 | h2 | h3):,} patients ({len(h1 | h2 | h3)/len(exposure)*100:.1f}%)
AND Logic (All criteria): {len(h1 & h2 & h3):,} patients ({len(h1 & h2 & h3)/len(exposure)*100:.2f}%)

Individual Criteria:
• H1 only: {len(h1 - h2 - h3):,} ({len(h1 - h2 - h3)/len(exposure)*100:.1f}%)
• H2 only: {len(h2 - h1 - h3):,} ({len(h2 - h1 - h3)/len(exposure)*100:.1f}%)
• H3 only: {len(h3 - h1 - h2):,} ({len(h3 - h1 - h2)/len(exposure)*100:.1f}%)

Two Criteria:
• H1 & H2: {len((h1 & h2) - h3):,} ({len((h1 & h2) - h3)/len(exposure)*100:.1f}%)
• H1 & H3: {len((h1 & h3) - h2):,} ({len((h1 & h3) - h2)/len(exposure)*100:.1f}%)
• H2 & H3: {len((h2 & h3) - h1):,} ({len((h2 & h3) - h1)/len(exposure)*100:.1f}%)"""

# Add text box
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', bbox=props)

# Add interpretation
interp_text = f"Key Finding: Only {len(h1 & h2 & h3):,} patients ({len(h1 & h2 & h3)/len(h1 | h2 | h3)*100:.1f}% of exposed) meet all three criteria"
ax.text(0.5, -0.1, interp_text, transform=ax.transAxes, ha='center', 
        fontsize=12, fontweight='bold', color='darkred',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

plt.tight_layout()
plt.savefig('analysis/exposure_validation_enhanced/criteria_venn_diagram_fixed.png', dpi=300, bbox_inches='tight')
plt.close()

# Also create a bar chart version for clearer comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left: Stacked bar showing criteria combinations
categories = ['H1 only', 'H2 only', 'H3 only', 'H1+H2', 'H1+H3', 'H2+H3', 'All 3']
values = [
    len(h1 - h2 - h3),
    len(h2 - h1 - h3),
    len(h3 - h1 - h2),
    len((h1 & h2) - h3),
    len((h1 & h3) - h2),
    len((h2 & h3) - h1),
    len(h1 & h2 & h3)
]

bars = ax1.bar(categories, values, color=colors)
ax1.set_xlabel('Criteria Combination', fontsize=12)
ax1.set_ylabel('Number of Patients', fontsize=12)
ax1.set_title('Distribution of Criteria Combinations', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:,}\n({val/len(exposure)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10)

# Right: Comparison of individual criteria sizes
ax2.bar(['H1: Normal\nLabs', 'H2: Referral\nLoops', 'H3: Drug\nPersistence'],
        [len(h1), len(h2), len(h3)],
        color=['#FF6B6B', '#4ECDC4', '#95E1D3'])

ax2.set_ylabel('Number of Patients', fontsize=12)
ax2.set_title('Individual Criteria Prevalence', fontsize=14, fontweight='bold')

# Add value labels
for i, (label, size) in enumerate(zip(['H1', 'H2', 'H3'], [len(h1), len(h2), len(h3)])):
    ax2.text(i, size, f'{size:,}\n({size/len(exposure)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10)

fig.suptitle('Exposure Criteria Analysis - Alternative View', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/exposure_validation_enhanced/criteria_analysis_bars.png', dpi=300, bbox_inches='tight')
plt.close()

print("Fixed Venn diagram and created alternative visualizations")