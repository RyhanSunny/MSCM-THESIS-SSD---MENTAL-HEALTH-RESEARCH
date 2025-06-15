#!/usr/bin/env python3
"""
Fix exposure comparison visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load data
exposure = pd.read_parquet('data_derived/exposure.parquet')

# Create figure with different layout
fig = plt.figure(figsize=(16, 8))

# Create grid
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.5, 1.5, 1])

# 1. OR Logic pie chart (top left)
ax1 = fig.add_subplot(gs[0, 0])
or_exposed = exposure['exposure_flag'].sum()
or_unexposed = len(exposure) - or_exposed

ax1.pie([or_exposed, or_unexposed], 
        labels=[f'Exposed\n{or_exposed:,}\n({or_exposed/len(exposure)*100:.1f}%)',
               f'Unexposed\n{or_unexposed:,}\n({or_unexposed/len(exposure)*100:.1f}%)'],
        colors=['#FF6B6B', '#4ECDC4'],
        autopct='',
        startangle=90,
        textprops={'fontsize': 11})
ax1.set_title('OR Logic (Any Criterion)', fontsize=14, fontweight='bold')

# 2. AND Logic - use bar chart instead (top right)
ax2 = fig.add_subplot(gs[0, 1])
and_exposed = exposure['exposure_flag_strict'].sum()
and_unexposed = len(exposure) - and_exposed

# Create stacked bar
total = len(exposure)
ax2.bar(0, and_exposed, width=0.5, color='#FF6B6B', label=f'Exposed: {and_exposed:,}')
ax2.bar(0, and_unexposed, width=0.5, bottom=and_exposed, color='#4ECDC4', 
        label=f'Unexposed: {and_unexposed:,}')

ax2.set_ylim(0, total)
ax2.set_xlim(-0.5, 0.5)
ax2.set_xticks([])
ax2.set_ylabel('Number of Patients', fontsize=12)
ax2.set_title('AND Logic (All Criteria)', fontsize=14, fontweight='bold')

# Add percentage annotations
ax2.text(0, and_exposed/2, f'{and_exposed/total*100:.2f}%', 
         ha='center', va='center', fontweight='bold', fontsize=12)
ax2.text(0, and_exposed + and_unexposed/2, f'{and_unexposed/total*100:.1f}%', 
         ha='center', va='center', fontweight='bold', fontsize=12)

ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

# 3. Direct comparison bar chart (bottom left, spanning two columns)
ax3 = fig.add_subplot(gs[1, :2])
comparison_data = pd.DataFrame({
    'OR Logic': [or_exposed, or_unexposed],
    'AND Logic': [and_exposed, and_unexposed]
}, index=['Exposed', 'Unexposed'])

x = np.arange(2)
width = 0.35

bars1 = ax3.bar(x - width/2, comparison_data['OR Logic'], width, 
                label='OR Logic', color=['#FF6B6B', '#4ECDC4'])
bars2 = ax3.bar(x + width/2, comparison_data['AND Logic'], width, 
                label='AND Logic', color=['#FF7979', '#9EDCDC'])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10)

ax3.set_xlabel('Exposure Status', fontsize=12)
ax3.set_ylabel('Number of Patients', fontsize=12)
ax3.set_title('Direct Comparison: OR vs AND Logic', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(['Exposed', 'Unexposed'])
ax3.legend()

# 4. Key statistics panel (right)
ax4 = fig.add_subplot(gs[:, 2])
ax4.axis('off')

# Calculate statistics
ratio = or_exposed / and_exposed if and_exposed > 0 else float('inf')
or_rate = or_exposed / total * 100
and_rate = and_exposed / total * 100

# Create text summary
stats_text = f"""KEY STATISTICS

Total Cohort:
{total:,} patients

OR Logic:
• Exposed: {or_exposed:,}
• Rate: {or_rate:.1f}%

AND Logic:
• Exposed: {and_exposed:,}
• Rate: {and_rate:.2f}%

Discrepancy:
{ratio:.0f}x difference

Clinical Impact:
OR = Broader phenotype
AND = Severe cases only

Power Impact:
OR = Adequate power
AND = Severely limited
"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

# Main title
fig.suptitle('Exposure Definition Analysis: OR vs AND Logic', fontsize=18, fontweight='bold')

# Bottom annotation
fig.text(0.5, 0.02, 
         f'Critical Finding: {ratio:.0f}x difference in exposed population size has major implications for study validity',
         ha='center', fontsize=12, style='italic', color='darkred')

plt.tight_layout()
plt.savefig('analysis/exposure_validation_enhanced/exposure_comparison_fixed.png', dpi=300, bbox_inches='tight')
plt.close()

print("Fixed exposure comparison visualization created")