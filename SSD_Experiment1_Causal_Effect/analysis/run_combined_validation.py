#!/usr/bin/env python3
"""
Run combined validation analysis using existing data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path('analysis/combined_validation_results')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading data...")

# Load all derived data
exposure = pd.read_parquet('data_derived/exposure.parquet')
cohort = pd.read_parquet('data_derived/cohort.parquet')
outcomes = pd.read_parquet('data_derived/outcomes.parquet')
confounders = pd.read_parquet('data_derived/confounders.parquet')
autoencoder = pd.read_parquet('data_derived/mediator_autoencoder.parquet')

# Merge data
data = exposure.merge(cohort[['Patient_ID', 'Age_at_2015', 'Sex']], on='Patient_ID')
data = data.merge(outcomes[['Patient_ID', 'total_encounters', 'ed_visits', 'medical_costs']], on='Patient_ID')
data = data.merge(confounders[['Patient_ID', 'Charlson']], on='Patient_ID')
data = data.merge(autoencoder[['Patient_ID', 'SSD_severity_index']], on='Patient_ID')

print(f"Loaded {len(data):,} patients with complete data")

# 1. Create comprehensive comparison dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SSD Study: OR vs AND Logic Comprehensive Analysis', fontsize=18, fontweight='bold')

# Exposure comparison
or_exposed = data[data['exposure_flag']]
or_unexposed = data[~data['exposure_flag']]
and_exposed = data[data['exposure_flag_strict']]
and_unexposed = data[~data['exposure_flag_strict']]

# 1.1 Exposure rates
exposure_data = pd.DataFrame({
    'OR Logic': [len(or_exposed), len(or_unexposed)],
    'AND Logic': [len(and_exposed), len(and_unexposed)]
}, index=['Exposed', 'Unexposed'])

exposure_data.T.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Exposure Rates by Definition', fontsize=14)
ax1.set_ylabel('Number of Patients')
ax1.tick_params(axis='x', rotation=0)

# Add percentage labels
for i, (idx, row) in enumerate(exposure_data.T.iterrows()):
    total = row.sum()
    for j, val in enumerate(row):
        ax1.text(i + (j-0.5)*0.2, val + 1000, f'{val/total*100:.1f}%', 
                ha='center', fontsize=10)

# 1.2 Healthcare utilization comparison
util_metrics = ['total_encounters', 'ed_visits', 'medical_costs']
util_labels = ['Mean Encounters', 'ED Visit Rate', 'Mean Cost ($)']

or_util = [
    or_exposed['total_encounters'].mean(),
    (or_exposed['ed_visits'] > 0).mean() * 100,
    or_exposed['medical_costs'].mean()
]
or_unexp_util = [
    or_unexposed['total_encounters'].mean(),
    (or_unexposed['ed_visits'] > 0).mean() * 100,
    or_unexposed['medical_costs'].mean()
]

and_util = [
    and_exposed['total_encounters'].mean() if len(and_exposed) > 0 else 0,
    (and_exposed['ed_visits'] > 0).mean() * 100 if len(and_exposed) > 0 else 0,
    and_exposed['medical_costs'].mean() if len(and_exposed) > 0 else 0
]
and_unexp_util = [
    and_unexposed['total_encounters'].mean(),
    (and_unexposed['ed_visits'] > 0).mean() * 100,
    and_unexposed['medical_costs'].mean()
]

x = np.arange(len(util_labels))
width = 0.2

ax2.bar(x - 1.5*width, or_util, width, label='OR Exposed', color='#FF6B6B')
ax2.bar(x - 0.5*width, or_unexp_util, width, label='OR Unexposed', color='#FFB6B6')
ax2.bar(x + 0.5*width, and_util, width, label='AND Exposed', color='#4ECDC4')
ax2.bar(x + 1.5*width, and_unexp_util, width, label='AND Unexposed', color='#9EDCDC')

ax2.set_xticks(x)
ax2.set_xticklabels(util_labels)
ax2.set_title('Healthcare Utilization Comparison', fontsize=14)
ax2.legend(loc='upper left')

# 1.3 Comorbidity burden
cci_bins = [0, 1, 3, 5, 10]
cci_labels = ['0', '1-2', '3-4', '5+']

or_exp_cci = pd.cut(or_exposed['Charlson'], bins=cci_bins, labels=cci_labels, right=False).value_counts(normalize=True).sort_index()
or_unexp_cci = pd.cut(or_unexposed['Charlson'], bins=cci_bins, labels=cci_labels, right=False).value_counts(normalize=True).sort_index()
and_exp_cci = pd.cut(and_exposed['Charlson'], bins=cci_bins, labels=cci_labels, right=False).value_counts(normalize=True).sort_index() if len(and_exposed) > 0 else pd.Series([0]*4, index=cci_labels)
and_unexp_cci = pd.cut(and_unexposed['Charlson'], bins=cci_bins, labels=cci_labels, right=False).value_counts(normalize=True).sort_index()

x = np.arange(len(cci_labels))
ax3.bar(x - 1.5*width, or_exp_cci.values * 100, width, label='OR Exposed', color='#FF6B6B')
ax3.bar(x - 0.5*width, or_unexp_cci.values * 100, width, label='OR Unexposed', color='#FFB6B6')
ax3.bar(x + 0.5*width, and_exp_cci.values * 100, width, label='AND Exposed', color='#4ECDC4')
ax3.bar(x + 1.5*width, and_unexp_cci.values * 100, width, label='AND Unexposed', color='#9EDCDC')

ax3.set_xticks(x)
ax3.set_xticklabels(cci_labels)
ax3.set_xlabel('Charlson Comorbidity Index')
ax3.set_ylabel('Percentage (%)')
ax3.set_title('Comorbidity Distribution', fontsize=14)
ax3.legend()

# 1.4 Severity index comparison
ax4.violinplot([or_exposed['SSD_severity_index'], or_unexposed['SSD_severity_index']], 
               positions=[0, 1], showmeans=True)
if len(and_exposed) > 0:
    ax4.violinplot([and_exposed['SSD_severity_index'], and_unexposed['SSD_severity_index']], 
                   positions=[2.5, 3.5], showmeans=True)

ax4.set_xticks([0.5, 3])
ax4.set_xticklabels(['OR Logic', 'AND Logic'])
ax4.set_ylabel('Severity Index')
ax4.set_title('Autoencoder Severity Index Distribution', fontsize=14)

# Add sample sizes
ax4.text(0, ax4.get_ylim()[1]*0.95, f'n={len(or_exposed):,}', ha='center', fontsize=9)
ax4.text(1, ax4.get_ylim()[1]*0.95, f'n={len(or_unexposed):,}', ha='center', fontsize=9)
if len(and_exposed) > 0:
    ax4.text(2.5, ax4.get_ylim()[1]*0.95, f'n={len(and_exposed):,}', ha='center', fontsize=9)
ax4.text(3.5, ax4.get_ylim()[1]*0.95, f'n={len(and_unexposed):,}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'combined_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created combined analysis dashboard")

# 2. Create criteria combination analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Count combinations
combo_counts = {
    'H1 only': len(data[(data['H1_normal_labs']) & (~data['H2_referral_loop']) & (~data['H3_drug_persistence'])]),
    'H2 only': len(data[(~data['H1_normal_labs']) & (data['H2_referral_loop']) & (~data['H3_drug_persistence'])]),
    'H3 only': len(data[(~data['H1_normal_labs']) & (~data['H2_referral_loop']) & (data['H3_drug_persistence'])]),
    'H1+H2': len(data[(data['H1_normal_labs']) & (data['H2_referral_loop']) & (~data['H3_drug_persistence'])]),
    'H1+H3': len(data[(data['H1_normal_labs']) & (~data['H2_referral_loop']) & (data['H3_drug_persistence'])]),
    'H2+H3': len(data[(~data['H1_normal_labs']) & (data['H2_referral_loop']) & (data['H3_drug_persistence'])]),
    'All 3': len(data[(data['H1_normal_labs']) & (data['H2_referral_loop']) & (data['H3_drug_persistence'])]),
    'None': len(data[(~data['H1_normal_labs']) & (~data['H2_referral_loop']) & (~data['H3_drug_persistence'])])
}

# Bar chart of combinations
ax1.bar(combo_counts.keys(), combo_counts.values(), color='skyblue', edgecolor='navy')
ax1.set_xlabel('Criteria Combination')
ax1.set_ylabel('Number of Patients')
ax1.set_title('Distribution of Criteria Combinations', fontsize=14)
ax1.tick_params(axis='x', rotation=45)

# Add value labels
for i, (k, v) in enumerate(combo_counts.items()):
    ax1.text(i, v + 500, f'{v:,}', ha='center', fontsize=9)

# Pie chart for AND logic breakdown
and_breakdown = {
    'Meet all 3 criteria': combo_counts['All 3'],
    'Missing 1 criterion': combo_counts['H1+H2'] + combo_counts['H1+H3'] + combo_counts['H2+H3'],
    'Missing 2+ criteria': sum(combo_counts.values()) - combo_counts['All 3'] - 
                          (combo_counts['H1+H2'] + combo_counts['H1+H3'] + combo_counts['H2+H3'])
}

colors = ['#2F3542', '#FFA502', '#95E1D3']
wedges, texts, autotexts = ax2.pie(and_breakdown.values(), labels=and_breakdown.keys(), 
                                    autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('AND Logic Eligibility Breakdown', fontsize=14)

fig.suptitle('Exposure Criteria Combination Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'criteria_combination_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created criteria combination analysis")

# 3. Statistical summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'total_patients': len(data),
    
    'or_logic': {
        'exposed': len(or_exposed),
        'unexposed': len(or_unexposed),
        'percent_exposed': round(len(or_exposed) / len(data) * 100, 2),
        'mean_age_exposed': round(or_exposed['Age_at_2015'].mean(), 1),
        'mean_encounters_exposed': round(or_exposed['total_encounters'].mean(), 1),
        'mean_cost_exposed': round(or_exposed['medical_costs'].mean(), 2),
        'mean_cci_exposed': round(or_exposed['Charlson'].mean(), 2),
        'mean_severity_exposed': round(or_exposed['SSD_severity_index'].mean(), 3)
    },
    
    'and_logic': {
        'exposed': len(and_exposed),
        'unexposed': len(and_unexposed),
        'percent_exposed': round(len(and_exposed) / len(data) * 100, 2),
        'mean_age_exposed': round(and_exposed['Age_at_2015'].mean(), 1) if len(and_exposed) > 0 else 0,
        'mean_encounters_exposed': round(and_exposed['total_encounters'].mean(), 1) if len(and_exposed) > 0 else 0,
        'mean_cost_exposed': round(and_exposed['medical_costs'].mean(), 2) if len(and_exposed) > 0 else 0,
        'mean_cci_exposed': round(and_exposed['Charlson'].mean(), 2) if len(and_exposed) > 0 else 0,
        'mean_severity_exposed': round(and_exposed['SSD_severity_index'].mean(), 3) if len(and_exposed) > 0 else 0
    },
    
    'criteria_combinations': combo_counts,
    
    'discrepancy_factor': round(len(or_exposed) / len(and_exposed), 1) if len(and_exposed) > 0 else float('inf')
}

# Save summary
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

with open(output_dir / 'combined_validation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, cls=NumpyEncoder)

print("\n" + "="*60)
print("COMBINED VALIDATION ANALYSIS COMPLETE")
print("="*60)
print(f"\nTotal patients: {len(data):,}")
print(f"\nOR Logic:")
print(f"  Exposed: {len(or_exposed):,} ({len(or_exposed)/len(data)*100:.1f}%)")
print(f"  Mean encounters: {or_exposed['total_encounters'].mean():.1f}")
print(f"  Mean cost: ${or_exposed['medical_costs'].mean():,.0f}")
print(f"\nAND Logic:")
print(f"  Exposed: {len(and_exposed):,} ({len(and_exposed)/len(data)*100:.1f}%)")
if len(and_exposed) > 0:
    print(f"  Mean encounters: {and_exposed['total_encounters'].mean():.1f}")
    print(f"  Mean cost: ${and_exposed['medical_costs'].mean():,.0f}")
print(f"\nDiscrepancy factor: {len(or_exposed)/len(and_exposed):.0f}x" if len(and_exposed) > 0 else "\nDiscrepancy factor: Infinite")
print("="*60)
print(f"\nResults saved to: {output_dir}")