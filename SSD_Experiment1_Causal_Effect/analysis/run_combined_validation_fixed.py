#!/usr/bin/env python3
"""
Fixed combined validation analysis with improved visualizations
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
output_dir = Path('analysis/combined_validation_results_fixed')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading data...")

# Load all derived data
exposure = pd.read_parquet('data_derived/exposure.parquet')
cohort = pd.read_parquet('data_derived/cohort.parquet')
outcomes = pd.read_parquet('data_derived/outcomes.parquet')
confounders = pd.read_parquet('data_derived/confounders.parquet')
autoencoder = pd.read_parquet('data_derived/mediator_autoencoder.parquet')

# Clean sex variable
cohort['Sex_clean'] = cohort['Sex'].str.upper()
cohort['Sex_clean'] = cohort['Sex_clean'].map({
    'MALE': 'Male',
    'FEMALE': 'Female',
    'M': 'Male',
    'F': 'Female',
    'UNKNOWN': 'Unknown',
    'UNDIFFERENTIATED': 'Unknown'
})
cohort['Sex_clean'] = cohort['Sex_clean'].fillna('Unknown')

# Merge data
data = exposure.merge(cohort[['Patient_ID', 'Age_at_2018', 'Sex_clean']], on='Patient_ID')
data = data.merge(outcomes[['Patient_ID', 'total_encounters', 'ed_visits', 'medical_costs']], on='Patient_ID')
data = data.merge(confounders[['Patient_ID', 'Charlson']], on='Patient_ID')
data = data.merge(autoencoder[['Patient_ID', 'SSD_severity_index']], on='Patient_ID')

print(f"Loaded {len(data):,} patients with complete data")

# 1. Create improved comprehensive comparison dashboard
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

# Exposure comparison
or_exposed = data[data['exposure_flag']]
or_unexposed = data[~data['exposure_flag']]
and_exposed = data[data['exposure_flag_strict']]
and_unexposed = data[~data['exposure_flag_strict']]

# 1.1 Exposure rates with better visualization
ax1 = fig.add_subplot(gs[0, 0])

# Create stacked bars showing both OR and AND
exposure_data = pd.DataFrame({
    'Exposed': [len(or_exposed), len(and_exposed)],
    'Unexposed': [len(or_unexposed), len(and_unexposed)]
}, index=['OR Logic', 'AND Logic'])

exposure_data.plot(kind='bar', stacked=True, ax=ax1, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Exposure Rates by Definition', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Patients')
ax1.tick_params(axis='x', rotation=0)

# Add percentage labels
totals = [(len(or_exposed) + len(or_unexposed)), (len(and_exposed) + len(and_unexposed))]
for i, total in enumerate(totals):
    exposed_pct = [len(or_exposed), len(and_exposed)][i] / total * 100
    ax1.text(i, total + 5000, f'{exposed_pct:.1f}% exposed', ha='center', fontsize=11)

# 1.2 Healthcare utilization comparison with proper scales
ax2 = fig.add_subplot(gs[0, 1])

# Create subplot for each metric due to different scales
util_data = pd.DataFrame({
    'OR Exposed': [or_exposed['total_encounters'].mean(), 
                   (or_exposed['ed_visits'] > 0).mean() * 100,
                   or_exposed['medical_costs'].mean() / 100],  # Scale costs
    'OR Unexposed': [or_unexposed['total_encounters'].mean(),
                     (or_unexposed['ed_visits'] > 0).mean() * 100,
                     or_unexposed['medical_costs'].mean() / 100],
    'AND Exposed': [and_exposed['total_encounters'].mean() if len(and_exposed) > 0 else 0,
                    (and_exposed['ed_visits'] > 0).mean() * 100 if len(and_exposed) > 0 else 0,
                    and_exposed['medical_costs'].mean() / 100 if len(and_exposed) > 0 else 0],
    'AND Unexposed': [and_unexposed['total_encounters'].mean(),
                      (and_unexposed['ed_visits'] > 0).mean() * 100,
                      and_unexposed['medical_costs'].mean() / 100]
}, index=['Encounters', 'ED Visit %', 'Cost ($100s)'])

util_data.T.plot(kind='bar', ax=ax2)
ax2.set_title('Healthcare Utilization Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Value (see legend for units)')
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')

# 1.3 Comorbidity burden
ax3 = fig.add_subplot(gs[1, 0])
cci_bins = [0, 1, 3, 5, 10]
cci_labels = ['0', '1-2', '3-4', '5+']

# Calculate percentages for each group
or_exp_cci = pd.cut(or_exposed['Charlson'], bins=cci_bins, labels=cci_labels, right=False).value_counts(normalize=True).sort_index() * 100
or_unexp_cci = pd.cut(or_unexposed['Charlson'], bins=cci_bins, labels=cci_labels, right=False).value_counts(normalize=True).sort_index() * 100

if len(and_exposed) > 0:
    and_exp_cci = pd.cut(and_exposed['Charlson'], bins=cci_bins, labels=cci_labels, right=False).value_counts(normalize=True).sort_index() * 100
else:
    and_exp_cci = pd.Series([0]*4, index=cci_labels)
and_unexp_cci = pd.cut(and_unexposed['Charlson'], bins=cci_bins, labels=cci_labels, right=False).value_counts(normalize=True).sort_index() * 100

# Create grouped bar chart
cci_data = pd.DataFrame({
    'OR Exposed': or_exp_cci,
    'OR Unexposed': or_unexp_cci,
    'AND Exposed': and_exp_cci,
    'AND Unexposed': and_unexp_cci
})

cci_data.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#FFB6B6', '#4ECDC4', '#9EDCDC'])
ax3.set_xlabel('Charlson Comorbidity Index')
ax3.set_ylabel('Percentage (%)')
ax3.set_title('Comorbidity Distribution', fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=0)

# 1.4 Severity index comparison with box plots
ax4 = fig.add_subplot(gs[1, 1])

# Prepare data for box plot
severity_data = []
severity_labels = []

severity_data.extend([or_exposed['SSD_severity_index'].values, or_unexposed['SSD_severity_index'].values])
severity_labels.extend(['OR Exposed', 'OR Unexposed'])

if len(and_exposed) > 0:
    severity_data.extend([and_exposed['SSD_severity_index'].values, and_unexposed['SSD_severity_index'].values])
    severity_labels.extend(['AND Exposed', 'AND Unexposed'])

bp = ax4.boxplot(severity_data, tick_labels=severity_labels, patch_artist=True)

# Color the boxes
colors = ['#FF6B6B', '#FFB6B6', '#4ECDC4', '#9EDCDC']
for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
    patch.set_facecolor(color)

ax4.set_ylabel('Severity Index')
ax4.set_title('Autoencoder Severity Index Distribution', fontsize=14, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)

# Add mean values
for i, d in enumerate(severity_data):
    ax4.text(i+1, np.mean(d), f'μ={np.mean(d):.2f}', ha='center', va='bottom', fontsize=9)

# 1.5 Demographics summary (bottom panel)
ax5 = fig.add_subplot(gs[2, :])

# Create summary table
summary_data = {
    'Metric': ['N', 'Age (mean)', 'Female (%)', 'Encounters (mean)', 'ED visits (%)', 'Cost (mean $)', 'CCI (mean)', 'Severity (mean)'],
    'OR Exposed': [
        f"{len(or_exposed):,}",
        f"{or_exposed['Age_at_2018'].mean():.1f}",
        f"{(or_exposed['Sex_clean'] == 'Female').mean() * 100:.1f}",
        f"{or_exposed['total_encounters'].mean():.1f}",
        f"{(or_exposed['ed_visits'] > 0).mean() * 100:.1f}",
        f"${or_exposed['medical_costs'].mean():.0f}",
        f"{or_exposed['Charlson'].mean():.2f}",
        f"{or_exposed['SSD_severity_index'].mean():.2f}"
    ],
    'OR Unexposed': [
        f"{len(or_unexposed):,}",
        f"{or_unexposed['Age_at_2018'].mean():.1f}",
        f"{(or_unexposed['Sex_clean'] == 'Female').mean() * 100:.1f}",
        f"{or_unexposed['total_encounters'].mean():.1f}",
        f"{(or_unexposed['ed_visits'] > 0).mean() * 100:.1f}",
        f"${or_unexposed['medical_costs'].mean():.0f}",
        f"{or_unexposed['Charlson'].mean():.2f}",
        f"{or_unexposed['SSD_severity_index'].mean():.2f}"
    ],
    'AND Exposed': [
        f"{len(and_exposed):,}",
        f"{and_exposed['Age_at_2018'].mean():.1f}" if len(and_exposed) > 0 else "N/A",
        f"{(and_exposed['Sex_clean'] == 'Female').mean() * 100:.1f}" if len(and_exposed) > 0 else "N/A",
        f"{and_exposed['total_encounters'].mean():.1f}" if len(and_exposed) > 0 else "N/A",
        f"{(and_exposed['ed_visits'] > 0).mean() * 100:.1f}" if len(and_exposed) > 0 else "N/A",
        f"${and_exposed['medical_costs'].mean():.0f}" if len(and_exposed) > 0 else "N/A",
        f"{and_exposed['Charlson'].mean():.2f}" if len(and_exposed) > 0 else "N/A",
        f"{and_exposed['SSD_severity_index'].mean():.2f}" if len(and_exposed) > 0 else "N/A"
    ],
    'AND Unexposed': [
        f"{len(and_unexposed):,}",
        f"{and_unexposed['Age_at_2018'].mean():.1f}",
        f"{(and_unexposed['Sex_clean'] == 'Female').mean() * 100:.1f}",
        f"{and_unexposed['total_encounters'].mean():.1f}",
        f"{(and_unexposed['ed_visits'] > 0).mean() * 100:.1f}",
        f"${and_unexposed['medical_costs'].mean():.0f}",
        f"{and_unexposed['Charlson'].mean():.2f}",
        f"{and_unexposed['SSD_severity_index'].mean():.2f}"
    ]
}

# Create table - transpose data for proper display
table_data = []
for i in range(len(summary_data['Metric'])):
    row = [summary_data[col][i] for col in summary_data.keys()]
    table_data.append(row)

ax5.axis('tight')
ax5.axis('off')
table = ax5.table(cellText=table_data,
                  colLabels=list(summary_data.keys()),
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Style the header
for i in range(len(summary_data.keys())):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold')

ax5.set_title('Summary Statistics Table', fontsize=14, fontweight='bold', pad=20)

fig.suptitle('SSD Study: OR vs AND Logic Comprehensive Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'combined_analysis_dashboard_fixed.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created fixed combined analysis dashboard")

# 2. Create improved criteria combination analysis
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
bars = ax1.bar(combo_counts.keys(), combo_counts.values(), color='skyblue', edgecolor='navy')
ax1.set_xlabel('Criteria Combination')
ax1.set_ylabel('Number of Patients')
ax1.set_title('Distribution of Criteria Combinations', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Add value labels and percentages
for bar, (k, v) in zip(bars, combo_counts.items()):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{v:,}\n({v/len(data)*100:.1f}%)', ha='center', va='bottom', fontsize=9)

# Highlight the "All 3" bar
all_3_idx = list(combo_counts.keys()).index('All 3')
bars[all_3_idx].set_color('#FF6B6B')

# Pie chart for AND logic breakdown
and_breakdown = {
    'Meet all 3 criteria': combo_counts['All 3'],
    'Meet 2 criteria': combo_counts['H1+H2'] + combo_counts['H1+H3'] + combo_counts['H2+H3'],
    'Meet 1 criterion': combo_counts['H1 only'] + combo_counts['H2 only'] + combo_counts['H3 only'],
    'Meet no criteria': combo_counts['None']
}

colors = ['#2F3542', '#FFA502', '#95E1D3', '#DFE4EA']
wedges, texts, autotexts = ax2.pie(and_breakdown.values(), labels=and_breakdown.keys(), 
                                    autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(data)):,})',
                                    colors=colors, startangle=90)
ax2.set_title('Criteria Fulfillment Summary', fontsize=14, fontweight='bold')

# Enhance text
for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')

fig.suptitle('Exposure Criteria Combination Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'criteria_combination_analysis_fixed.png', dpi=300, bbox_inches='tight')
plt.close()

print("Created fixed criteria combination analysis")

# 3. Create statistical summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'total_patients': len(data),
    
    'or_logic': {
        'exposed': len(or_exposed),
        'unexposed': len(or_unexposed),
        'percent_exposed': round(len(or_exposed) / len(data) * 100, 2),
        'mean_age_exposed': round(or_exposed['Age_at_2018'].mean(), 1),
        'mean_encounters_exposed': round(or_exposed['total_encounters'].mean(), 1),
        'mean_cost_exposed': round(or_exposed['medical_costs'].mean(), 2),
        'mean_cci_exposed': round(or_exposed['Charlson'].mean(), 2),
        'mean_severity_exposed': round(or_exposed['SSD_severity_index'].mean(), 3),
        'female_pct_exposed': round((or_exposed['Sex_clean'] == 'Female').mean() * 100, 1)
    },
    
    'and_logic': {
        'exposed': len(and_exposed),
        'unexposed': len(and_unexposed),
        'percent_exposed': round(len(and_exposed) / len(data) * 100, 2),
        'mean_age_exposed': round(and_exposed['Age_at_2018'].mean(), 1) if len(and_exposed) > 0 else 0,
        'mean_encounters_exposed': round(and_exposed['total_encounters'].mean(), 1) if len(and_exposed) > 0 else 0,
        'mean_cost_exposed': round(and_exposed['medical_costs'].mean(), 2) if len(and_exposed) > 0 else 0,
        'mean_cci_exposed': round(and_exposed['Charlson'].mean(), 2) if len(and_exposed) > 0 else 0,
        'mean_severity_exposed': round(and_exposed['SSD_severity_index'].mean(), 3) if len(and_exposed) > 0 else 0,
        'female_pct_exposed': round((and_exposed['Sex_clean'] == 'Female').mean() * 100, 1) if len(and_exposed) > 0 else 0
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

with open(output_dir / 'combined_validation_summary_fixed.json', 'w') as f:
    json.dump(summary, f, indent=2, cls=NumpyEncoder)

print("\n" + "="*60)
print("FIXED COMBINED VALIDATION ANALYSIS COMPLETE")
print("="*60)
print(f"\nTotal patients: {len(data):,}")
print(f"\nOR Logic:")
print(f"  Exposed: {len(or_exposed):,} ({len(or_exposed)/len(data)*100:.1f}%)")
print(f"  Female: {(or_exposed['Sex_clean'] == 'Female').mean() * 100:.1f}%")
print(f"  Mean encounters: {or_exposed['total_encounters'].mean():.1f}")
print(f"  Mean cost: ${or_exposed['medical_costs'].mean():,.0f}")
print(f"\nAND Logic:")
print(f"  Exposed: {len(and_exposed):,} ({len(and_exposed)/len(data)*100:.1f}%)")
if len(and_exposed) > 0:
    print(f"  Female: {(and_exposed['Sex_clean'] == 'Female').mean() * 100:.1f}%")
    print(f"  Mean encounters: {and_exposed['total_encounters'].mean():.1f}")
    print(f"  Mean cost: ${and_exposed['medical_costs'].mean():,.0f}")
print(f"\nDiscrepancy factor: {len(or_exposed)/len(and_exposed):.0f}x" if len(and_exposed) > 0 else "\nDiscrepancy factor: Infinite")
print("="*60)
print(f"\nResults saved to: {output_dir}")