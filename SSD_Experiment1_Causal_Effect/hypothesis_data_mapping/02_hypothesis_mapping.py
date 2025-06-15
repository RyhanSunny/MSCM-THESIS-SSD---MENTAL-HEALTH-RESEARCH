#!/usr/bin/env python3
"""
02_hypothesis_mapping.py - Map data availability to each hypothesis
Author: Ryhan Suny
Date: 2025-05-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Read the blueprint to extract hypotheses
blueprint_path = Path("SSD THESIS final METHODOLOGIES blueprint (1).md")
with open(blueprint_path, 'r', encoding='utf-8') as f:
    blueprint_content = f.read()

# Read SSD hypotheses report
hypotheses_path = Path("SSD_Hypotheses_Report.md")
with open(hypotheses_path, 'r', encoding='utf-8') as f:
    hypotheses_content = f.read()

# Define all hypotheses and their data requirements
hypotheses = {
    "H1": {
        "description": "Patients with SSD patterns (normal labs, referral loops, medication persistence) have higher healthcare utilization",
        "data_required": [
            "Lab results with normal ranges",
            "Referral data with specialty types", 
            "Medication data with duration",
            "Encounter data for utilization metrics",
            "ED visit identification"
        ],
        "data_available": {
            "Lab results": "PARTIAL - CSV file exists but normal range detection needs verification",
            "Referrals": "YES - 1.1M referrals with specialty info",
            "Medications": "YES - 7.7M prescriptions with ATC codes and duration",
            "Encounters": "YES - 11.6M encounters with types",
            "ED visits": "NEEDS CHECK - EncounterType has categories but ED not explicit"
        }
    },
    "H2": {
        "description": "Healthcare costs increase with SSD symptom severity",
        "data_required": [
            "Healthcare cost data",
            "Billing/claims data",
            "Resource utilization data",
            "Severity index (autoencoder)"
        ],
        "data_available": {
            "Cost data": "NO - No cost/billing columns found",
            "Claims": "NO - No claims table found",
            "Utilization": "YES - Can derive from encounters/procedures",
            "Severity index": "YES - Autoencoder model exists"
        }
    },
    "H3": {
        "description": "SSD patients have more frequent provider changes",
        "data_required": [
            "Provider IDs in encounters",
            "Temporal sequence of visits",
            "Provider specialty information"
        ],
        "data_available": {
            "Provider IDs": "YES - Provider_ID in encounters",
            "Visit dates": "YES - EncounterDate available",
            "Provider specialty": "NO - Provider table not found"
        }
    },
    "H4": {
        "description": "Psychological factors mediate the relationship",
        "data_required": [
            "Mental health diagnoses",
            "Anxiety/depression codes",
            "Psychological assessment data"
        ],
        "data_available": {
            "MH diagnoses": "NO - Zero F-codes found in data",
            "Anxiety/depression": "NEEDS CHECK - May use ICD-9 codes",
            "Assessments": "NO - No psychological assessment data"
        }
    },
    "H5": {
        "description": "Health anxiety mediates SSD outcomes",
        "data_required": [
            "Health anxiety indicators",
            "Frequent health concerns",
            "Hypochondriasis codes"
        ],
        "data_available": {
            "Direct measures": "NO - No health anxiety scales",
            "Proxy measures": "PARTIAL - Could use visit frequency",
            "Diagnosis codes": "NEEDS CHECK - ICD codes for hypochondriasis"
        }
    },
    "H6": {
        "description": "Physician diagnostic uncertainty affects outcomes",
        "data_required": [
            "NYD (Not Yet Diagnosed) codes",
            "Diagnostic changes over time",
            "Provider notes/uncertainty markers"
        ],
        "data_available": {
            "NYD codes": "YES - Defined in config",
            "Diagnostic changes": "YES - Can track via encounters",
            "Provider notes": "NO - No text/note fields"
        }
    },
    "H7": {
        "description": "Socioeconomic deprivation moderates effects",
        "data_required": [
            "Postal codes for deprivation index",
            "Income/education data",
            "Pampalon deprivation index"
        ],
        "data_available": {
            "Postal codes": "NO - ResidencePostalCode 0% complete",
            "SES indicators": "PARTIAL - Education/Occupation/Housing exist but completeness unknown",
            "Deprivation index": "NO - Cannot calculate without postal codes"
        }
    },
    "H8": {
        "description": "Multimorbidity patterns differ in SSD",
        "data_required": [
            "Charlson comorbidity data",
            "Chronic condition diagnoses",
            "Disease clustering analysis"
        ],
        "data_available": {
            "Charlson": "YES - Can calculate from diagnoses",
            "Chronic conditions": "YES - 2.6M health conditions",
            "Clustering": "YES - Sufficient diagnosis data"
        }
    },
    "H9": {
        "description": "Temporal patterns of utilization change",
        "data_required": [
            "Longitudinal encounter data",
            "Pre/post exposure periods",
            "Time series of utilization"
        ],
        "data_available": {
            "Longitudinal data": "YES - Encounters from 2010-2022",
            "Time periods": "YES - Can define windows",
            "Time series": "YES - Can construct from encounters"
        }
    }
}

# Create data completeness summary
completeness_summary = []
for h_id, h_data in hypotheses.items():
    total_required = len(h_data['data_required'])
    
    # Count availability
    yes_count = sum(1 for status in h_data['data_available'].values() if status.startswith("YES"))
    partial_count = sum(1 for status in h_data['data_available'].values() if status.startswith("PARTIAL"))
    no_count = sum(1 for status in h_data['data_available'].values() if status.startswith("NO"))
    needs_check = sum(1 for status in h_data['data_available'].values() if status.startswith("NEEDS CHECK"))
    
    completeness_summary.append({
        'Hypothesis': h_id,
        'Description': h_data['description'][:50] + '...',
        'Required': total_required,
        'Available': yes_count,
        'Partial': partial_count,
        'Missing': no_count,
        'Needs_Check': needs_check,
        'Completeness_Score': (yes_count + 0.5*partial_count) / total_required
    })

completeness_df = pd.DataFrame(completeness_summary)

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Stacked bar chart
categories = ['Available', 'Partial', 'Missing', 'Needs_Check']
colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']

bottom = np.zeros(len(completeness_df))
for i, cat in enumerate(categories):
    values = completeness_df[cat].values
    ax1.bar(completeness_df['Hypothesis'], values, bottom=bottom, 
            label=cat, color=colors[i])
    bottom += values

ax1.set_ylabel('Number of Data Requirements')
ax1.set_title('Data Availability by Hypothesis')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# Completeness score heatmap
score_matrix = completeness_df.pivot_table(
    index='Hypothesis', 
    values='Completeness_Score'
).values.reshape(-1, 1)

im = ax2.imshow(score_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax2.set_yticks([0])
ax2.set_yticklabels(['Completeness'])
ax2.set_xticks(range(len(completeness_df)))
ax2.set_xticklabels(completeness_df['Hypothesis'])
ax2.set_title('Data Completeness Score by Hypothesis')

# Add text annotations
for i, score in enumerate(score_matrix.flatten()):
    ax2.text(i, 0, f'{score:.2f}', ha='center', va='center',
            color='white' if score < 0.5 else 'black', fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1)
cbar.set_label('Completeness Score (0=Missing, 1=Complete)')

plt.tight_layout()
plt.savefig('hypothesis_data_mapping/data_completeness_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Save detailed mapping
with open('hypothesis_data_mapping/hypothesis_mapping.json', 'w') as f:
    json.dump(hypotheses, f, indent=2)

# Save completeness summary
completeness_df.to_csv('hypothesis_data_mapping/completeness_summary.csv', index=False)

print("\nData Completeness Summary:")
print(completeness_df.to_string(index=False))
print(f"\nAverage completeness score: {completeness_df['Completeness_Score'].mean():.2f}")
print(f"\nHypotheses with >70% data availability: {(completeness_df['Completeness_Score'] > 0.7).sum()}")
print(f"Hypotheses with <50% data availability: {(completeness_df['Completeness_Score'] < 0.5).sum()}")

# Critical missing data summary
print("\n" + "="*60)
print("CRITICAL MISSING DATA:")
print("="*60)
print("1. POSTAL CODES: ResidencePostalCode field exists but 0% populated")
print("   - Impact: Cannot calculate Pampalon deprivation index (H7)")
print("   - Alternative: Use available SES indicators (education, occupation, housing)")
print("\n2. COST/BILLING DATA: No cost or claims data found")
print("   - Impact: Cannot directly test H2 (costs by severity)")
print("   - Alternative: Use utilization as proxy for costs")
print("\n3. MENTAL HEALTH DIAGNOSES: Zero F-codes (ICD-10 MH) found")
print("   - Impact: H4 (psychological mediation) cannot be tested")
print("   - Alternative: Check for ICD-9 mental health codes (290-319)")
print("\n4. PROVIDER SPECIALTY: Provider table not included")
print("   - Impact: H3 (provider changes) limited")
print("   - Alternative: Use referral specialties as proxy")