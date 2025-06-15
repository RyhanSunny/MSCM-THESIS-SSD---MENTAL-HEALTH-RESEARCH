#!/usr/bin/env python3
"""
Enhanced Exposure Validation Analysis - Comparing AND vs OR Logic
Author: Ryhan Suny
Date: 2025-05-25

This script analyzes both AND and OR logic exposure definitions to provide
a comprehensive comparison for decision-making.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles
from pathlib import Path
import json
import logging
from datetime import datetime
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path('analysis/exposure_validation_enhanced')
output_dir.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load exposure and cohort data"""
    logger.info("Loading data...")
    exposure = pd.read_parquet('data_derived/exposure.parquet')
    cohort = pd.read_parquet('data_derived/cohort.parquet')
    
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
    
    # Merge to get demographics
    data = exposure.merge(cohort[['Patient_ID', 'Age_at_2018', 'Sex', 'Sex_clean']], on='Patient_ID')
    
    logger.info(f"Loaded {len(data):,} patients")
    return data

def create_exposure_comparison_chart(data, output_dir):
    """Create side-by-side comparison of AND vs OR logic"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # OR Logic
    or_exposed = data['exposure_flag'].sum()
    or_unexposed = len(data) - or_exposed
    
    # AND Logic
    and_exposed = data['exposure_flag_strict'].sum()
    and_unexposed = len(data) - and_exposed
    
    # Plot OR logic
    ax1.pie([or_exposed, or_unexposed], 
            labels=[f'Exposed\n{or_exposed:,}\n({or_exposed/len(data)*100:.1f}%)',
                   f'Unexposed\n{or_unexposed:,}\n({or_unexposed/len(data)*100:.1f}%)'],
            colors=['#FF6B6B', '#4ECDC4'],
            autopct='',
            startangle=90)
    ax1.set_title('OR Logic (Any Criterion)', fontsize=16, fontweight='bold')
    
    # Plot AND logic
    ax2.pie([and_exposed, and_unexposed], 
            labels=[f'Exposed\n{and_exposed:,}\n({and_exposed/len(data)*100:.1f}%)',
                   f'Unexposed\n{and_unexposed:,}\n({and_unexposed/len(data)*100:.1f}%)'],
            colors=['#FF6B6B', '#4ECDC4'],
            autopct='',
            startangle=90)
    ax2.set_title('AND Logic (All Criteria)', fontsize=16, fontweight='bold')
    
    # Add comparison text
    fig.suptitle('Exposure Definition Comparison: OR vs AND Logic', fontsize=18, fontweight='bold')
    
    # Add key statistics
    ratio = or_exposed / and_exposed if and_exposed > 0 else float('inf')
    fig.text(0.5, 0.02, f'Difference Factor: {ratio:.0f}x more patients with OR logic', 
             ha='center', fontsize=14, style='italic', color='red')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'exposure_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created exposure comparison chart")
    
    return {
        'or_exposed': or_exposed,
        'or_unexposed': or_unexposed,
        'and_exposed': and_exposed,
        'and_unexposed': and_unexposed,
        'ratio': ratio
    }

def create_criteria_venn_diagram(data, output_dir):
    """Create Venn diagram showing criteria overlap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate overlaps
    h1 = set(data[data['H1_normal_labs']]['Patient_ID'])
    h2 = set(data[data['H2_referral_loop']]['Patient_ID'])
    h3 = set(data[data['H3_drug_persistence']]['Patient_ID'])
    
    # Create Venn diagram
    venn = venn3([h1, h2, h3], 
                 ('H1: Normal Labs', 'H2: Referral Loops', 'H3: Drug Persistence'),
                 ax=ax)
    
    # Customize colors
    venn.get_patch_by_id('100').set_color('#FF6B6B')
    venn.get_patch_by_id('010').set_color('#4ECDC4')
    venn.get_patch_by_id('001').set_color('#95E1D3')
    venn.get_patch_by_id('110').set_color('#FFA502')
    venn.get_patch_by_id('101').set_color('#FF7979')
    venn.get_patch_by_id('011').set_color('#7BED9F')
    venn.get_patch_by_id('111').set_color('#2F3542')
    
    # Add circles
    venn3_circles([h1, h2, h3], linewidth=2, color='grey')
    
    # Title
    ax.set_title('Exposure Criteria Overlap Analysis', fontsize=16, fontweight='bold', pad=20)
    
    # Add annotations
    total_or = len(h1 | h2 | h3)
    total_and = len(h1 & h2 & h3)
    ax.text(0.5, -0.15, f'OR Logic (Any): {total_or:,} patients\nAND Logic (All): {total_and:,} patients',
            transform=ax.transAxes, ha='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'criteria_venn_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created criteria Venn diagram")
    
    # Return detailed counts
    return {
        'h1_only': len(h1 - h2 - h3),
        'h2_only': len(h2 - h1 - h3),
        'h3_only': len(h3 - h1 - h2),
        'h1_h2_only': len((h1 & h2) - h3),
        'h1_h3_only': len((h1 & h3) - h2),
        'h2_h3_only': len((h2 & h3) - h1),
        'all_three': len(h1 & h2 & h3),
        'any_criterion': len(h1 | h2 | h3)
    }

def create_demographic_comparison(data, output_dir):
    """Compare demographics between OR and AND logic exposed groups"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # OR Logic exposed
    or_exposed = data[data['exposure_flag']]
    or_unexposed = data[~data['exposure_flag']]
    
    # AND Logic exposed
    and_exposed = data[data['exposure_flag_strict']]
    and_unexposed = data[~data['exposure_flag_strict']]
    
    # 1. Age distribution - OR Logic
    ax1.hist([or_unexposed['Age_at_2018'], or_exposed['Age_at_2018']], 
             bins=20, label=['Unexposed', 'Exposed'], color=['#4ECDC4', '#FF6B6B'], alpha=0.7)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Count')
    ax1.set_title('Age Distribution - OR Logic')
    ax1.legend()
    
    # 2. Age distribution - AND Logic
    ax2.hist([and_unexposed['Age_at_2018'], and_exposed['Age_at_2018']], 
             bins=20, label=['Unexposed', 'Exposed'], color=['#4ECDC4', '#FF6B6B'], alpha=0.7)
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Count')
    ax2.set_title('Age Distribution - AND Logic')
    ax2.legend()
    
    # 3. Sex distribution - OR Logic (show percentages)
    or_sex_exposed = or_exposed['Sex_clean'].value_counts(normalize=True) * 100
    or_sex_unexposed = or_unexposed['Sex_clean'].value_counts(normalize=True) * 100
    
    sex_data_or = pd.DataFrame({
        'Exposed': or_sex_exposed,
        'Unexposed': or_sex_unexposed
    }).fillna(0)
    
    if 'Male' in sex_data_or.index and 'Female' in sex_data_or.index:
        sex_data_or = sex_data_or.loc[['Male', 'Female']]
        sex_data_or.T.plot(kind='bar', ax=ax3, color=['#3498db', '#e74c3c'])
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Sex Distribution - OR Logic')
        ax3.tick_params(axis='x', rotation=0)
        ax3.legend(['Male', 'Female'])
        
        # Add percentage labels
        for i, (idx, row) in enumerate(sex_data_or.T.iterrows()):
            ax3.text(i-0.15, row['Female'] + 1, f"{row['Female']:.1f}%", ha='center', fontsize=10)
    
    # 4. Sex distribution - AND Logic
    if len(and_exposed) > 0:
        and_sex_exposed = and_exposed['Sex_clean'].value_counts(normalize=True) * 100
        and_sex_unexposed = and_unexposed['Sex_clean'].value_counts(normalize=True) * 100
        
        sex_data_and = pd.DataFrame({
            'Exposed': and_sex_exposed,
            'Unexposed': and_sex_unexposed
        }).fillna(0)
        
        if 'Male' in sex_data_and.index and 'Female' in sex_data_and.index:
            sex_data_and = sex_data_and.loc[['Male', 'Female']]
            sex_data_and.T.plot(kind='bar', ax=ax4, color=['#3498db', '#e74c3c'])
            ax4.set_ylabel('Percentage (%)')
            ax4.tick_params(axis='x', rotation=0)
            ax4.legend(['Male', 'Female'])
            
            for i, (idx, row) in enumerate(sex_data_and.T.iterrows()):
                if 'Female' in row:
                    ax4.text(i-0.15, row['Female'] + 1, f"{row['Female']:.1f}%", ha='center', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for AND logic', 
                 ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_title('Sex Distribution - AND Logic')
    
    fig.suptitle('Demographic Comparison: OR vs AND Logic', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'demographic_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created demographic comparison")
    
    return {
        'or_mean_age_exposed': or_exposed['Age_at_2018'].mean(),
        'or_mean_age_unexposed': or_unexposed['Age_at_2018'].mean(),
        'and_mean_age_exposed': and_exposed['Age_at_2018'].mean() if len(and_exposed) > 0 else 0,
        'and_mean_age_unexposed': and_unexposed['Age_at_2018'].mean(),
        'or_female_pct_exposed': (or_exposed['Sex_clean'] == 'Female').mean() * 100,
        'or_female_pct_unexposed': (or_unexposed['Sex_clean'] == 'Female').mean() * 100,
        'and_female_pct_exposed': (and_exposed['Sex_clean'] == 'Female').mean() * 100 if len(and_exposed) > 0 else 0,
        'and_female_pct_unexposed': (and_unexposed['Sex_clean'] == 'Female').mean() * 100
    }

def create_criteria_intensity_analysis(data, output_dir):
    """Analyze the intensity of each criterion"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Normal lab count distribution
    data_h1 = data[data['H1_normal_labs']]
    ax1.hist(data_h1['normal_lab_count'], bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Normal Lab Count')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title(f'H1: Normal Lab Intensity\n(n={len(data_h1):,})')
    ax1.axvline(data_h1['normal_lab_count'].median(), color='red', linestyle='--', 
                label=f'Median: {data_h1["normal_lab_count"].median():.0f}')
    ax1.legend()
    
    # 2. Referral count distribution
    data_h2 = data[data['H2_referral_loop']]
    ax2.hist(data_h2['symptom_referral_n'], bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Referral Count')
    ax2.set_ylabel('Number of Patients')
    ax2.set_title(f'H2: Referral Loop Intensity\n(n={len(data_h2):,})')
    ax2.axvline(data_h2['symptom_referral_n'].median(), color='teal', linestyle='--',
                label=f'Median: {data_h2["symptom_referral_n"].median():.0f}')
    ax2.legend()
    
    # 3. Drug persistence days
    data_h3 = data[data['H3_drug_persistence']]
    ax3.hist(data_h3['drug_days_in_window'], bins=30, color='#95E1D3', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Days on Medication')
    ax3.set_ylabel('Number of Patients')
    ax3.set_title(f'H3: Drug Persistence Intensity\n(n={len(data_h3):,})')
    ax3.axvline(data_h3['drug_days_in_window'].median(), color='green', linestyle='--',
                label=f'Median: {data_h3["drug_days_in_window"].median():.0f}')
    ax3.legend()
    
    fig.suptitle('Exposure Criteria Intensity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'criteria_intensity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created criteria intensity analysis")

def create_power_analysis_comparison(stats, output_dir):
    """Create power analysis comparison between OR and AND logic"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample sizes
    or_exposed = stats['or_exposed']
    and_exposed = stats['and_exposed']
    
    # Calculate minimum detectable effect sizes for 80% power
    # Using two-sample t-test approximation
    alpha = 0.05
    power = 0.80
    
    # For balanced design (1:1 matching)
    or_n_per_group = min(or_exposed, stats['or_unexposed'])
    and_n_per_group = min(and_exposed, stats['and_unexposed'])
    
    # Cohen's d calculations (approximation)
    or_mde = 2.8 / np.sqrt(or_n_per_group)  # Approximation for 80% power
    and_mde = 2.8 / np.sqrt(and_n_per_group) if and_n_per_group > 0 else np.inf
    
    # Create comparison
    categories = ['OR Logic', 'AND Logic']
    exposed_counts = [or_exposed, and_exposed]
    colors = ['#FF6B6B', '#FFA502']
    
    bars = ax.bar(categories, exposed_counts, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, count, mde in zip(bars, exposed_counts, [or_mde, and_mde]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n(MDE: {mde:.3f})',
                ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Number of Exposed Patients', fontsize=12)
    ax.set_title('Statistical Power Comparison: OR vs AND Logic', fontsize=16, fontweight='bold')
    
    # Add explanation
    ax.text(0.5, 0.95, 'MDE = Minimum Detectable Effect Size (Cohen\'s d) for 80% power',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    # Add recommendation based on power
    if and_mde > 0.2:
        recommendation = "AND logic is severely underpowered (MDE > 0.2)"
        color = 'red'
    else:
        recommendation = "Both approaches have adequate power"
        color = 'green'
    
    ax.text(0.5, 0.02, f'Recommendation: {recommendation}',
            transform=ax.transAxes, ha='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'power_analysis_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Created power analysis comparison")
    
    return {'or_mde': or_mde, 'and_mde': and_mde}

def generate_summary_report(data, stats, overlap_stats, demo_stats, power_stats, output_dir):
    """Generate comprehensive summary statistics"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_patients': len(data),
        
        'or_logic': {
            'exposed': int(stats['or_exposed']),
            'unexposed': int(stats['or_unexposed']),
            'percent_exposed': round(stats['or_exposed'] / len(data) * 100, 2),
            'mean_age_exposed': round(demo_stats['or_mean_age_exposed'], 1),
            'mean_age_unexposed': round(demo_stats['or_mean_age_unexposed'], 1),
            'female_pct_exposed': round(demo_stats['or_female_pct_exposed'], 1),
            'female_pct_unexposed': round(demo_stats['or_female_pct_unexposed'], 1),
            'minimum_detectable_effect': round(power_stats['or_mde'], 3)
        },
        
        'and_logic': {
            'exposed': int(stats['and_exposed']),
            'unexposed': int(stats['and_unexposed']),
            'percent_exposed': round(stats['and_exposed'] / len(data) * 100, 2),
            'mean_age_exposed': round(demo_stats['and_mean_age_exposed'], 1),
            'mean_age_unexposed': round(demo_stats['and_mean_age_unexposed'], 1),
            'female_pct_exposed': round(demo_stats['and_female_pct_exposed'], 1),
            'female_pct_unexposed': round(demo_stats['and_female_pct_unexposed'], 1),
            'minimum_detectable_effect': round(power_stats['and_mde'], 3) if power_stats['and_mde'] != np.inf else 'Infinity'
        },
        
        'discrepancy_factor': round(stats['ratio'], 1),
        
        'criteria_overlap': overlap_stats,
        
        'individual_criteria': {
            'h1_normal_labs': int(data['H1_normal_labs'].sum()),
            'h2_referral_loops': int(data['H2_referral_loop'].sum()),
            'h3_drug_persistence': int(data['H3_drug_persistence'].sum())
        },
        
        'recommendations': {
            'power_consideration': 'AND logic may be underpowered for causal inference' if power_stats['and_mde'] > 0.8 else 'Both approaches have adequate power',
            'clinical_validity': 'AND logic provides more specific phenotype but limited sample',
            'suggested_approach': 'Consider OR logic for primary analysis with AND logic sensitivity analysis' if power_stats['and_mde'] > 0.8 else 'Either approach viable depending on clinical priorities'
        }
    }
    
    # Save JSON summary
    with open(output_dir / 'exposure_validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Generated summary report")
    
    return summary

def generate_latex_report(summary, output_dir):
    """Generate LaTeX report with findings"""
    latex_content = r'''
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}

\definecolor{alertred}{RGB}{220,53,69}
\definecolor{successgreen}{RGB}{40,167,69}

\title{Enhanced Exposure Validation Report\\
\large Comprehensive Analysis of OR vs AND Logic}
\author{Ryhan Suny\\Toronto Metropolitan University}
\date{''' + datetime.now().strftime('%B %d, %Y') + r'''}

\begin{document}
\maketitle

\section{Executive Summary}

This report presents a comprehensive analysis comparing OR logic (any criterion) versus AND logic (all criteria) for SSD exposure definition. The analysis reveals a ''' + f"{summary['discrepancy_factor']:.0f}" + r'''-fold difference in exposed population size with significant implications for study power and validity.

\section{Key Findings}

\subsection{Population Impact}

\begin{table}[H]
\centering
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{OR Logic} & \textbf{AND Logic} & \textbf{Ratio} \\
\midrule
Exposed Patients & ''' + f"{summary['or_logic']['exposed']:,}" + r''' & ''' + f"{summary['and_logic']['exposed']:,}" + r''' & ''' + f"{summary['discrepancy_factor']:.0f}" + r''':1 \\
Percent Exposed & ''' + f"{summary['or_logic']['percent_exposed']:.1f}" + r'''\% & ''' + f"{summary['and_logic']['percent_exposed']:.1f}" + r'''\% & - \\
Minimum Detectable Effect & ''' + f"{summary['or_logic']['minimum_detectable_effect']:.3f}" + r''' & ''' + str(summary['and_logic']['minimum_detectable_effect']) + r''' & - \\
\bottomrule
\end{tabular}
\caption{Comparison of exposure definitions}
\end{table}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{exposure_comparison.png}
\caption{Visual comparison of OR vs AND logic exposure rates}
\end{figure}

\subsection{Criteria Overlap Analysis}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{criteria_venn_diagram.png}
\caption{Venn diagram showing overlap between exposure criteria}
\end{figure}

The overlap analysis reveals:
\begin{itemize}
\item Only ''' + f"{summary['criteria_overlap']['all_three']:,}" + r''' patients meet all three criteria
\item ''' + f"{summary['criteria_overlap']['any_criterion']:,}" + r''' patients meet at least one criterion
\item Substantial heterogeneity exists in criterion combinations
\end{itemize}

\subsection{Demographic Differences}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{demographic_comparison.png}
\caption{Demographic characteristics by exposure definition}
\end{figure}

\subsection{Statistical Power Analysis}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{power_analysis_comparison.png}
\caption{Statistical power comparison between approaches}
\end{figure}

''' + (r'''\textcolor{alertred}{\textbf{Critical Finding}}: The AND logic approach has a minimum detectable effect size of ''' + str(summary['and_logic']['minimum_detectable_effect']) + r''', indicating severe power limitations.}''' if summary['and_logic']['minimum_detectable_effect'] == 'Infinity' or (isinstance(summary['and_logic']['minimum_detectable_effect'], float) and summary['and_logic']['minimum_detectable_effect'] > 0.8) else '') + r'''

\subsection{Criteria Intensity}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{criteria_intensity.png}
\caption{Distribution of criterion intensity measures}
\end{figure}

\section{Recommendations}

Based on the comprehensive analysis:

\begin{enumerate}
\item \textbf{Primary Analysis}: ''' + summary['recommendations']['suggested_approach'] + r'''
\item \textbf{Power Consideration}: ''' + summary['recommendations']['power_consideration'] + r'''
\item \textbf{Clinical Validity}: ''' + summary['recommendations']['clinical_validity'] + r'''
\end{enumerate}

\section{Conclusion}

The choice between OR and AND logic represents a fundamental trade-off between statistical power and clinical specificity. The AND logic identifies a highly specific but extremely small cohort, while OR logic provides adequate power but may include heterogeneous phenotypes.

\end{document}
'''
    
    # Save LaTeX file
    with open(output_dir / 'exposure_validation_report.tex', 'w') as f:
        f.write(latex_content)
    
    logger.info("Generated LaTeX report")

def main():
    """Main execution function"""
    logger.info("Starting enhanced exposure validation analysis")
    
    # Load data
    data = load_data()
    
    # Create visualizations
    stats = create_exposure_comparison_chart(data, output_dir)
    overlap_stats = create_criteria_venn_diagram(data, output_dir)
    demo_stats = create_demographic_comparison(data, output_dir)
    create_criteria_intensity_analysis(data, output_dir)
    power_stats = create_power_analysis_comparison(stats, output_dir)
    
    # Generate reports
    summary = generate_summary_report(data, stats, overlap_stats, demo_stats, power_stats, output_dir)
    generate_latex_report(summary, output_dir)
    
    logger.info("Enhanced exposure validation complete")
    
    # Print summary to console
    print("\n" + "="*60)
    print("ENHANCED EXPOSURE VALIDATION COMPLETE")
    print("="*60)
    print(f"\nOR Logic: {stats['or_exposed']:,} exposed ({stats['or_exposed']/len(data)*100:.1f}%)")
    print(f"AND Logic: {stats['and_exposed']:,} exposed ({stats['and_exposed']/len(data)*100:.1f}%)")
    print(f"Discrepancy Factor: {stats['ratio']:.0f}x")
    print(f"\nMinimum Detectable Effect Sizes (80% power):")
    print(f"  OR Logic: {power_stats['or_mde']:.3f}")
    print(f"  AND Logic: {power_stats['and_mde']:.3f}" if power_stats['and_mde'] != np.inf else "  AND Logic: Infinity (severely underpowered)")
    print(f"\nRecommendation: {summary['recommendations']['suggested_approach']}")
    print("="*60)

if __name__ == "__main__":
    main()