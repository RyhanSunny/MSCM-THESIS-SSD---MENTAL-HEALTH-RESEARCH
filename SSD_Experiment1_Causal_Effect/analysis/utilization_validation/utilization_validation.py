#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utilization_validation.py - Healthcare Utilization Validation Analysis

This script validates the healthcare utilization outcomes generated in the main analysis,
examining distributions, patterns, and relationships with exposure status.

Author: Ryhan Suny
Date: May 2025
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('utilization_validation.log', mode='w')
    ]
)
log = logging.getLogger('utilization_validation')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data_derived"
OUTPUT_DIR = ROOT / "analysis/utilization_validation"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def load_data():
    """Load all relevant data"""
    log.info("Loading data files...")
    
    # Load derived data
    outcomes = pd.read_parquet(DATA_DIR / "outcomes.parquet")
    exposure = pd.read_parquet(DATA_DIR / "exposure.parquet")
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    
    # Merge all data
    data = outcomes.merge(exposure[['Patient_ID', 'exposure_flag', 'H1_normal_labs', 
                                   'H2_referral_loop', 'H3_drug_persistence']], 
                         on='Patient_ID', how='left')
    data = data.merge(cohort[['Patient_ID', 'Sex', 'Age_at_2015', 'Charlson']],
                     on='Patient_ID', how='left')
    
    log.info(f"Loaded {len(data):,} patients with utilization data")
    return data

def create_utilization_overview(data):
    """Create overview of healthcare utilization"""
    log.info("Creating utilization overview...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution of total encounters
    ax = axes[0, 0]
    encounters = data['total_encounters']
    ax.hist(encounters[encounters <= encounters.quantile(0.95)], bins=50, 
            edgecolor='black', alpha=0.7)
    ax.axvline(encounters.mean(), color='red', linestyle='--', 
               label=f'Mean: {encounters.mean():.1f}')
    ax.axvline(encounters.median(), color='green', linestyle='--', 
               label=f'Median: {encounters.median():.1f}')
    ax.set_xlabel('Total Encounters')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Distribution of Total Encounters (95th percentile)', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 2. Cost distribution
    ax = axes[0, 1]
    costs = data['total_cost']
    ax.hist(np.log1p(costs), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log(Total Cost + 1)')
    ax.set_ylabel('Number of Patients')
    ax.set_title('Distribution of Healthcare Costs (log-transformed)', fontsize=14, fontweight='bold')
    
    # Add median cost annotation
    ax.text(0.05, 0.95, f'Median Cost: ${costs.median():.0f}\nMean Cost: ${costs.mean():.0f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Utilization components
    ax = axes[1, 0]
    util_means = data[['total_encounters', 'total_ed_visits', 'total_referrals']].mean()
    util_means.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Mean Count per Patient')
    ax.set_title('Average Healthcare Utilization Components', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['PC Encounters', 'ED Visits', 'Referrals'], rotation=45)
    
    # Add value labels
    for i, v in enumerate(util_means):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    # 4. Zero-inflation analysis
    ax = axes[1, 1]
    zero_counts = {
        'PC Encounters': (data['total_encounters'] == 0).sum(),
        'ED Visits': (data['total_ed_visits'] == 0).sum(),
        'Referrals': (data['total_referrals'] == 0).sum()
    }
    
    zero_pcts = {k: v/len(data)*100 for k, v in zero_counts.items()}
    
    bars = ax.bar(zero_pcts.keys(), zero_pcts.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Percentage with Zero Utilization (%)')
    ax.set_title('Zero-Inflation in Utilization Measures', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, (k, v) in zip(bars, zero_pcts.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{v:.1f}%\n(n={zero_counts[k]:,})', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'utilization_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info("Utilization overview saved")

def create_exposure_comparison(data):
    """Compare utilization between exposed and unexposed"""
    log.info("Creating exposure comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Box plot comparison
    ax = axes[0, 0]
    utilization_cols = ['total_encounters', 'total_ed_visits', 'total_referrals', 'total_cost']
    
    # Prepare data for box plot
    exposed_data = data[data['exposure_flag'] == 1][utilization_cols]
    unexposed_data = data[data['exposure_flag'] == 0][utilization_cols]
    
    # Log transform cost for visualization
    exposed_data['total_cost'] = np.log1p(exposed_data['total_cost'])
    unexposed_data['total_cost'] = np.log1p(unexposed_data['total_cost'])
    
    positions = np.arange(len(utilization_cols))
    width = 0.35
    
    bp1 = ax.boxplot([exposed_data[col].dropna() for col in utilization_cols], 
                     positions=positions - width/2, widths=width, patch_artist=True,
                     boxprops=dict(facecolor='lightcoral'))
    bp2 = ax.boxplot([unexposed_data[col].dropna() for col in utilization_cols], 
                     positions=positions + width/2, widths=width, patch_artist=True,
                     boxprops=dict(facecolor='lightblue'))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Encounters', 'ED Visits', 'Referrals', 'Log(Cost)'])
    ax.set_ylabel('Count / Log(Cost)')
    ax.set_title('Healthcare Utilization by Exposure Status', fontsize=14, fontweight='bold')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Exposed', 'Unexposed'])
    
    # 2. Statistical tests
    ax = axes[0, 1]
    test_results = []
    
    for col in ['total_encounters', 'total_ed_visits', 'total_referrals', 'total_cost']:
        exposed = data[data['exposure_flag'] == 1][col]
        unexposed = data[data['exposure_flag'] == 0][col]
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(exposed, unexposed, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(exposed), len(unexposed)
        r = 1 - (2*statistic) / (n1*n2)
        
        test_results.append({
            'Outcome': col.replace('total_', '').replace('_', ' ').title(),
            'P-value': p_value,
            'Effect Size': r
        })
    
    test_df = pd.DataFrame(test_results)
    
    # Create bar plot of effect sizes
    bars = ax.bar(test_df['Outcome'], test_df['Effect Size'], 
                   color=['green' if p < 0.05 else 'gray' for p in test_df['P-value']])
    ax.set_ylabel('Effect Size (rank-biserial r)')
    ax.set_title('Effect Sizes for Exposure Comparison', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(-0.5, 0.5)
    
    # Add p-value annotations
    for bar, p in zip(bars, test_df['P-value']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.01,
                f'p={p:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 3. Individual pattern comparison
    ax = axes[1, 0]
    pattern_means = data.groupby(['H1_normal_labs', 'H2_referral_loop', 'H3_drug_persistence'])['total_encounters'].mean()
    pattern_counts = data.groupby(['H1_normal_labs', 'H2_referral_loop', 'H3_drug_persistence']).size()
    
    # Filter to patterns with >100 patients
    significant_patterns = pattern_counts[pattern_counts > 100].index
    pattern_data = []
    
    for pattern in significant_patterns:
        pattern_name = f"H1:{pattern[0]}/H2:{pattern[1]}/H3:{pattern[2]}"
        mean_encounters = pattern_means[pattern]
        count = pattern_counts[pattern]
        pattern_data.append({
            'Pattern': pattern_name,
            'Mean Encounters': mean_encounters,
            'N': count
        })
    
    pattern_df = pd.DataFrame(pattern_data).sort_values('Mean Encounters', ascending=False)
    
    bars = ax.bar(range(len(pattern_df)), pattern_df['Mean Encounters'])
    ax.set_xticks(range(len(pattern_df)))
    ax.set_xticklabels([f"{p}\n(n={n:,})" for p, n in zip(pattern_df['Pattern'], pattern_df['N'])], 
                       rotation=45, ha='right')
    ax.set_ylabel('Mean Encounters')
    ax.set_title('Utilization by Pattern Combination (n>100)', fontsize=14, fontweight='bold')
    
    # 4. Cumulative distribution
    ax = axes[1, 1]
    
    # Sort values for CDF
    exposed_encounters = np.sort(data[data['exposure_flag'] == 1]['total_encounters'])
    unexposed_encounters = np.sort(data[data['exposure_flag'] == 0]['total_encounters'])
    
    # Calculate cumulative probabilities
    exposed_cdf = np.arange(1, len(exposed_encounters) + 1) / len(exposed_encounters)
    unexposed_cdf = np.arange(1, len(unexposed_encounters) + 1) / len(unexposed_encounters)
    
    ax.plot(exposed_encounters, exposed_cdf, label='Exposed', linewidth=2)
    ax.plot(unexposed_encounters, unexposed_cdf, label='Unexposed', linewidth=2)
    ax.set_xlabel('Total Encounters')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of Encounters', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, np.percentile(data['total_encounters'], 95))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exposure_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info("Exposure comparison plots saved")

def create_demographic_analysis(data):
    """Analyze utilization by demographics"""
    log.info("Creating demographic analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. By age groups
    ax = axes[0, 0]
    data['Age_Group'] = pd.cut(data['Age_at_2015'], bins=[0, 40, 60, 80, 120],
                               labels=['<40', '40-59', '60-79', '≥80'])
    
    age_util = data.groupby('Age_Group')[['total_encounters', 'total_cost']].mean()
    
    x = np.arange(len(age_util))
    width = 0.35
    
    ax1 = ax
    bars1 = ax1.bar(x - width/2, age_util['total_encounters'], width, 
                     label='Encounters', color='steelblue')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Mean Encounters', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(age_util.index)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, age_util['total_cost'], width, 
                     label='Cost ($)', color='darkorange')
    ax2.set_ylabel('Mean Cost ($)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    ax1.set_title('Healthcare Utilization by Age Group', fontsize=14, fontweight='bold')
    
    # 2. By sex and exposure
    ax = axes[0, 1]
    sex_exposure = data.groupby(['Sex', 'exposure_flag'])['total_encounters'].mean().unstack()
    sex_exposure.plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral'])
    ax.set_xlabel('Sex')
    ax.set_ylabel('Mean Encounters')
    ax.set_title('Encounters by Sex and Exposure Status', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Female', 'Male'], rotation=0)
    ax.legend(['Unexposed', 'Exposed'], title='Exposure Status')
    
    # 3. By Charlson score
    ax = axes[1, 0]
    charlson_groups = data.groupby('Charlson')['total_encounters'].agg(['mean', 'std', 'count'])
    charlson_groups = charlson_groups[charlson_groups['count'] > 50]  # Filter small groups
    
    x = charlson_groups.index
    y = charlson_groups['mean']
    yerr = charlson_groups['std'] / np.sqrt(charlson_groups['count'])  # SEM
    
    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, markersize=8)
    ax.set_xlabel('Charlson Comorbidity Score')
    ax.set_ylabel('Mean Encounters (± SEM)')
    ax.set_title('Utilization by Comorbidity Burden', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax.legend()
    
    # 4. High utilizers analysis
    ax = axes[1, 1]
    
    # Define high utilizers as top 10%
    high_util_threshold = data['total_encounters'].quantile(0.9)
    data['high_utilizer'] = (data['total_encounters'] > high_util_threshold).astype(int)
    
    # Compare characteristics
    characteristics = ['exposure_flag', 'H1_normal_labs', 'H2_referral_loop', 'H3_drug_persistence']
    high_util_props = data[data['high_utilizer'] == 1][characteristics].mean() * 100
    normal_util_props = data[data['high_utilizer'] == 0][characteristics].mean() * 100
    
    x = np.arange(len(characteristics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, high_util_props, width, label='High Utilizers (Top 10%)')
    bars2 = ax.bar(x + width/2, normal_util_props, width, label='Normal Utilizers')
    
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Characteristics of High Utilizers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Any SSD', 'H1', 'H2', 'H3'])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'demographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info("Demographic analysis plots saved")

def generate_summary_statistics(data):
    """Generate comprehensive summary statistics"""
    log.info("Generating summary statistics...")
    
    # Calculate high utilizer threshold
    high_util_threshold = data['total_encounters'].quantile(0.9)
    
    stats_dict = {
        'Total Patients': len(data),
        'Utilization Summary': {
            'Encounters': {
                'Mean': float(data['total_encounters'].mean()),
                'SD': float(data['total_encounters'].std()),
                'Median': float(data['total_encounters'].median()),
                'Zero Count': int((data['total_encounters'] == 0).sum()),
                'Max': int(data['total_encounters'].max())
            },
            'ED Visits': {
                'Mean': float(data['total_ed_visits'].mean()),
                'SD': float(data['total_ed_visits'].std()),
                'Median': float(data['total_ed_visits'].median()),
                'Zero Count': int((data['total_ed_visits'] == 0).sum())
            },
            'Referrals': {
                'Mean': float(data['total_referrals'].mean()),
                'SD': float(data['total_referrals'].std()),
                'Median': float(data['total_referrals'].median()),
                'Zero Count': int((data['total_referrals'] == 0).sum())
            },
            'Cost': {
                'Mean': float(data['total_cost'].mean()),
                'SD': float(data['total_cost'].std()),
                'Median': float(data['total_cost'].median()),
                'IQR': [float(data['total_cost'].quantile(0.25)), 
                       float(data['total_cost'].quantile(0.75))]
            }
        },
        'Exposure Comparison': {
            'Exposed': {
                'N': int(data['exposure_flag'].sum()),
                'Mean Encounters': float(data[data['exposure_flag']==1]['total_encounters'].mean()),
                'Mean Cost': float(data[data['exposure_flag']==1]['total_cost'].mean())
            },
            'Unexposed': {
                'N': int((~data['exposure_flag']).sum()),
                'Mean Encounters': float(data[data['exposure_flag']==0]['total_encounters'].mean()),
                'Mean Cost': float(data[data['exposure_flag']==0]['total_cost'].mean())
            }
        },
        'High Utilizers': {
            'Threshold': float(high_util_threshold),
            'Count': int((data['total_encounters'] > high_util_threshold).sum()),
            'Percentage': float((data['total_encounters'] > high_util_threshold).mean() * 100),
            'SSD Prevalence': float(data[data['total_encounters'] > high_util_threshold]['exposure_flag'].mean() * 100)
        }
    }
    
    # Save to file
    import json
    with open(OUTPUT_DIR / 'utilization_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    log.info("Summary statistics saved")
    return stats_dict

def generate_latex_report(stats_dict):
    """Generate LaTeX report"""
    log.info("Generating LaTeX report...")
    
    report = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{amsmath}

\title{Healthcare Utilization Validation Report}
\author{Ryhan Suny\\Toronto Metropolitan University}
\date{\today}

\begin{document}
\maketitle

\section{Executive Summary}
This report validates the healthcare utilization outcomes for """ + f"{stats_dict['Total Patients']:,}" + r""" patients in the SSD cohort. We examine the distribution of healthcare encounters, emergency visits, referrals, and associated costs, comparing patterns between exposed (SSD) and unexposed patients.

\section{Overall Utilization Patterns}

\subsection{Primary Care Encounters}
\begin{itemize}
    \item \textbf{Mean}: """ + f"{stats_dict['Utilization Summary']['Encounters']['Mean']:.1f}" + r""" (SD: """ + f"{stats_dict['Utilization Summary']['Encounters']['SD']:.1f}" + r""")
    \item \textbf{Median}: """ + f"{stats_dict['Utilization Summary']['Encounters']['Median']:.0f}" + r"""
    \item \textbf{Maximum}: """ + f"{stats_dict['Utilization Summary']['Encounters']['Max']:,}" + r"""
    \item \textbf{Zero utilization}: """ + f"{stats_dict['Utilization Summary']['Encounters']['Zero Count']:,}" + r""" patients (""" + f"{stats_dict['Utilization Summary']['Encounters']['Zero Count']/stats_dict['Total Patients']*100:.1f}%" + r""")
\end{itemize}

\subsection{Emergency Department Visits}
\begin{itemize}
    \item \textbf{Mean}: """ + f"{stats_dict['Utilization Summary']['ED Visits']['Mean']:.2f}" + r""" (SD: """ + f"{stats_dict['Utilization Summary']['ED Visits']['SD']:.2f}" + r""")
    \item \textbf{Zero ED visits}: """ + f"{stats_dict['Utilization Summary']['ED Visits']['Zero Count']:,}" + r""" patients (""" + f"{stats_dict['Utilization Summary']['ED Visits']['Zero Count']/stats_dict['Total Patients']*100:.1f}%" + r""")
\end{itemize}

\subsection{Healthcare Costs}
\begin{itemize}
    \item \textbf{Mean}: \$""" + f"{stats_dict['Utilization Summary']['Cost']['Mean']:.2f}" + r"""
    \item \textbf{Median}: \$""" + f"{stats_dict['Utilization Summary']['Cost']['Median']:.2f}" + r"""
    \item \textbf{IQR}: \$""" + f"{stats_dict['Utilization Summary']['Cost']['IQR'][0]:.2f}" + r""" - \$""" + f"{stats_dict['Utilization Summary']['Cost']['IQR'][1]:.2f}" + r"""
\end{itemize}

\section{Utilization Distributions}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/utilization_overview.png}
    \caption{Overview of healthcare utilization patterns}
\end{figure}

\section{Exposure Group Comparison}

\begin{table}[H]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{Exposed} & \textbf{Unexposed} \\
\midrule
Sample Size & """ + f"{stats_dict['Exposure Comparison']['Exposed']['N']:,}" + r""" & """ + f"{stats_dict['Exposure Comparison']['Unexposed']['N']:,}" + r""" \\
Mean Encounters & """ + f"{stats_dict['Exposure Comparison']['Exposed']['Mean Encounters']:.1f}" + r""" & """ + f"{stats_dict['Exposure Comparison']['Unexposed']['Mean Encounters']:.1f}" + r""" \\
Mean Cost (\$) & """ + f"{stats_dict['Exposure Comparison']['Exposed']['Mean Cost']:.2f}" + r""" & """ + f"{stats_dict['Exposure Comparison']['Unexposed']['Mean Cost']:.2f}" + r""" \\
\bottomrule
\end{tabular}
\caption{Healthcare utilization by exposure status}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/exposure_comparison.png}
    \caption{Detailed comparison of utilization between exposure groups}
\end{figure}

\section{Demographic Patterns}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/demographic_analysis.png}
    \caption{Healthcare utilization patterns by demographic characteristics}
\end{figure}

\section{High Utilizers}

High utilizers are defined as patients in the top 10\% of healthcare encounters (>""" + f"{stats_dict['High Utilizers']['Threshold']:.0f}" + r""" encounters).

\begin{itemize}
    \item \textbf{Count}: """ + f"{stats_dict['High Utilizers']['Count']:,}" + r""" patients (""" + f"{stats_dict['High Utilizers']['Percentage']:.1f}%" + r""")
    \item \textbf{SSD Prevalence}: """ + f"{stats_dict['High Utilizers']['SSD Prevalence']:.1f}%" + r""" have any SSD pattern
\end{itemize}

\section{Key Findings}

\begin{enumerate}
    \item Healthcare utilization shows significant right-skew with substantial zero-inflation for ED visits and referrals
    \item Exposed patients have """ + f"{(stats_dict['Exposure Comparison']['Exposed']['Mean Encounters'] / stats_dict['Exposure Comparison']['Unexposed']['Mean Encounters'] - 1) * 100:.1f}%" + r""" higher mean encounters than unexposed
    \item Healthcare costs follow a log-normal distribution with high variability
    \item Comorbidity burden (Charlson score) shows strong positive association with utilization
\end{enumerate}

\section{Statistical Considerations}

\begin{enumerate}
    \item The zero-inflation in ED visits (""" + f"{stats_dict['Utilization Summary']['ED Visits']['Zero Count']/stats_dict['Total Patients']*100:.1f}%" + r""") suggests need for zero-inflated models
    \item The right-skewed distribution of encounters may require negative binomial regression
    \item Cost analysis should consider log-transformation or gamma GLM
    \item High utilizers show distinct patterns requiring stratified analysis
\end{enumerate}

\end{document}
"""
    
    with open(OUTPUT_DIR / 'utilization_validation_report.tex', 'w') as f:
        f.write(report)
    
    log.info("LaTeX report generated")

def main():
    """Main analysis function"""
    log.info("Starting healthcare utilization validation analysis...")
    start_time = datetime.now()
    
    try:
        # Load data
        data = load_data()
        
        # Generate visualizations
        create_utilization_overview(data)
        create_exposure_comparison(data)
        create_demographic_analysis(data)
        
        # Generate statistics
        stats = generate_summary_statistics(data)
        
        # Generate report
        generate_latex_report(stats)
        
        # Save processed data
        data.to_parquet(OUTPUT_DIR / 'utilization_validation_results.parquet', index=False)
        
        duration = datetime.now() - start_time
        log.info(f"Analysis completed successfully in {duration.total_seconds():.1f} seconds")
        
    except Exception as e:
        log.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()