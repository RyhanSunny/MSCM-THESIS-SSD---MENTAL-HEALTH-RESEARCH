#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exposure_validation.py - SSD Exposure Pattern Validation Analysis

This script performs an independent validation of the SSD exposure patterns (H1, H2, H3)
identified in the main analysis. It analyzes the distribution and overlap of the three
criteria and generates visualizations for research presentation.

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
from matplotlib_venn import venn3
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('exposure_validation.log', mode='w')
    ]
)
log = logging.getLogger('exposure_validation')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data_derived"
OUTPUT_DIR = ROOT / "analysis/exposure_validation"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def load_data():
    """Load exposure and cohort data"""
    log.info("Loading data files...")
    
    exposure = pd.read_parquet(DATA_DIR / "exposure.parquet")
    cohort = pd.read_parquet(DATA_DIR / "cohort.parquet")
    
    # Merge to get demographics
    data = exposure.merge(cohort[['Patient_ID', 'Sex', 'Age_at_2018', 'Charlson']], 
                         on='Patient_ID', how='left')
    
    log.info(f"Loaded {len(data):,} patients with exposure data")
    return data

def create_exposure_distribution_plot(data):
    """Create bar plot of individual and combined exposure patterns"""
    log.info("Creating exposure distribution plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Individual criteria
    criteria_data = {
        'H1: ≥3 Normal Labs': data['H1_normal_labs'].sum(),
        'H2: ≥2 Referral Loops': data['H2_referral_loop'].sum(),
        'H3: ≥90 Drug Days': data['H3_drug_persistence'].sum()
    }
    
    criteria_pct = {k: v/len(data)*100 for k, v in criteria_data.items()}
    
    bars1 = ax1.bar(criteria_pct.keys(), criteria_pct.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Percentage of Patients (%)', fontsize=12)
    ax1.set_title('Individual SSD Criteria Prevalence', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, (k, v) in zip(bars1, criteria_pct.items()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{v:.1f}%\n(n={criteria_data[k]:,})', 
                ha='center', va='bottom', fontsize=10)
    
    # Combined exposure logic
    combined_data = {
        'Any Criterion\n(OR Logic)': data['exposure_flag'].sum(),
        'All Criteria\n(AND Logic)': data['exposure_flag_strict'].sum()
    }
    
    combined_pct = {k: v/len(data)*100 for k, v in combined_data.items()}
    
    bars2 = ax2.bar(combined_pct.keys(), combined_pct.values(), color=['#d62728', '#9467bd'])
    ax2.set_ylabel('Percentage of Patients (%)', fontsize=12)
    ax2.set_title('Combined Exposure Definitions', fontsize=14, fontweight='bold')
    
    # Add value labels
    for bar, (k, v) in zip(bars2, combined_pct.items()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{v:.1f}%\n(n={combined_data[k]:,})', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exposure_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info("Exposure distribution plot saved")

def create_venn_diagram(data):
    """Create Venn diagram showing overlap between criteria"""
    log.info("Creating Venn diagram...")
    
    # Calculate overlaps
    h1_only = ((data['H1_normal_labs']) & 
               (~data['H2_referral_loop']) & 
               (~data['H3_drug_persistence'])).sum()
    
    h2_only = ((~data['H1_normal_labs']) & 
               (data['H2_referral_loop']) & 
               (~data['H3_drug_persistence'])).sum()
    
    h3_only = ((~data['H1_normal_labs']) & 
               (~data['H2_referral_loop']) & 
               (data['H3_drug_persistence'])).sum()
    
    h1_h2 = ((data['H1_normal_labs']) & 
             (data['H2_referral_loop']) & 
             (~data['H3_drug_persistence'])).sum()
    
    h1_h3 = ((data['H1_normal_labs']) & 
             (~data['H2_referral_loop']) & 
             (data['H3_drug_persistence'])).sum()
    
    h2_h3 = ((~data['H1_normal_labs']) & 
             (data['H2_referral_loop']) & 
             (data['H3_drug_persistence'])).sum()
    
    h1_h2_h3 = ((data['H1_normal_labs']) & 
                (data['H2_referral_loop']) & 
                (data['H3_drug_persistence'])).sum()
    
    plt.figure(figsize=(10, 8))
    venn = venn3(subsets=(h1_only, h2_only, h1_h2, h3_only, h1_h3, h2_h3, h1_h2_h3),
                 set_labels=('H1: Normal Labs\n(n={:,})'.format(data['H1_normal_labs'].sum()),
                           'H2: Referral Loops\n(n={:,})'.format(data['H2_referral_loop'].sum()),
                           'H3: Drug Persistence\n(n={:,})'.format(data['H3_drug_persistence'].sum())))
    
    plt.title('Overlap Between SSD Exposure Criteria', fontsize=16, fontweight='bold', pad=20)
    
    # Add percentage annotations
    total_exposed = data['exposure_flag'].sum()
    plt.text(0.5, -0.4, f'Total with any criterion: {total_exposed:,} ({total_exposed/len(data)*100:.1f}%)',
             ha='center', transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'criteria_venn_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info("Venn diagram saved")

def create_demographic_stratification(data):
    """Create demographic stratification analysis"""
    log.info("Creating demographic stratification analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. By Sex
    ax = axes[0, 0]
    sex_data = data.groupby('Sex').agg({
        'H1_normal_labs': 'mean',
        'H2_referral_loop': 'mean',
        'H3_drug_persistence': 'mean',
        'exposure_flag': 'mean'
    }) * 100
    
    sex_data.plot(kind='bar', ax=ax)
    ax.set_title('Exposure Patterns by Sex', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prevalence (%)')
    ax.set_xticklabels(['Female', 'Male'], rotation=0)
    ax.legend(['H1: Normal Labs', 'H2: Referrals', 'H3: Drugs', 'Any Pattern'])
    
    # 2. By Age Groups
    ax = axes[0, 1]
    data['Age_Group'] = pd.cut(data['Age_at_2018'], bins=[0, 40, 60, 80, 120], 
                               labels=['<40', '40-59', '60-79', '≥80'])
    
    age_data = data.groupby('Age_Group').agg({
        'exposure_flag': 'mean'
    }) * 100
    
    age_data.plot(kind='bar', ax=ax, color='darkblue')
    ax.set_title('Any SSD Pattern by Age Group', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prevalence (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    # 3. By Charlson Score
    ax = axes[1, 0]
    data['Charlson_Cat'] = pd.cut(data['Charlson'], bins=[-0.1, 0, 1, 2, 20], 
                                  labels=['0', '1', '2', '≥3'])
    
    charlson_data = data.groupby('Charlson_Cat').agg({
        'exposure_flag': 'mean'
    }) * 100
    
    charlson_data.plot(kind='bar', ax=ax, color='darkgreen')
    ax.set_title('Any SSD Pattern by Charlson Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Prevalence (%)')
    ax.set_xlabel('Charlson Score')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 4. Correlation heatmap
    ax = axes[1, 1]
    corr_data = data[['H1_normal_labs', 'H2_referral_loop', 'H3_drug_persistence', 
                     'normal_lab_count', 'symptom_referral_n', 'drug_days_in_window']].corr()
    
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Between Exposure Variables', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['H1', 'H2', 'H3', 'Lab Count', 'Ref Count', 'Drug Days'], rotation=45)
    ax.set_yticklabels(['H1', 'H2', 'H3', 'Lab Count', 'Ref Count', 'Drug Days'], rotation=0)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'demographic_stratification.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info("Demographic stratification plot saved")

def generate_summary_statistics(data):
    """Generate detailed summary statistics"""
    log.info("Generating summary statistics...")
    
    stats_dict = {
        'Total Patients': len(data),
        'H1 (Normal Labs)': {
            'Count': data['H1_normal_labs'].sum(),
            'Percentage': data['H1_normal_labs'].mean() * 100,
            'Mean Lab Count': data[data['H1_normal_labs']]['normal_lab_count'].mean(),
            'SD Lab Count': data[data['H1_normal_labs']]['normal_lab_count'].std()
        },
        'H2 (Referral Loops)': {
            'Count': data['H2_referral_loop'].sum(),
            'Percentage': data['H2_referral_loop'].mean() * 100,
            'Mean Referral Count': data[data['H2_referral_loop']]['symptom_referral_n'].mean(),
            'SD Referral Count': data[data['H2_referral_loop']]['symptom_referral_n'].std()
        },
        'H3 (Drug Persistence)': {
            'Count': data['H3_drug_persistence'].sum(),
            'Percentage': data['H3_drug_persistence'].mean() * 100,
            'Mean Drug Days': data[data['H3_drug_persistence']]['drug_days_in_window'].mean(),
            'SD Drug Days': data[data['H3_drug_persistence']]['drug_days_in_window'].std()
        },
        'Combined Exposure': {
            'OR Logic Count': data['exposure_flag'].sum(),
            'OR Logic Percentage': data['exposure_flag'].mean() * 100,
            'AND Logic Count': data['exposure_flag_strict'].sum(),
            'AND Logic Percentage': data['exposure_flag_strict'].mean() * 100
        },
        'Overlap Statistics': {
            'H1 and H2': ((data['H1_normal_labs']) & (data['H2_referral_loop'])).sum(),
            'H1 and H3': ((data['H1_normal_labs']) & (data['H3_drug_persistence'])).sum(),
            'H2 and H3': ((data['H2_referral_loop']) & (data['H3_drug_persistence'])).sum(),
            'All Three': data['exposure_flag_strict'].sum()
        }
    }
    
    # Save to file
    import json
    with open(OUTPUT_DIR / 'exposure_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2, default=str)
    
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
\usepackage{hyperref}

\title{SSD Exposure Pattern Validation Report}
\author{Ryhan Suny\\Toronto Metropolitan University}
\date{\today}

\begin{document}
\maketitle

\section{Executive Summary}
This report presents an independent validation of the Somatic Symptom Disorder (SSD) exposure patterns identified in the main analysis. We examined three hypothesized patterns (H1: Normal lab cascades, H2: Referral loops, H3: Medication persistence) and their overlap in a cohort of """ + f"{stats_dict['Total Patients']:,}" + r""" Canadian primary care patients.

\section{Key Findings}

\subsection{Individual Pattern Prevalence}
\begin{itemize}
    \item \textbf{H1 - Normal Lab Cascades}: """ + f"{stats_dict['H1 (Normal Labs)']['Count']:,} ({stats_dict['H1 (Normal Labs)']['Percentage']:.1f}%)" + r""" patients
    \item \textbf{H2 - Referral Loops}: """ + f"{stats_dict['H2 (Referral Loops)']['Count']:,} ({stats_dict['H2 (Referral Loops)']['Percentage']:.1f}%)" + r""" patients
    \item \textbf{H3 - Medication Persistence}: """ + f"{stats_dict['H3 (Drug Persistence)']['Count']:,} ({stats_dict['H3 (Drug Persistence)']['Percentage']:.1f}%)" + r""" patients
\end{itemize}

\subsection{Combined Exposure Definitions}
\begin{itemize}
    \item \textbf{OR Logic (Any Pattern)}: """ + f"{stats_dict['Combined Exposure']['OR Logic Count']:,} ({stats_dict['Combined Exposure']['OR Logic Percentage']:.1f}%)" + r""" patients
    \item \textbf{AND Logic (All Patterns)}: """ + f"{stats_dict['Combined Exposure']['AND Logic Count']:,} ({stats_dict['Combined Exposure']['AND Logic Percentage']:.1f}%)" + r""" patients
\end{itemize}

\section{Pattern Distribution}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/exposure_distribution.png}
    \caption{Distribution of individual and combined SSD exposure patterns}
\end{figure}

\section{Pattern Overlap Analysis}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/criteria_venn_diagram.png}
    \caption{Venn diagram showing overlap between the three SSD criteria}
\end{figure}

The overlap analysis reveals:
\begin{itemize}
    \item """ + f"{stats_dict['Overlap Statistics']['H1 and H2']:,}" + r""" patients meet both H1 and H2 criteria
    \item """ + f"{stats_dict['Overlap Statistics']['H1 and H3']:,}" + r""" patients meet both H1 and H3 criteria
    \item """ + f"{stats_dict['Overlap Statistics']['H2 and H3']:,}" + r""" patients meet both H2 and H3 criteria
    \item """ + f"{stats_dict['Overlap Statistics']['All Three']:,}" + r""" patients meet all three criteria
\end{itemize}

\section{Demographic Stratification}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/demographic_stratification.png}
    \caption{Exposure patterns stratified by demographics and clinical characteristics}
\end{figure}

\section{Clinical Implications}

The stark difference between OR logic (""" + f"{stats_dict['Combined Exposure']['OR Logic Percentage']:.1f}%" + r""") and AND logic (""" + f"{stats_dict['Combined Exposure']['AND Logic Percentage']:.1f}%" + r""") exposure definitions has critical implications for:

\begin{enumerate}
    \item \textbf{Statistical Power}: The AND logic definition captures only """ + f"{stats_dict['Combined Exposure']['AND Logic Count']:,}" + r""" patients, potentially limiting power for causal inference.
    \item \textbf{Clinical Heterogeneity}: The OR logic captures a more heterogeneous population that may dilute treatment effects.
    \item \textbf{Policy Implications}: Different definitions lead to vastly different estimates of disease burden.
\end{enumerate}

\section{Recommendations}

\begin{enumerate}
    \item Consider using the OR logic definition for primary analysis to ensure adequate statistical power
    \item Perform sensitivity analyses using the AND logic definition
    \item Explore individual patterns (H1, H2, H3) as separate exposures in secondary analyses
    \item Consider developing a continuous exposure score based on the number of criteria met
\end{enumerate}

\end{document}
"""
    
    with open(OUTPUT_DIR / 'exposure_validation_report.tex', 'w') as f:
        f.write(report)
    
    log.info("LaTeX report generated")

def main():
    """Main analysis function"""
    log.info("Starting exposure pattern validation analysis...")
    start_time = datetime.now()
    
    try:
        # Load data
        data = load_data()
        
        # Generate visualizations
        create_exposure_distribution_plot(data)
        create_venn_diagram(data)
        create_demographic_stratification(data)
        
        # Generate statistics
        stats = generate_summary_statistics(data)
        
        # Generate report
        generate_latex_report(stats)
        
        # Save processed data
        data.to_parquet(OUTPUT_DIR / 'exposure_validation_results.parquet', index=False)
        
        duration = datetime.now() - start_time
        log.info(f"Analysis completed successfully in {duration.total_seconds():.1f} seconds")
        
    except Exception as e:
        log.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()