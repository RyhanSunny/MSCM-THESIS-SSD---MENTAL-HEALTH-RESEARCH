#!/usr/bin/env python3
"""
death_rates_analysis.py - Crude death rate analysis

Calculates death rates by year and treatment group to identify potential
immortal time bias or differential mortality that could affect results.

Hypothesis Support:
- H1: Healthcare utilization - mortality as potential confounder
- H3: Medication use - death as competing event

Output:
- results/death_rates_table.csv: Death rates by year and group
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import load_config
from src.artefact_tracker import ArtefactTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_death_rates(df, treatment_col='ssd_flag'):
    """Calculate death rates by year and treatment group"""
    logger.info("Calculating death rates")
    
    # Ensure date columns are datetime
    date_cols = ['index_date', 'death_date', 'last_encounter_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Calculate follow-up time
    if 'death_date' in df.columns:
        df['end_date'] = df['death_date'].fillna(df['last_encounter_date'])
    else:
        df['end_date'] = df['last_encounter_date']
        df['death_date'] = pd.NaT  # Create empty death_date column
    
    df['followup_days'] = (df['end_date'] - df['index_date']).dt.days
    df['followup_years'] = df['followup_days'] / 365.25
    
    # Death indicator
    df['death_event'] = df['death_date'].notna().astype(int)
    
    # Extract year from index date
    df['index_year'] = df['index_date'].dt.year
    
    # Calculate rates by year and treatment
    death_rates = []
    
    # Overall rates
    for year in sorted(df['index_year'].unique()):
        if pd.isna(year):
            continue
            
        year_mask = df['index_year'] == year
        
        # Overall
        year_df = df[year_mask]
        deaths = year_df['death_event'].sum()
        person_years = year_df['followup_years'].sum()
        
        if person_years > 0:
            rate = deaths / person_years * 1000  # per 1000 person-years
            
            death_rates.append({
                'year': int(year),
                'group': 'Overall',
                'deaths': int(deaths),
                'n_patients': len(year_df),
                'person_years': round(person_years, 1),
                'rate_per_1000': round(rate, 2),
                'rate_95ci_lower': round(rate - 1.96 * np.sqrt(deaths) / person_years * 1000, 2),
                'rate_95ci_upper': round(rate + 1.96 * np.sqrt(deaths) / person_years * 1000, 2)
            })
        
        # By treatment group
        for treatment in [0, 1]:
            group_mask = year_mask & (df[treatment_col] == treatment)
            group_df = df[group_mask]
            
            if len(group_df) == 0:
                continue
                
            deaths = group_df['death_event'].sum()
            person_years = group_df['followup_years'].sum()
            
            if person_years > 0:
                rate = deaths / person_years * 1000
                
                death_rates.append({
                    'year': int(year),
                    'group': 'SSD' if treatment == 1 else 'Control',
                    'deaths': int(deaths),
                    'n_patients': len(group_df),
                    'person_years': round(person_years, 1),
                    'rate_per_1000': round(rate, 2),
                    'rate_95ci_lower': round(rate - 1.96 * np.sqrt(max(deaths, 1)) / person_years * 1000, 2),
                    'rate_95ci_upper': round(rate + 1.96 * np.sqrt(max(deaths, 1)) / person_years * 1000, 2)
                })
    
    # Calculate rate ratios
    rates_df = pd.DataFrame(death_rates)
    
    # Add rate ratios for SSD vs Control
    for year in rates_df['year'].unique():
        ssd_rate = rates_df[(rates_df['year'] == year) & (rates_df['group'] == 'SSD')]['rate_per_1000'].values
        control_rate = rates_df[(rates_df['year'] == year) & (rates_df['group'] == 'Control')]['rate_per_1000'].values
        
        if len(ssd_rate) > 0 and len(control_rate) > 0 and control_rate[0] > 0:
            rate_ratio = ssd_rate[0] / control_rate[0]
            logger.info(f"Year {year} - Rate ratio (SSD/Control): {rate_ratio:.2f}")
    
    return rates_df

def create_death_rate_plot(rates_df, output_path):
    """Create visualization of death rates"""
    logger.info("Creating death rate plot")
    
    # Filter out overall rates for plotting
    plot_df = rates_df[rates_df['group'] != 'Overall'].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Death rates by year and group
    for group in plot_df['group'].unique():
        group_data = plot_df[plot_df['group'] == group]
        
        ax1.errorbar(
            group_data['year'],
            group_data['rate_per_1000'],
            yerr=[
                group_data['rate_per_1000'] - group_data['rate_95ci_lower'],
                group_data['rate_95ci_upper'] - group_data['rate_per_1000']
            ],
            marker='o',
            label=group,
            capsize=5
        )
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Death Rate (per 1,000 person-years)')
    ax1.set_title('Mortality Rates by Year and Treatment Group')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Person-years of follow-up
    pivot_py = plot_df.pivot(index='year', columns='group', values='person_years')
    pivot_py.plot(kind='bar', ax=ax2)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Person-Years of Follow-up')
    ax2.set_title('Follow-up Time by Year and Group')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def check_immortal_time_bias(df, treatment_col='ssd_flag'):
    """Check for potential immortal time bias"""
    logger.info("Checking for immortal time bias")
    
    # Time from index to treatment assignment
    # In this simplified version, we assume treatment is assigned at index
    # In practice, would track actual treatment initiation date
    
    # Check if early deaths differ by group
    early_death_window = 30  # days
    
    df['early_death'] = (
        df['death_date'].notna() & 
        ((df['death_date'] - df['index_date']).dt.days <= early_death_window)
    ).astype(int)
    
    # Compare early death rates
    results = {}
    for treatment in [0, 1]:
        group = 'SSD' if treatment == 1 else 'Control'
        group_df = df[df[treatment_col] == treatment]
        
        early_deaths = group_df['early_death'].sum()
        n_patients = len(group_df)
        rate = early_deaths / n_patients * 100
        
        results[group] = {
            'n_patients': n_patients,
            'early_deaths': early_deaths,
            'early_death_rate': round(rate, 2)
        }
    
    # Chi-square test
    from scipy.stats import chi2_contingency
    contingency_table = [
        [results['Control']['early_deaths'], 
         results['Control']['n_patients'] - results['Control']['early_deaths']],
        [results['SSD']['early_deaths'], 
         results['SSD']['n_patients'] - results['SSD']['early_deaths']]
    ]
    
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    
    results['chi2_test'] = {
        'statistic': round(chi2, 3),
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Calculate death rates by year and treatment group"
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without saving outputs')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Initialize tracker
    tracker = ArtefactTracker()
    tracker.track("script_start", {"script": "death_rates_analysis.py"})
    
    # Load data
    data_path = Path("data_derived/patient_master.parquet")
    if not data_path.exists():
        logger.error(f"Patient master data not found at {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Calculate death rates
    rates_df = calculate_death_rates(df)
    
    # Check for immortal time bias
    immortal_time_results = check_immortal_time_bias(df)
    
    # Save results
    if not args.dry_run:
        # Save death rates table
        output_path = Path("results/death_rates_table.csv")
        output_path.parent.mkdir(exist_ok=True)
        rates_df.to_csv(output_path, index=False)
        logger.info(f"Saved death rates to {output_path}")
        
        # Save immortal time bias results
        import json
        bias_path = Path("results/immortal_time_bias.json")
        with open(bias_path, 'w') as f:
            json.dump(immortal_time_results, f, indent=2)
        
        # Create plot
        figures_dir = Path("figures")
        figures_dir.mkdir(exist_ok=True)
        plot_path = figures_dir / "death_rates_plot.pdf"
        create_death_rate_plot(rates_df, plot_path)
        
        # Track outputs
        tracker.track("output_generated", {
            "file": str(output_path),
            "total_deaths": rates_df[rates_df['group'] == 'Overall']['deaths'].sum()
        })
        
        # Update study documentation
        if Path("scripts/update_study_doc.py").exists():
            import subprocess
            subprocess.run([
                "python", "scripts/update_study_doc.py",
                "--step", "Death rates analysis complete",
                "--kv", f"death_rates_table={output_path}"
            ])
    
    # Print summary
    print("\n=== Death Rates Analysis Summary ===")
    print(f"Total patients analyzed: {len(df):,}")
    print(f"Total deaths: {df['death_event'].sum() if 'death_event' in df.columns else 0:,}")
    
    print("\nDeath rates by year:")
    overall_rates = rates_df[rates_df['group'] == 'Overall']
    for _, row in overall_rates.iterrows():
        print(f"  {row['year']}: {row['rate_per_1000']:.1f} per 1,000 person-years "
              f"({row['deaths']} deaths, {row['person_years']:.0f} person-years)")
    
    print("\nDeath rates by treatment group:")
    for group in ['Control', 'SSD']:
        group_rates = rates_df[rates_df['group'] == group]
        if not group_rates.empty:
            mean_rate = group_rates['rate_per_1000'].mean()
            print(f"  {group}: {mean_rate:.1f} per 1,000 person-years (mean)")
    
    print(f"\nImmortal time bias check (30-day mortality):")
    print(f"  Control: {immortal_time_results['Control']['early_death_rate']:.1f}%")
    print(f"  SSD: {immortal_time_results['SSD']['early_death_rate']:.1f}%")
    print(f"  p-value: {immortal_time_results['chi2_test']['p_value']:.4f}")
    print("=====================================\n")

if __name__ == "__main__":
    main()