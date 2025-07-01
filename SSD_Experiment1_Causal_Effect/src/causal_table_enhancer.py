#!/usr/bin/env python3
"""
causal_table_enhancer.py - Add causal language to results tables

Enhances existing results tables with explicit causal interpretation
and proper language following best practices in causal inference reporting.

Following CLAUDE.md requirements:
- Evidence-based implementation
- Clear documentation
- Version numbering and timestamps

Author: Ryhan Suny (Toronto Metropolitan University)
Version: 1.0
Date: 2025-07-01
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def enhance_results_tables_with_causal_language(
    tables_dir: Path = Path("tables"),
    output_dir: Path = Path("tables")
) -> Dict[str, str]:
    """
    Enhance existing tables with explicit causal language.
    
    Returns dict of enhanced table paths.
    """
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    enhanced_tables = {}
    
    # Enhance main results table
    main_table_path = tables_dir / "main_results.csv"
    if main_table_path.exists():
        enhanced_main = _enhance_main_results_table(main_table_path, output_dir)
        enhanced_tables["main_results"] = enhanced_main
    
    # Enhance baseline table 
    baseline_path = tables_dir / "baseline_table.csv"
    if baseline_path.exists():
        enhanced_baseline = _enhance_baseline_table(baseline_path, output_dir)
        enhanced_tables["baseline"] = enhanced_baseline
    
    # Create causal interpretation guide
    guide_path = _create_causal_interpretation_guide(output_dir)
    enhanced_tables["interpretation_guide"] = guide_path
    
    # Create footnotes template
    footnotes_path = _create_table_footnotes(output_dir)
    enhanced_tables["footnotes"] = footnotes_path
    
    logger.info(f"Enhanced {len(enhanced_tables)} tables with causal language")
    return enhanced_tables


def _enhance_main_results_table(input_path: Path, output_dir: Path) -> str:
    """Enhance main results table with causal language (≤50 lines)."""
    df = pd.read_csv(input_path)
    
    # Add causal interpretation column
    causal_interpretations = {
        'H1': 'The average causal effect of normal laboratory cascades on healthcare utilization',
        'H2': 'The causal effect of referral loops on mental health crisis (limited by data)',
        'H3': 'The average treatment effect of medication persistence on ED visits'
    }
    
    # Create enhanced dataframe
    enhanced_rows = []
    
    for _, row in df.iterrows():
        enhanced_row = row.to_dict()
        
        # Add causal language to description
        hyp = row.get('Hypothesis', '')
        if hyp in causal_interpretations:
            enhanced_row['Causal Interpretation'] = causal_interpretations[hyp]
        
        # Enhance effect estimate description
        if 'IRR (95% CI)' in row:
            irr_text = row['IRR (95% CI)']
            enhanced_row['Causal Effect Estimate'] = f"ATE: {irr_text}"
            enhanced_row['Interpretation'] = _interpret_effect_size(row.get('IRR', 1.0))
        
        enhanced_rows.append(enhanced_row)
    
    enhanced_df = pd.DataFrame(enhanced_rows)
    
    # Add table caption with causal language
    caption = """Table 2. Causal Effect Estimates from Target Trial Emulation

Effects represent the average treatment effect (ATE) estimated using targeted maximum 
likelihood estimation (TMLE) with stabilized inverse probability of treatment weights. 
All estimates adjust for measured confounders identified through directed acyclic graph 
(DAG) analysis. Confidence intervals incorporate uncertainty from multiple imputation 
(m=30) using Rubin's Rules with Barnard-Rubin small-sample adjustment."""
    
    # Save enhanced table
    output_path = output_dir / "main_results_causal.csv"
    enhanced_df.to_csv(output_path, index=False)
    
    # Save with caption as markdown
    md_path = output_dir / "main_results_causal.md"
    with open(md_path, 'w') as f:
        f.write(caption + "\n\n")
        f.write(enhanced_df.to_markdown(index=False))
    
    return str(output_path)


def _enhance_baseline_table(input_path: Path, output_dir: Path) -> str:
    """Enhance baseline table with balance language (≤50 lines)."""
    df = pd.read_csv(input_path)
    
    # Add standardized mean difference interpretation
    if 'SMD' in df.columns:
        df['Balance Assessment'] = df['SMD'].apply(
            lambda x: 'Balanced' if pd.notna(x) and abs(float(x)) < 0.1 
            else 'Imbalanced' if pd.notna(x) else 'N/A'
        )
    
    # Add causal context to caption
    caption = """Table 1. Baseline Characteristics by Exposure Status After Propensity Score Weighting

Standardized mean differences (SMD) <0.1 indicate adequate balance for causal inference. 
Balance was achieved through inverse probability of treatment weighting (IPTW) with 
weight trimming following Crump et al. (2009). The weighted pseudo-population 
approximates a randomized trial where treatment assignment is independent of measured 
confounders, satisfying the conditional exchangeability assumption."""
    
    # Save enhanced table
    output_path = output_dir / "baseline_causal.csv"
    df.to_csv(output_path, index=False)
    
    # Save with caption as markdown
    md_path = output_dir / "baseline_causal.md"
    with open(md_path, 'w') as f:
        f.write(caption + "\n\n")
        f.write(df.to_markdown(index=False))
    
    return str(output_path)


def _interpret_effect_size(irr: float) -> str:
    """Provide causal interpretation of effect size (≤50 lines)."""
    if pd.isna(irr):
        return "Effect not estimable"
    
    irr = float(irr)
    
    if irr < 0.8:
        magnitude = "substantial protective"
    elif irr < 0.95:
        magnitude = "modest protective"
    elif irr < 1.05:
        magnitude = "null"
    elif irr < 1.2:
        magnitude = "modest harmful"
    else:
        magnitude = "substantial harmful"
    
    pct_change = (irr - 1) * 100
    
    if abs(pct_change) < 5:
        return f"No meaningful causal effect detected (IRR={irr:.2f})"
    elif pct_change > 0:
        return f"{magnitude.capitalize()} causal effect: {pct_change:.0f}% increase in outcome"
    else:
        return f"{magnitude.capitalize()} causal effect: {abs(pct_change):.0f}% decrease in outcome"


def _create_causal_interpretation_guide(output_dir: Path) -> str:
    """Create guide for interpreting causal estimates (≤50 lines)."""
    guide_content = """# Guide to Interpreting Causal Effect Estimates

## Key Terms

**Average Treatment Effect (ATE)**: The expected difference in outcomes if the entire 
population were exposed vs. unexposed to the SSD pattern. This is a causal parameter 
identified under our assumptions.

**Incidence Rate Ratio (IRR)**: For count outcomes, the ratio of expected counts 
under exposure vs. non-exposure. IRR=1.4 means 40% more healthcare encounters caused 
by the exposure.

**Adjusted Odds Ratio (aOR)**: For binary outcomes, the odds ratio after adjusting 
for confounders. Under rare outcome assumption, approximates the risk ratio.

## Causal Assumptions

Our causal interpretations rely on:

1. **Conditional Exchangeability**: Given measured confounders, exposed and unexposed 
   groups are exchangeable (no unmeasured confounding)
2. **Positivity**: Every patient has non-zero probability of both exposure states
3. **Consistency**: Well-defined exposure with no multiple versions
4. **No Interference**: One patient's exposure doesn't affect another's outcome

## Interpreting Confidence Intervals

- 95% CI excludes 1.0 → Statistically significant causal effect
- Width reflects uncertainty from sampling and multiple imputation
- Barnard-Rubin adjustment ensures valid coverage for finite samples

## Clinical Significance

Statistical significance ≠ Clinical importance. Consider:
- Number needed to treat/harm (NNT/NNH)
- Absolute risk differences
- Cost-effectiveness thresholds
- Patient preferences

## Limitations

- E-values indicate robustness to unmeasured confounding
- Negative controls test for residual bias
- Results apply to CPCSSN population meeting eligibility criteria
"""
    
    guide_path = output_dir / "causal_interpretation_guide.md"
    with open(guide_path, 'w') as f:
        f.write(guide_content)
    
    return str(guide_path)


def _create_table_footnotes(output_dir: Path) -> str:
    """Create standardized footnotes for tables (≤50 lines)."""
    footnotes = {
        "main_results": """
Abbreviations: ATE, average treatment effect; CI, confidence interval; IRR, incidence 
rate ratio; aOR, adjusted odds ratio; TMLE, targeted maximum likelihood estimation.

† All effects are causal under stated assumptions (see Methods).
‡ Estimates incorporate uncertainty from 30 multiply imputed datasets.
§ P-values adjusted for multiple comparisons using false discovery rate.
""",
        
        "baseline": """
Abbreviations: SD, standard deviation; SMD, standardized mean difference; 
IPTW, inverse probability of treatment weighting.

* SMD <0.1 indicates adequate balance for unbiased causal effect estimation.
† Weighted statistics create pseudo-population mimicking randomized trial.
""",
        
        "sensitivity": """
Abbreviations: E-value, minimum unmeasured confounding to explain away effect;
RR, risk ratio; HR, hazard ratio.

* E-value >2 suggests robustness to unmeasured confounding.
† Negative controls should show null effects if no residual bias.
‡ Alternative specifications test robustness to modeling choices.
"""
    }
    
    # Save footnotes
    footnotes_path = output_dir / "table_footnotes.json"
    with open(footnotes_path, 'w') as f:
        json.dump(footnotes, f, indent=2)
    
    # Create markdown version
    md_path = output_dir / "table_footnotes.md"
    with open(md_path, 'w') as f:
        f.write("# Standardized Table Footnotes\n\n")
        for table, notes in footnotes.items():
            f.write(f"## {table.replace('_', ' ').title()}\n")
            f.write(notes + "\n")
    
    return str(footnotes_path)


if __name__ == "__main__":
    # Enhance tables with causal language
    enhanced = enhance_results_tables_with_causal_language()
    
    print("✓ Tables enhanced with causal language")
    for table_type, path in enhanced.items():
        print(f"  - {table_type}: {Path(path).name}")