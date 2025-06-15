# Charlson Comorbidity Index Validation Analysis

This directory contains scripts and outputs for an independent validation of the Charlson Comorbidity Index (CCI) calculation used in the MSCM thesis research.

## Overview

The analysis:
1. Implements the Quan (2011) Canadian adaptation of the Charlson index
2. Validates calculations against source data from CPCSSN checkpoint_1_20250318_024427
3. Generates enhanced visualizations and a detailed statistical report
4. Provides reproducible evidence for the CCI calculations used in the main analysis

## Requirements

- Python 3.8+
- Required packages:
  - pandas>=1.5.0
  - numpy>=1.23.0
  - matplotlib>=3.6.0
  - seaborn>=0.12.0
  - scipy>=1.9.0
  - pyarrow>=10.0.0
  - pdflatex (for report generation)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
charlson_validation/
├── charlson_validation.py   # Main analysis script
├── README.md               # This file
├── requirements.txt        # Python package requirements
├── charlson_report.tex     # Generated LaTeX report
├── charlson_results.parquet # Detailed results data
├── charlson_validation.log # Analysis log
└── figures/               # Generated visualizations
    ├── charlson_distribution.png
    ├── condition_prevalence_with_ci.png
    ├── comorbidity_correlation.png
    └── age_distribution.png
```

## Running the Analysis

1. Ensure you have access to the CPCSSN checkpoint data at:
   ```
   .../Notebooks/data/interim/checkpoint_1_20250318_024427/
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis script:
   ```bash
   python charlson_validation.py
   ```

4. Generate the PDF report:
   ```bash
   pdflatex charlson_report.tex
   ```

## Output Files

- `charlson_results.parquet`: Complete patient-level results including:
  - Individual Charlson condition flags
  - Total Charlson score
  - Patient IDs for linking to other analyses

- `charlson_report.pdf`: Enhanced report including:
  - Methods description
  - Summary statistics
  - Distribution analysis with statistical tests
  - Confidence intervals for prevalence estimates
  - Comparison with published literature
  - Enhanced visualizations

- `figures/`: Enhanced visualizations including:
  - Charlson score distribution
  - Condition prevalence with 95% confidence intervals
  - Comorbidity correlation heatmap
  - Age-stratified analysis (if demographic data available)

- `charlson_validation.log`: Detailed log of the analysis process including:
  - Data validation results
  - Processing statistics
  - Performance metrics
  - Warning messages

## Validation Approach

The script implements several validation checks:

1. **Data Validation**:
   - Required column presence
   - Data format validation
   - Missing value detection
   - ICD code format verification

2. **Code Completeness**:
   - Verifies all 17 Charlson conditions from Quan (2011)
   - Validates ICD-9/ICD-10 code mappings
   - Confirms weight assignments (1-6)

3. **Statistical Validation**:
   - Confidence intervals for prevalence estimates
   - Distribution analysis (normality tests)
   - Correlation analysis between conditions
   - Age-stratified analysis

4. **Performance Optimization**:
   - Vectorized calculations for large datasets
   - Efficient data structures
   - Progress monitoring

## References

1. Quan H et al. (2011). Updating and Validating the Charlson Comorbidity Index and Score for Risk Adjustment in Hospital Discharge Abstracts Using Data From 6 Countries. Am J Epidemiol 173(6):676-82.
2. Williamson T et al. (2020). Charlson Comorbidity Distribution in Canadian Primary Care. PLoS ONE 15(4): e0231635.

## Contact

For questions about this analysis, please contact the research team.
