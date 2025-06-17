# SSD Causal Analysis - Manuscript Submission Package

**Generated**: 2025-06-17 09:08:11  
**Version**: Week 3 Final  
**Author**: Ryhan Suny, Toronto Metropolitan University

## Contents

This package contains all artifacts for the manuscript:
*"Causal Effect of Somatic Symptom Patterns on Healthcare Utilization: A Population-Based Cohort Study"*

### 📊 Figures
- `figures/`: All publication-ready figures
- `figures/hires/`: High-resolution versions (≥300 DPI)
- DAG, forest plot, love plot, CONSORT flowchart

### 📋 Tables  
- `tables/`: Main results tables in Markdown and CSV
- Baseline characteristics, hypothesis results, sensitivity analyses

### 📚 Documentation
- `documentation/Methods_Supplement.md`: Detailed statistical methods
- `documentation/STROBE_CI_Checklist.md`: Reporting checklist
- `documentation/ROBINS_I_Assessment.md`: Bias assessment
- `documentation/Glossary.md`: Terms and definitions

### 📈 Results
- `results/`: JSON files with numerical results
- Hypothesis H1-H3 effect estimates and confidence intervals
- Power analysis and sample size calculations

### 💻 Code
- `code/`: Source code for reproducibility
- Complete pipeline from cohort building to analysis
- Docker environment and dependencies

### 📦 Bundles
- `bundles/`: Individual component bundles
- Figures, documentation, and code packages

## Reproducibility

All analyses can be reproduced using:
```bash
# Build environment
docker build -t ssd-pipeline:1.1 .

# Run complete pipeline
make all

# Generate Week 3 artifacts
make week3-all
```

## Quality Assurance

- ✅ All figures at publication quality (≥300 DPI)
- ✅ Complete statistical analysis following STROBE-CI guidelines
- ✅ Comprehensive bias assessment (ROBINS-I)
- ✅ Code tested and documented
- ✅ Reproducible environment specified

## File Integrity

See `MANIFEST.json` for complete file listing and metadata.
All files validated for completeness and format compliance.

## Contact

**Ryhan Suny**  
Email: sajibrayhan.suny@torontomu.ca  
ORCID: 0000-0000-0000-0001  
Toronto Metropolitan University

## Acknowledgments

Data provided by Canadian Primary Care Sentinel Surveillance Network (CPCSSN).  
Research supported by Car4Mind team, University of Toronto.

---
*Generated with automated submission packager v3.0*
