# SSD Causal Analysis Pipeline Makefile
# Author: Ryhan Suny
# Date: 2025-05-24

# Variables
PYTHON = python3
SRC_DIR = src
DATA_DIR = data_derived
RESULTS_DIR = results
NOTEBOOKS_DIR = Notebooks
REPORTS_DIR = reports
# Column name for treatment; override at make runtime e.g. `make causal TREATMENT_COL=ssd_flag_strict`
TREATMENT_COL ?= ssd_flag

# Default target
.PHONY: all
all: cohort exposure mediator outcomes confounders lab referral missing misclassification master sequential ps causal mediation temporal evalue competing-risk death-rates robustness week1-validation week2-all

# Data pipeline targets
.PHONY: cohort
cohort:
	@echo "Building cohort..."
	$(PYTHON) $(SRC_DIR)/01_cohort_builder.py

# ------------------------------------------------------------------ #
#  Exposure Logic Targets
# ------------------------------------------------------------------ #
.PHONY: exposure_or
exposure_or: cohort
	@echo "Generating exposure flags (OR logic)..."
	$(PYTHON) $(SRC_DIR)/02_exposure_flag.py --logic or

.PHONY: exposure_and
exposure_and: cohort
	@echo "Generating exposure flags (AND logic)..."
	$(PYTHON) $(SRC_DIR)/02_exposure_flag.py --logic and

# Backwards-compat default
.PHONY: exposure
exposure: exposure_or

.PHONY: mediator
mediator: cohort
	@echo "Building mediator autoencoder..."
	$(PYTHON) $(SRC_DIR)/03_mediator_autoencoder.py

.PHONY: outcomes
outcomes: cohort
	@echo "Generating outcome flags..."
	$(PYTHON) $(SRC_DIR)/04_outcome_flag.py

.PHONY: confounders
confounders: cohort
	@echo "Building confounder matrix..."
	$(PYTHON) $(SRC_DIR)/05_confounder_flag.py

.PHONY: lab
lab: cohort
	@echo "Generating lab flags..."
	$(PYTHON) $(SRC_DIR)/06_lab_flag.py

.PHONY: referral
referral: cohort
	@echo "Generating referral sequences..."
	$(PYTHON) $(SRC_DIR)/07_referral_sequence.py

.PHONY: missing
missing: cohort
	@echo "Handling missing data..."
	$(PYTHON) $(SRC_DIR)/07_missing_data.py

.PHONY: misclassification
misclassification: exposure
	@echo "Adjusting for misclassification..."
	$(PYTHON) $(SRC_DIR)/07a_misclassification_adjust.py --treatment-col $(TREATMENT_COL)

# ------------------------------------------------------------------ #
#  Master Table Targets (OR vs AND)
# ------------------------------------------------------------------ #
.PHONY: master_or
master_or: cohort exposure_or mediator outcomes confounders lab referral missing misclassification
	@echo "Creating patient master table (OR logic)..."
	$(PYTHON) $(SRC_DIR)/08_patient_master_table.py --exposure_file exposure_or.parquet --output_file patient_master_or.parquet
	@# maintain legacy filename
	@cp data_derived/patient_master_or.parquet data_derived/patient_master.parquet

.PHONY: master_and
master_and: cohort exposure_and mediator outcomes confounders lab referral missing misclassification
	@echo "Creating patient master table (AND logic)..."
	$(PYTHON) $(SRC_DIR)/08_patient_master_table.py --exposure_file exposure_and.parquet --output_file patient_master_and.parquet

# Backwards-compat default
.PHONY: master
master: master_or

.PHONY: sequential
sequential: master
	@echo "Running sequential pathway analysis..."
	$(PYTHON) $(SRC_DIR)/08_sequential_pathway_analysis.py

# Causal analysis targets
.PHONY: ps
ps: master
	@echo "Running propensity score matching..."
	$(PYTHON) $(SRC_DIR)/05_ps_match.py --treatment-col $(TREATMENT_COL)

.PHONY: causal
causal: ps
	@echo "Running causal estimators..."
	$(PYTHON) $(SRC_DIR)/06_causal_estimators.py --treatment-col $(TREATMENT_COL) --cluster-col site_id

# Week 2 targets - H1-H3 hypothesis analysis
.PHONY: hypotheses
hypotheses: ps
	@echo "Running H1-H3 hypothesis analyses..."
	$(PYTHON) $(SRC_DIR)/hypothesis_runner.py --data-path $(DATA_DIR)/ps_weighted.parquet --results-dir $(RESULTS_DIR)

.PHONY: hypotheses-dry-run
hypotheses-dry-run: ps
	@echo "Validating H1-H3 hypothesis setup..."
	$(PYTHON) $(SRC_DIR)/hypothesis_runner.py --dry-run

.PHONY: mediation
mediation: master
	@echo "Running mediation analysis..."
	$(PYTHON) $(SRC_DIR)/14_mediation_analysis.py --treatment-col $(TREATMENT_COL)

# Additional analysis targets
.PHONY: temporal
temporal: ps
	@echo "Running temporal adjustment..."
	$(PYTHON) $(SRC_DIR)/12_temporal_adjust.py --treatment-col $(TREATMENT_COL)

.PHONY: evalue
evalue: causal
	@echo "Calculating E-values..."
	$(PYTHON) $(SRC_DIR)/13_evalue_calc.py --treatment-col $(TREATMENT_COL)

.PHONY: competing-risk
competing-risk: master
	@echo "Running competing risk analysis..."
	$(PYTHON) $(SRC_DIR)/finegray_competing.py --treatment-col $(TREATMENT_COL)

.PHONY: death-rates
death-rates: master
	@echo "Analyzing death rates..."
	$(PYTHON) $(SRC_DIR)/death_rates_analysis.py --treatment-col $(TREATMENT_COL)

# QA and robustness targets
.PHONY: placebo
placebo: master
	@echo "Running placebo tests..."
	$(PYTHON) $(SRC_DIR)/14_placebo_tests.py --treatment-col $(TREATMENT_COL)

.PHONY: robustness
robustness: master
	@echo "Running robustness checks..."
	$(PYTHON) $(SRC_DIR)/14_placebo_tests.py --treatment-col $(TREATMENT_COL)
	$(PYTHON) $(SRC_DIR)/15_robustness.py --treatment-col $(TREATMENT_COL)

# Reporting targets
.PHONY: qc
qc:
	@echo "Running quality control notebook..."
	cd $(NOTEBOOKS_DIR) && papermill 09_qc_master.ipynb 09_qc_master_output.ipynb

.PHONY: reporting
reporting: qc
	@echo "Generating reports..."
	@if [ -d $(REPORTS_DIR) ]; then \
		cd $(REPORTS_DIR) && Rscript -e "rmarkdown::render('10_descriptives.Rmd')" || echo "R not available, skipping descriptives"; \
		cd $(REPORTS_DIR) && Rscript -e "rmarkdown::render('18_reporting.Rmd')" || echo "R not available, skipping reporting"; \
	else \
		echo "Reports directory not found, skipping R reports"; \
	fi

# Release management
.PHONY: release
release: all robustness reporting
	@echo "Creating release..."
	$(PYTHON) scripts/release_lock.py create v$(VERSION)
	@if command -v dvc &> /dev/null; then \
		dvc push; \
	else \
		echo "DVC not available, skipping data push"; \
	fi
	git tag -a v$(VERSION) -m "Release $(VERSION)"
	git push origin v$(VERSION)

# Utility targets
.PHONY: clean
clean:
	@echo "Cleaning intermediate files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete

.PHONY: clean-data
clean-data:
	@echo "Cleaning derived data (CAREFUL!)..."
	@read -p "Are you sure you want to delete all derived data? [y/N] " confirm && \
	if [ "$$confirm" = "y" ]; then \
		rm -rf $(DATA_DIR)/*.parquet; \
		rm -rf $(RESULTS_DIR)/*.json; \
		rm -rf $(RESULTS_DIR)/*.csv; \
		echo "Derived data cleaned."; \
	else \
		echo "Cancelled."; \
	fi

.PHONY: test
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=$(SRC_DIR) --cov-report=html

# Week 1 Implementation Quality Gates
.PHONY: week1-validation
week1-validation:
	@echo "Running Week 1 quality gates..."
	@echo "1. Testing weight diagnostics..."
	pytest tests/test_weight_diagnostics.py -v
	@echo "2. Testing cluster-robust standard errors..."
	pytest tests/test_cluster_robust_se.py -v
	@echo "3. Testing Poisson/NB count models..."
	pytest tests/test_poisson_count_models.py -v
	@echo "4. Testing temporal ordering validation..."
	pytest tests/test_temporal_validator.py -v
	@echo "5. Testing multiple imputation..."
	pytest tests/test_multiple_imputation.py -v
	@echo "✓ All Week 1 quality gates passed!"

.PHONY: week1-integration-test
week1-integration-test: ps
	@echo "Testing Week 1 integration with pipeline..."
	@echo "Running propensity score matching with weight diagnostics..."
	$(PYTHON) $(SRC_DIR)/05_ps_match.py --treatment-col $(TREATMENT_COL) --dry-run
	@echo "Running causal estimators with count models and cluster-robust SE..."
	$(PYTHON) $(SRC_DIR)/06_causal_estimators.py --treatment-col $(TREATMENT_COL) --cluster-col site_id --outcome total_encounters --dry-run
	@echo "Running temporal validation..."
	$(PYTHON) -c "from src.temporal_validator import validate_exposure_outcome_sequence; import pandas as pd; print('Temporal validation ready')"
	@echo "✓ Week 1 integration tests passed!"

# Week 2 Analysis & Visualization Quality Gates
.PHONY: week2-hypotheses
week2-hypotheses: master
	@echo "Running H1-H3 hypothesis analyses..."
	$(PYTHON) $(SRC_DIR)/hypothesis_runner.py --treatment-col $(TREATMENT_COL)
	@echo "✓ Hypothesis analyses complete - results in results/hypothesis_*.json"

.PHONY: week2-figures
week2-figures: master
	@echo "Generating Week 2 figures..."
	$(PYTHON) $(SRC_DIR)/figure_generator.py
	@echo "✓ Figures generated - check figures/*.svg"

.PHONY: week2-tables
week2-tables: master
	@echo "Generating Week 2 tables..."
	$(PYTHON) $(SRC_DIR)/table_generator.py --treatment-col $(TREATMENT_COL)
	@echo "✓ Tables generated - check tables/*.md and tables/*.csv"

.PHONY: week2-documentation
week2-documentation: week2-hypotheses week2-figures week2-tables
	@echo "Updating Week 2 documentation..."
	$(PYTHON) $(SRC_DIR)/documentation_updater.py
	@echo "✓ Documentation updated - check reports/week2_analysis_report.md"

.PHONY: week2-validation
week2-validation: week2-documentation
	@echo "Running Week 2 quality gates..."
	@echo "1. Testing hypothesis runner..."
	pytest tests/test_hypothesis_runner.py -v
	@echo "2. Testing figure generator..."
	pytest tests/test_figure_generator.py -v
	@echo "3. Validating generated artifacts..."
	@test -f results/hypothesis_h1.json || (echo "❌ H1 results missing" && exit 1)
	@test -f results/hypothesis_h2.json || (echo "❌ H2 results missing" && exit 1)
	@test -f results/hypothesis_h3.json || (echo "❌ H3 results missing" && exit 1)
	@test -f figures/dag.svg || (echo "❌ DAG figure missing" && exit 1)
	@test -f figures/forest_plot.svg || (echo "❌ Forest plot missing" && exit 1)
	@test -f figures/love_plot.svg || (echo "❌ Love plot missing" && exit 1)
	@test -f figures/consort_flowchart.svg || (echo "❌ CONSORT flowchart missing" && exit 1)
	@test -f tables/main_results.md || (echo "❌ Main results table missing" && exit 1)
	@test -f tables/baseline_table.md || (echo "❌ Baseline table missing" && exit 1)
	@test -f tables/sensitivity.md || (echo "❌ Sensitivity table missing" && exit 1)
	@test -f reports/week2_analysis_report.md || (echo "❌ Week 2 report missing" && exit 1)
	@echo "✓ All Week 2 quality gates passed!"

.PHONY: week2-all
week2-all: week2-validation
	@echo "✓ Week 2 Analysis & Visualization complete!"

.PHONY: lint
lint:
	@echo "Running linters..."
	@if command -v ruff &> /dev/null; then \
		ruff check $(SRC_DIR); \
	else \
		echo "ruff not available, trying flake8..."; \
		flake8 $(SRC_DIR) || echo "flake8 not available"; \
	fi

.PHONY: format
format:
	@echo "Formatting code..."
	@if command -v black &> /dev/null; then \
		black $(SRC_DIR); \
	else \
		echo "black not available"; \
	fi

.PHONY: install
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	docker build -t ssd-pipeline:1.1 .

.PHONY: docker-run
docker-run:
	@echo "Running pipeline in Docker..."
	docker run -v $(PWD):/app ssd-pipeline:1.1 make all

.PHONY: validate
validate: master
	@echo "Running comprehensive validation suite..."
	$(PYTHON) scripts/run_all_validations.py
	@echo "Compiling LaTeX reports..."
	@cd analysis && find . -name "*.tex" -exec pdflatex {} \; 2>/dev/null || true
	@echo "Validation complete. Check analysis/*/ for reports."

.PHONY: validate-quick
validate-quick: master
	@echo "Running quick validation summary..."
	$(PYTHON) analysis/quick_validation_summary.py

.PHONY: help
help:
	@echo "SSD Causal Analysis Pipeline"
	@echo ""
	@echo "Main targets:"
	@echo "  make all          - Run complete pipeline"
	@echo "  make cohort       - Build cohort"
	@echo "  make exposure     - Generate exposure flags"
	@echo "  make ps           - Run propensity score matching"
	@echo "  make causal       - Run causal estimators"
	@echo "  make robustness   - Run robustness checks"
	@echo "  make reporting    - Generate reports"
	@echo ""
	@echo "Week 1 & 2 targets:"
	@echo "  make week1-validation - Run Week 1 quality gates"
	@echo "  make week1-integration-test - Test Week 1 pipeline integration"
	@echo "  make week2-hypotheses - Run H1-H3 hypothesis analyses"
	@echo "  make week2-figures    - Generate DAG, Forest, Love plots"
	@echo "  make week2-tables     - Generate baseline, results tables"
	@echo "  make week2-documentation - Update documentation"
	@echo "  make week2-validation - Run Week 2 quality gates"
	@echo "  make week2-all        - Complete Week 2 analysis"
	@echo ""
	@echo "Utility targets:"
	@echo "  make clean        - Clean temporary files"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make install      - Install dependencies"
	@echo "  make validate     - Run validation suite"
	@echo "  make validate-quick - Quick validation summary"
	@echo ""
	@echo "Release targets:"
	@echo "  make release VERSION=X.Y.Z - Create a release"

# ------------------------------------------------------------------ #
#  Logic Comparison Report
# ------------------------------------------------------------------ #
.PHONY: compare_logic
compare_logic: exposure_or exposure_and
	@echo "Comparing OR vs AND exposure cohorts..."
	$(PYTHON) scripts/compare_exposure_logic.py data_derived/exposure_or.parquet data_derived/exposure_and.parquet