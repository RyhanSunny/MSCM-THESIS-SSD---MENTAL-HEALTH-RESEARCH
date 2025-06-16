# SSD Causal Analysis Pipeline Makefile
# Author: Ryhan Suny
# Date: 2025-05-24

# Variables
PYTHON = python
SRC_DIR = src
DATA_DIR = data_derived
RESULTS_DIR = results
NOTEBOOKS_DIR = Notebooks
REPORTS_DIR = reports

# Default target
.PHONY: all
all: cohort exposure mediator outcomes confounders lab referral missing master sequential ps causal temporal evalue competing-risk death-rates robustness

# Data pipeline targets
.PHONY: cohort
cohort:
	@echo "Building cohort..."
	$(PYTHON) $(SRC_DIR)/01_cohort_builder.py

.PHONY: exposure
exposure: cohort
	@echo "Generating exposure flags..."
	$(PYTHON) $(SRC_DIR)/02_exposure_flag.py

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

.PHONY: master
master: cohort exposure mediator outcomes confounders lab referral
	@echo "Creating patient master table..."
	$(PYTHON) $(SRC_DIR)/08_patient_master_table.py

.PHONY: sequential
sequential: master
	@echo "Running sequential pathway analysis..."
	$(PYTHON) $(SRC_DIR)/08_sequential_pathway_analysis.py

# Causal analysis targets
.PHONY: ps
ps: master
	@echo "Running propensity score matching..."
	$(PYTHON) $(SRC_DIR)/05_ps_match.py

.PHONY: causal
causal: ps
	@echo "Running causal estimators..."
	$(PYTHON) $(SRC_DIR)/06_causal_estimators.py

# Additional analysis targets
.PHONY: temporal
temporal: ps
	@echo "Running temporal adjustment..."
	$(PYTHON) $(SRC_DIR)/12_temporal_adjust.py

.PHONY: evalue
evalue: causal
	@echo "Calculating E-values..."
	$(PYTHON) $(SRC_DIR)/13_evalue_calc.py

.PHONY: competing-risk
competing-risk: master
	@echo "Running competing risk analysis..."
	$(PYTHON) $(SRC_DIR)/finegray_competing.py

.PHONY: death-rates
death-rates: master
	@echo "Analyzing death rates..."
	$(PYTHON) $(SRC_DIR)/death_rates_analysis.py

# QA and robustness targets
.PHONY: placebo
placebo: master
	@echo "Running placebo tests..."
	$(PYTHON) $(SRC_DIR)/14_placebo_tests.py

.PHONY: robustness
robustness: master
	@echo "Running robustness checks..."
	$(PYTHON) $(SRC_DIR)/14_placebo_tests.py
	$(PYTHON) $(SRC_DIR)/15_robustness.py

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