# SSD Pipeline: 5-Minute Quick Start

## ⚡ **Immediate Execution**

```bash
# 1. Navigate to project directory
cd "SSD_Experiment1_Causal_Effect"

# 2. Build Docker image (5 minutes)
docker build -t ssd-pipeline:latest .

# 3. Run complete pipeline (2-3 hours)
docker run -it \
  -v "$PWD:/app" \
  -v "$PWD/data_derived:/app/data_derived" \
  ssd-pipeline:latest make all

# 4. Check results
docker run --rm -v "$PWD:/app" ssd-pipeline:latest ls -la data_derived/
```

## 📊 **Expected Output**

- **Duration**: 2-3 hours
- **Files**: 15+ parquet files in `data_derived/`
- **Key Result**: `patient_master_or.parquet` (256,746 × 79)
- **Exposure**: 142,986 patients exposed (55.9%)

## 🔍 **Logic Options**

| Command | Logic | Exposed | Purpose |
|---------|-------|---------|---------|
| `make all` | OR (default) | 142,986 (55.9%) | Primary analysis (DSM-5 proxy) |
| `make all TREATMENT_COL=ssd_flag_strict` | AND | 199 (0.08%) | Sensitivity (DSM-IV proxy) |
| `make compare_logic` | Both | - | Comparison |

## 🧬 **Treatment Column Options**

The pipeline supports two SSD definitions:

- **`ssd_flag` (default)**: OR logic - DSM-5 Somatic Symptom Disorder proxy
  - ≥3 normal labs OR ≥2 unresolved referrals OR ≥90 days psychotropic meds
- **`ssd_flag_strict`**: AND logic - DSM-IV Somatoform Disorders proxy  
  - ≥3 normal labs AND ≥2 unresolved referrals AND ≥90 days psychotropic meds

```bash
# Run strict DSM-IV sensitivity analysis
docker run -it -v "$PWD:/app" ssd-pipeline:latest make all TREATMENT_COL=ssd_flag_strict

# Run individual scripts with strict definition
docker run -it -v "$PWD:/app" ssd-pipeline:latest \
  python src/05_ps_match.py --treatment-col ssd_flag_strict
```

## 🛠️ **Common Commands**

```bash
# Individual stages
docker run -it -v "$PWD:/app" ssd-pipeline:latest make cohort
docker run -it -v "$PWD:/app" ssd-pipeline:latest make exposure  
docker run -it -v "$PWD:/app" ssd-pipeline:latest make master

# Debugging
docker run -it -v "$PWD:/app" ssd-pipeline:latest bash
docker run -it -v "$PWD:/app" ssd-pipeline:latest make help
docker run -it -v "$PWD:/app" ssd-pipeline:latest make test

# Validation  
docker run --rm -v "$PWD:/app" ssd-pipeline:latest make validate-quick
```

## ⚠️ **Prerequisites**

- Docker Desktop running
- 8GB+ RAM available
- 10GB+ free disk space
- In `SSD_Experiment1_Causal_Effect` directory

## 📞 **If Issues**

1. Check Docker: `docker --version && docker info`
2. Rebuild clean: `docker build -t ssd-pipeline:latest . --no-cache`
3. View logs: `docker run --rm -v "$PWD:/app" ssd-pipeline:latest tail *.log`
4. Consult: `DOCKER_EXECUTION_GUIDE.md`

---
**Status**: ✅ Production Ready (Validated June 16, 2025) 