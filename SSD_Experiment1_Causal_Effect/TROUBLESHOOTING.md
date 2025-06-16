# SSD Pipeline: Troubleshooting Guide

## 🚨 **Quick Diagnostics**

### 1. Docker Environment Check
```bash
# Test basic Docker functionality
docker --version          # Should show version 20+
docker info               # Should show server info
docker ps                 # Should list running containers
```

### 2. Pipeline Status Check
```bash
# Check if build completed
docker images | grep ssd-pipeline

# Check data availability
docker run --rm -v "$PWD:/app" ssd-pipeline:latest ls -la Notebooks/data/interim/

# Validate pipeline structure
docker run --rm -v "$PWD:/app" ssd-pipeline:latest make help
```

## 🔧 **Common Issues & Solutions**

### Issue 1: Docker Build Fails
**Symptoms:**
- `failed to checksum file venv/bin/python`
- `archive/tar: unknown file mode`

**Solution:**
```bash
# Clean build context
rm -rf venv __pycache__ .pytest_cache
docker system prune -f
docker build -t ssd-pipeline:latest . --no-cache
```

### Issue 2: Out of Memory
**Symptoms:**
- `docker: Error response from daemon: not enough memory`  
- Process killed during execution

**Solution:**
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory → 12GB+

# Alternative: Run stages individually
docker run -it -v "$PWD:/app" ssd-pipeline:latest make cohort
docker run -it -v "$PWD:/app" ssd-pipeline:latest make exposure
# ... continue stage by stage
```

### Issue 3: Permission Denied
**Symptoms:**
- `permission denied while trying to connect to Docker daemon`
- `Got permission denied while trying to connect to Docker`

**Solution:**
```bash
# Windows: Run PowerShell as Administrator
# Right-click PowerShell → "Run as administrator"

# Linux/macOS: Add user to docker group
sudo usermod -aG docker $USER
# Then logout and login again
```

### Issue 4: Data Files Missing
**Symptoms:**
- `FileNotFoundError: [Errno 2] No such file or directory: 'cohort.parquet'`
- `Required file cohort.parquet does not exist`

**Solution:**
```bash
# Check data checkpoint location
ls -la "Notebooks/data/interim/checkpoint_1_20250318_024427/"

# If data missing, verify mount points
docker run --rm -v "$PWD:/app" ssd-pipeline:latest find /app -name "*.csv" | head -10

# Run with correct data path
docker run -it \
  -v "$PWD:/app" \
  -v "$PWD/Notebooks:/app/Notebooks" \
  ssd-pipeline:latest make cohort
```

### Issue 5: Python Import Errors
**Symptoms:**
- `ModuleNotFoundError: No module named 'pandas'`
- `ImportError: cannot import name 'xxx'`

**Solution:**
```bash
# Rebuild image with fresh dependencies
docker build -t ssd-pipeline:latest . --no-cache

# Test environment inside container
docker run -it ssd-pipeline:latest python -c "import pandas; print(pandas.__version__)"
```

### Issue 6: Pipeline Hangs/Stalls
**Symptoms:**
- Process appears stuck
- No output for extended periods

**Solution:**
```bash
# Monitor container in separate terminal
docker stats

# Check logs
docker run --rm -v "$PWD:/app" ssd-pipeline:latest tail -f *.log

# Run with verbose output
docker run -it -v "$PWD:/app" ssd-pipeline:latest make all -d
```

## 🔍 **Advanced Debugging**

### Memory Usage Monitoring
```bash
# Real-time memory usage
docker stats --no-stream

# Check container resource limits
docker run --rm ssd-pipeline:latest cat /proc/meminfo | head -5
```

### Component Testing
```bash
# Test individual scripts
docker run --rm -v "$PWD:/app" ssd-pipeline:latest python src/01_cohort_builder.py --help
docker run --rm -v "$PWD:/app" ssd-pipeline:latest python src/02_exposure_flag.py --logic or --dry-run

# Run test suite
docker run --rm -v "$PWD:/app" ssd-pipeline:latest make test
```

### Data Validation
```bash
# Check data integrity
docker run --rm -v "$PWD:/app" ssd-pipeline:latest python -c "
import pandas as pd
df = pd.read_csv('Notebooks/data/interim/checkpoint_1_20250318_024427/PatientDemographic_merged_prepared.csv')
print(f'Patients: {len(df):,}')
print(f'Columns: {df.columns.tolist()[:5]}...')
"
```

## 📊 **Performance Troubleshooting**

### Slow Execution
**Check:**
- Available RAM: Should be 12GB+ for full pipeline
- CPU usage: Should utilize multiple cores
- Disk I/O: SSD recommended for data_derived/ directory

**Optimize:**
```bash
# Use tmpfs for temporary data (Linux/macOS)
docker run -it \
  --tmpfs /tmp:rw,size=2g \
  -v "$PWD:/app" \
  ssd-pipeline:latest make all

# Limit concurrent processing
export MAKEFLAGS="-j2"  # Limit to 2 parallel jobs
docker run -it -v "$PWD:/app" ssd-pipeline:latest make all
```

### GPU Issues (if applicable)
```bash
# Check GPU availability
docker run --rm --gpus all ssd-pipeline:latest nvidia-smi

# Run without GPU if needed
docker run -it -v "$PWD:/app" ssd-pipeline:latest \
  python src/05_ps_match.py --no-gpu
```

## 📝 **Logging & Diagnostics**

### Enable Verbose Logging
```bash
# Set debug level
export PYTHONPATH=/app/src
export LOG_LEVEL=DEBUG

docker run -it \
  -e LOG_LEVEL=DEBUG \
  -v "$PWD:/app" \
  ssd-pipeline:latest make all
```

### Save Full Logs
```bash
# Capture all output
docker run -it -v "$PWD:/app" ssd-pipeline:latest make all 2>&1 | tee pipeline_full.log

# Extract specific errors
grep -i "error\|failed\|exception" pipeline_full.log
```

## 🆘 **When All Else Fails**

### Nuclear Option: Complete Reset
```bash
# Remove all Docker containers and images
docker system prune -a -f
docker volume prune -f

# Remove all generated files
rm -rf data_derived/ results/ *.log

# Start completely fresh
docker build -t ssd-pipeline:latest . --no-cache
docker run -it -v "$PWD:/app" ssd-pipeline:latest make all
```

### Contact Information
If issues persist:
1. **Check logs**: Review `*.log` files for specific error messages
2. **Validate environment**: Ensure Docker has sufficient resources
3. **Test components**: Run individual pipeline stages
4. **Document issue**: Note exact error message and steps to reproduce

---

**Last Updated**: June 16, 2025  
**Pipeline Version**: Production Ready  
**Docker Image**: `ssd-pipeline:latest` 