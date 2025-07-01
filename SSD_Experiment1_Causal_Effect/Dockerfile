# Dockerfile for SSD Causal Analysis Pipeline

# Use official Python base image matching environment.yml
FROM python:3.11-slim

# Set reproducibility
ENV PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set environment hash for lockfile verification
COPY environment.yml /tmp/environment.yml
RUN ENV_HASH=$(sha256sum /tmp/environment.yml | cut -d' ' -f1) && \
    echo "ENV_HASH=${ENV_HASH}" >> /etc/environment

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        make \
        wget \
        r-base \
        r-base-dev \
        && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Update conda base environment with dependencies from environment.yml
# This matches your local setup using base environment
RUN conda update -n base conda && \
    conda env update -n base -f environment.yml && \
    conda clean -afy

# Use base environment (no activation needed)
SHELL ["/bin/bash", "-c"]

# Install critical R packages for survival analysis and causal inference
# Add missing packages from environment.yml
RUN R -e "install.packages(c('survival', 'cmprsk', 'grf', 'tmle', 'dagitty', 'MatchIt', 'sandwich', 'lmtest', 'pbapply', 'doParallel'), repos='https://cloud.r-project.org/', dependencies=TRUE)"

# Comprehensive dependency verification
RUN echo "=== Verifying Python packages ===" && \
    python -c "import pandas, numpy, sklearn, statsmodels, matplotlib, dowhy, econml, rpy2; print('✓ Core Python packages verified')" && \
    echo "=== Verifying R packages ===" && \
    R -e "library(survival); library(cmprsk); library(grf); library(tmle); cat('✓ R packages verified\n')" && \
    echo "=== Verifying rpy2 bridge ===" && \
    python -c "import rpy2.robjects as ro; ro.r('library(survival)'); print('✓ rpy2 bridge verified')" && \
    echo "=== All dependencies verified successfully ==="

# For full reproducibility, set random seeds in your Python scripts:
# import numpy as np, random, os
# np.random.seed(42); random.seed(42); os.environ["PYTHONHASHSEED"] = "42"

# Expose Jupyter port
EXPOSE 8888

# Create directories for data volumes
RUN mkdir -p /app/data_derived /app/results /app/logs /app/figures /app/tables /app/reports

# Default command: use bash in base environment
CMD ["/bin/bash"] 