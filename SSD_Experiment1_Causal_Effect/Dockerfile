# Dockerfile for SSD Causal Analysis Pipeline

# Use official Python base image
FROM python:3.12-slim

# Set reproducibility
ENV PYTHONHASHSEED=42

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        make \
        && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies with specific versions
RUN pip install --upgrade pip && \
    pip install pandas numpy scikit-learn matplotlib \
    lightgbm==4.3.0 \
    miceforest==6.* \
    econml==0.15.* \
    xgboost dowhy jupyterlab \
    pytest pytest-cov

# Note: dagitty is an R package, not Python. To install it:
# (Optional) Install R and tmle3/grf/dagitty for full parity
# RUN apt-get update && apt-get install -y r-base && \
#     R -e "install.packages(c('tmle3', 'grf', 'dagitty'), repos='https://cloud.r-project.org/')"

# For full reproducibility, set random seeds in your Python scripts:
# import numpy as np, random, os
# np.random.seed(42); random.seed(42); os.environ["PYTHONHASHSEED"] = "42"

# Expose Jupyter port
EXPOSE 8888

# Default command: start a shell (user can override to run pipeline or notebook)
CMD ["/bin/bash"] 