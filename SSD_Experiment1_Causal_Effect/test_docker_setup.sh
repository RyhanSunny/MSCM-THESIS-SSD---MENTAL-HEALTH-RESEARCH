#!/bin/bash
# Test script for Docker setup
# Author: Ryhan Suny
# Date: 2025-07-01

echo "========================================"
echo "SSD Pipeline Docker Setup Test"
echo "========================================"

# Check if Docker is installed and running
echo -n "1. Checking Docker installation... "
if command -v docker &> /dev/null; then
    echo "✓ Docker installed"
    docker --version
else
    echo "✗ Docker not found. Please install Docker."
    exit 1
fi

echo -n "2. Checking Docker daemon... "
if docker info &> /dev/null; then
    echo "✓ Docker daemon running"
else
    echo "✗ Docker daemon not running. Please start Docker."
    exit 1
fi

# Check if docker-compose is installed
echo -n "3. Checking docker-compose... "
if command -v docker-compose &> /dev/null; then
    echo "✓ docker-compose installed"
    docker-compose --version
else
    echo "⚠ docker-compose not found. Using docker compose plugin..."
    if docker compose version &> /dev/null; then
        echo "✓ Docker compose plugin available"
    else
        echo "✗ Neither docker-compose nor docker compose plugin found"
    fi
fi

# Check required directories
echo -n "4. Checking required directories... "
missing_dirs=()
for dir in src Makefile environment.yml; do
    if [ ! -e "$dir" ]; then
        missing_dirs+=("$dir")
    fi
done

if [ ${#missing_dirs[@]} -eq 0 ]; then
    echo "✓ All required files present"
else
    echo "✗ Missing: ${missing_dirs[*]}"
    exit 1
fi

# Build test
echo -n "5. Testing Docker build... "
if docker build -t ssd-pipeline:test --target base . &> /dev/null; then
    echo "✓ Base build successful"
else
    echo "✗ Docker build failed"
    exit 1
fi

# Test Python environment
echo -n "6. Testing Python environment... "
if docker run --rm ssd-pipeline:latest python -c "print('Hello from Python')" &> /dev/null; then
    echo "✓ Python working"
else
    echo "✗ Python test failed"
fi

# Test conda environment
echo -n "7. Testing conda packages... "
if docker run --rm ssd-pipeline:latest python -c "import pandas, numpy; print('Core packages OK')" &> /dev/null; then
    echo "✓ Core packages installed"
else
    echo "✗ Package import failed"
fi

# Test make command
echo -n "8. Testing make command... "
if docker run --rm ssd-pipeline:latest make help &> /dev/null; then
    echo "✓ Make command working"
else
    echo "✗ Make command failed"
fi

echo ""
echo "========================================"
echo "Docker setup test complete!"
echo "To run the full pipeline:"
echo "  docker-compose up ssd-pipeline"
echo "Or:"
echo "  docker run -it -v \"\$PWD:/app\" ssd-pipeline:latest make all"
echo "========================================"