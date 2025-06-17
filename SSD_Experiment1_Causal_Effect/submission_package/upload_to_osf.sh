#!/bin/bash
# OSF Upload Script for SSD Causal Analysis
# 
# Prerequisites:
# 1. Install OSF CLI: pip install osfclient
# 2. Create OSF project and get project ID
# 3. Generate OSF personal access token
#
# Usage: ./upload_to_osf.sh [PROJECT_ID] [TOKEN]

set -e

PROJECT_ID=${1:-"your-project-id"}
OSF_TOKEN=${2:-$OSF_TOKEN}

if [ -z "$OSF_TOKEN" ]; then
    echo "Error: OSF token required"
    echo "Usage: $0 [PROJECT_ID] [TOKEN]"
    echo "Or set OSF_TOKEN environment variable"
    exit 1
fi

echo "Uploading SSD Analysis to OSF..."
echo "Project ID: $PROJECT_ID"

# Find latest submission package
PACKAGE=$(ls -t SSD_Week3_*.zip | head -1)

if [ -z "$PACKAGE" ]; then
    echo "Error: No submission package found"
    exit 1
fi

echo "Package: $PACKAGE"

# Upload using OSF CLI
osf -p $PROJECT_ID upload $PACKAGE /manuscripts/

# Upload individual bundles
for bundle in *bundle*.zip; do
    if [ -f "$bundle" ]; then
        echo "Uploading $bundle..."
        osf -p $PROJECT_ID upload "$bundle" /supplements/
    fi
done

echo "Upload complete!"
echo "Visit: https://osf.io/$PROJECT_ID"
