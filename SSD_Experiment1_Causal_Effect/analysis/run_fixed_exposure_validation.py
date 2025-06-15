#!/usr/bin/env python3
"""
Run the fixed exposure validation with all improvements
"""

import subprocess
import sys

# Run the fixed script by importing and calling main
sys.path.insert(0, '.')

from analysis.exposure_validation_enhanced import main

if __name__ == "__main__":
    # Just call the original main function
    main()