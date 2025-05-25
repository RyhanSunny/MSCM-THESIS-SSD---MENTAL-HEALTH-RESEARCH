#!/usr/bin/env python3
"""
update_study_doc.py

Utility to update the study YAML documentation file after each pipeline step.

- Loads the most recent results/study_documentation_*.yaml (or creates a new one if none exists)
- Patches/appends keys passed via --step and optional --kv key=value arguments
- Saves a new YAML with a fresh timestamp (YYYYMMDD_HHMMSS)
- Prints the path to the new YAML file for CI/logging

Usage:
    python scripts/update_study_doc.py --step "Cohort built" --kv cohort_rows=250025
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import yaml

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

def find_latest_yaml():
    yamls = sorted(RESULTS_DIR.glob("study_documentation_*.yaml"), reverse=True)
    return yamls[0] if yamls else None

def load_yaml(path):
    if path and path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}

def save_yaml(data, step_desc):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"study_documentation_{timestamp}.yaml"
    # Add a log of this step
    if "log" not in data:
        data["log"] = []
    data["log"].append({"timestamp": timestamp, "step": step_desc})
    with open(out_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(str(out_path))
    return out_path

def parse_kv_list(kv_list):
    result = {}
    for kv in kv_list or []:
        if "=" not in kv:
            print(f"Warning: Skipping malformed --kv argument: {kv}", file=sys.stderr)
            continue
        k, v = kv.split("=", 1)
        # Try to parse numbers
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        result[k.strip()] = v
    return result

def main():
    parser = argparse.ArgumentParser(description="Update study documentation YAML.")
    parser.add_argument("--step", required=True, help="Description of the pipeline step completed.")
    parser.add_argument("--kv", nargs="*", help="Key-value pairs to add/update in the YAML (format: key=value).")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    latest_yaml = find_latest_yaml()
    data = load_yaml(latest_yaml)
    # Patch/append keys
    kv_dict = parse_kv_list(args.kv)
    data.update(kv_dict)
    # Save new YAML with updated log
    save_yaml(data, args.step)

if __name__ == "__main__":
    main() 