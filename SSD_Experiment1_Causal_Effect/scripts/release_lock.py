#!/usr/bin/env python3
"""
release_lock.py

Release lock mechanism to ensure reproducible releases.
Prevents accidental modifications to tagged releases.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import subprocess
import hashlib

# Lock file location
LOCK_FILE = Path(__file__).resolve().parents[1] / ".release_lock.json"

def get_git_info():
    """Get current git commit hash and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            text=True
        ).strip()
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
            text=True
        ).strip()
        
        # Check if there are uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], 
            text=True
        ).strip()
        
        has_changes = bool(status)
        
        return {
            "commit": commit,
            "branch": branch,
            "has_uncommitted_changes": has_changes
        }
    except subprocess.CalledProcessError as e:
        print(f"Error getting git info: {e}", file=sys.stderr)
        sys.exit(1)

def calculate_checksum(patterns=None):
    """Calculate checksum of important files."""
    if patterns is None:
        patterns = [
            "src/**/*.py",
            "scripts/**/*.py", 
            "utils/**/*.py",
            "Dockerfile",
            "*.yaml",
            "*.yml"
        ]
    
    files = []
    repo_root = Path(__file__).resolve().parents[1]
    
    for pattern in patterns:
        files.extend(repo_root.glob(pattern))
    
    # Sort for consistent ordering
    files = sorted(set(files))
    
    hasher = hashlib.sha256()
    for file in files:
        if file.is_file():
            hasher.update(file.read_bytes())
    
    return hasher.hexdigest()

def create_lock(tag, dry_run=False):
    """Create a release lock."""
    git_info = get_git_info()
    
    if git_info["has_uncommitted_changes"]:
        print("Error: Uncommitted changes detected. Commit all changes before creating a release lock.", file=sys.stderr)
        sys.exit(1)
    
    lock_data = {
        "version": tag,
        "timestamp": datetime.now().isoformat(),
        "git": git_info,
        "checksum": calculate_checksum(),
        "locked": True
    }
    
    if dry_run:
        print("DRY RUN - Would create lock:")
        print(json.dumps(lock_data, indent=2))
        return
    
    LOCK_FILE.write_text(json.dumps(lock_data, indent=2))
    print(f"Release lock created for version {tag}")
    print(f"Lock file: {LOCK_FILE}")

def verify_lock():
    """Verify the release hasn't been modified."""
    if not LOCK_FILE.exists():
        print("No release lock found.")
        return True
    
    lock_data = json.loads(LOCK_FILE.read_text())
    
    if not lock_data.get("locked", False):
        print("Release is not locked.")
        return True
    
    current_checksum = calculate_checksum()
    locked_checksum = lock_data.get("checksum", "")
    
    if current_checksum != locked_checksum:
        print(f"WARNING: Files have been modified since release {lock_data['version']}", file=sys.stderr)
        print(f"Release date: {lock_data['timestamp']}", file=sys.stderr)
        print(f"Expected checksum: {locked_checksum}", file=sys.stderr)
        print(f"Current checksum:  {current_checksum}", file=sys.stderr)
        return False
    
    print(f"Release lock verified for version {lock_data['version']}")
    return True

def unlock():
    """Remove release lock."""
    if not LOCK_FILE.exists():
        print("No release lock to remove.")
        return
    
    lock_data = json.loads(LOCK_FILE.read_text())
    lock_data["locked"] = False
    LOCK_FILE.write_text(json.dumps(lock_data, indent=2))
    print("Release lock removed.")

def main():
    parser = argparse.ArgumentParser(description="Release lock management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create lock
    create_parser = subparsers.add_parser("create", help="Create a release lock")
    create_parser.add_argument("tag", help="Release tag/version")
    create_parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    
    # Verify lock
    verify_parser = subparsers.add_parser("verify", help="Verify release integrity")
    
    # Unlock
    unlock_parser = subparsers.add_parser("unlock", help="Remove release lock")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_lock(args.tag, args.dry_run)
    elif args.command == "verify":
        if not verify_lock():
            sys.exit(1)
    elif args.command == "unlock":
        unlock()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()