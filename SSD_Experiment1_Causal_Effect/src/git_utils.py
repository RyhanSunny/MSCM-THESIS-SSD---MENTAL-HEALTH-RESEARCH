#!/usr/bin/env python3
"""
git_utils.py - Utilities for adding git metadata to output files

Following reviewer feedback: "Add git SHA and modification date to YAML study docs"

Author: Ryhan Suny
Date: June 30, 2025
Version: 1.0.0
"""

import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_git_sha(short: bool = True) -> str:
    """
    Get the git SHA of the current commit.
    
    Parameters:
    -----------
    short : bool, default True
        If True, return short SHA (7 chars), else full SHA
        
    Returns:
    --------
    str
        Git SHA or "unknown" if git not available
    """
    try:
        cmd = ['git', 'rev-parse']
        if short:
            cmd.append('--short')
        cmd.append('HEAD')
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Could not get git SHA: {e}")
        return "unknown"


def get_git_branch() -> str:
    """Get the current git branch name."""
    try:
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def add_git_metadata(data: Dict[str, Any], 
                    metadata_key: str = "_metadata") -> Dict[str, Any]:
    """
    Add git metadata to a dictionary (typically before saving to YAML/JSON).
    
    Parameters:
    -----------
    data : Dict[str, Any]
        Dictionary to add metadata to
    metadata_key : str, default "_metadata"
        Key under which to store metadata
        
    Returns:
    --------
    Dict[str, Any]
        Updated dictionary with git metadata
    """
    if metadata_key not in data:
        data[metadata_key] = {}
    
    data[metadata_key].update({
        "git_sha": get_git_sha(short=True),
        "git_sha_full": get_git_sha(short=False),
        "git_branch": get_git_branch(),
        "timestamp": datetime.now().isoformat(),
        "modification_date": datetime.now().strftime("%Y-%m-%d"),
        "modification_time": datetime.now().strftime("%H:%M:%S")
    })
    
    return data


def format_filename_with_timestamp(base_name: str, 
                                 extension: str = ".yaml",
                                 include_git_sha: bool = False) -> str:
    """
    Format a filename with timestamp and optionally git SHA.
    
    Parameters:
    -----------
    base_name : str
        Base filename without extension
    extension : str
        File extension (including dot)
    include_git_sha : bool
        Whether to include git SHA in filename
        
    Returns:
    --------
    str
        Formatted filename
        
    Example:
    --------
    >>> format_filename_with_timestamp("results", ".json", True)
    "results_20250630_143022_a1b2c3d.json"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if include_git_sha:
        git_sha = get_git_sha(short=True)
        if git_sha != "unknown":
            return f"{base_name}_{timestamp}_{git_sha}{extension}"
    
    return f"{base_name}_{timestamp}{extension}"


def log_git_info():
    """Log current git information for debugging."""
    logger.info(f"Git SHA: {get_git_sha()}")
    logger.info(f"Git branch: {get_git_branch()}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    # Test the utilities
    print(f"Git SHA (short): {get_git_sha(short=True)}")
    print(f"Git SHA (full): {get_git_sha(short=False)}")
    print(f"Git branch: {get_git_branch()}")
    
    test_data = {"test": "value"}
    updated_data = add_git_metadata(test_data)
    print(f"\nUpdated data with metadata:")
    import json
    print(json.dumps(updated_data, indent=2, default=str))