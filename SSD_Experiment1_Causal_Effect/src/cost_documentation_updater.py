"""
Cost Documentation Updater: Updates documentation to reflect proxy costs and SES limitations.

This module identifies documentation that needs updates to clarify that cost estimates
are proxies and that SES data is not available, adding appropriate disclaimers.

Author: Ryhan Suny
Date: 2025-06-21
Version: 1.0.0
"""

import re
from pathlib import Path
import logging
import argparse
from typing import List, Union, Optional
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_cost_proxy_documentation(
    doc_files: List[Union[str, Path]],
    backup: bool = True
) -> None:
    """
    Update documentation to clarify that cost estimates are proxies.
    
    Parameters:
    -----------
    doc_files : List[Union[str, Path]]
        List of documentation files to update
    backup : bool, default=True
        Whether to create backup before modification
        
    Raises:
    -------
    FileNotFoundError
        If any doc_file doesn't exist
    """
    logger.info("Updating documentation to clarify proxy cost estimates...")
    
    # Cost-related patterns to update (context-aware)
    cost_patterns = [
        (r'\bhealthcare utilization costs\b', 'healthcare utilization costs (proxy estimates based on encounter counts)'),
        (r'\bmedical costs\b', 'medical costs (proxy estimates)'),
        (r'\bcost[- ]effectiveness\b', 'cost-effectiveness (based on proxy cost estimates)'),
        (r'\btotal healthcare costs\b', 'total healthcare costs (proxy estimates)'),
        (r'\bCost[- ](\w+)\b', r'Cost \1 (proxy estimates)'),
        (r'\bCOST_([A-Z_]+)\b', r'COST_\1  # Proxy estimate'),
    ]
    
    proxy_disclaimer = """
**Note on Cost Estimates**: All cost estimates in this analysis are proxy calculations
based on encounter counts and average cost assumptions. These are not actual billing
amounts and should be interpreted as relative cost indicators for comparative analysis.
"""
    
    for doc_file in doc_files:
        doc_path = Path(doc_file)
        
        if not doc_path.exists():
            raise FileNotFoundError(f"Documentation file not found: {doc_path}")
        
        logger.info(f"Processing cost disclaimers for {doc_path}")
        
        # Create backup if requested
        if backup:
            backup_path = doc_path.with_suffix(doc_path.suffix + '.backup')
            logger.info(f"Creating backup at {backup_path}")
            shutil.copy2(doc_path, backup_path)
        
        # Read file content
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = 0
        
        # Apply cost pattern updates
        for pattern, replacement in cost_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                changes_made += 1
        
        # Add proxy disclaimer if cost-related content found
        cost_keywords = ['cost', 'Cost', 'COST', 'economic', 'Economic']
        if any(keyword in content for keyword in cost_keywords):
            # Check if disclaimer already exists
            if 'proxy calculations' not in content and 'proxy estimates' not in content:
                # Find a good place to insert disclaimer (after first heading)
                lines = content.split('\n')
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.startswith('#') and i > 0:  # Found first heading after start
                        insert_index = i + 1
                        break
                
                if insert_index > 0:
                    lines.insert(insert_index, proxy_disclaimer)
                    content = '\n'.join(lines)
                    changes_made += 1
        
        # Write updated content
        if content != original_content:
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✓ Updated {changes_made} cost references in {doc_path}")
        else:
            logger.info(f"No cost updates needed for {doc_path}")


def update_ses_limitation_documentation(
    doc_files: List[Union[str, Path]],
    backup: bool = True
) -> None:
    """
    Update documentation to clarify SES data limitations.
    
    Parameters:
    -----------
    doc_files : List[Union[str, Path]]
        List of documentation files to update
    backup : bool, default=True
        Whether to create backup before modification
        
    Raises:
    -------
    FileNotFoundError
        If any doc_file doesn't exist
    """
    logger.info("Updating documentation to clarify SES data limitations...")
    
    # SES-related patterns to update (context-aware to avoid corrupting words)
    ses_patterns = [
        (r'\bsocioeconomic status\b', 'socioeconomic status (data not available for this analysis)'),
        (r'\bsocioeconomic indicators\b', 'socioeconomic indicators (not available)'),
        (r'\bdeprivation\b', 'deprivation (data unavailable)'),
        (r'\bincome quintile\b', 'income quintile (data not collected)'),
        (r'\bSES\b', 'SES (socioeconomic status data not available)'),  # Word boundary to avoid partial matches
    ]
    
    ses_limitation_note = """
**Note on Socioeconomic Status (SES) Variables**: SES data including income quintiles,
deprivation indices, and detailed socioeconomic indicators are not available in this
dataset. Analyses that reference SES variables use available proxies or exclude these
variables. This represents a limitation of the current analysis.
"""
    
    for doc_file in doc_files:
        doc_path = Path(doc_file)
        
        if not doc_path.exists():
            raise FileNotFoundError(f"Documentation file not found: {doc_path}")
        
        logger.info(f"Processing SES limitations for {doc_path}")
        
        # Create backup if requested
        if backup:
            backup_path = doc_path.with_suffix(doc_path.suffix + '.backup')
            if not backup_path.exists():  # Don't overwrite existing backup
                logger.info(f"Creating backup at {backup_path}")
                shutil.copy2(doc_path, backup_path)
        
        # Read file content
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = 0
        
        # Apply SES pattern updates
        for pattern, replacement in ses_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                changes_made += 1
        
        # Add SES limitation note if SES-related content found
        ses_keywords = ['socioeconomic', 'Socioeconomic', 'SES', 'deprivation', 'income quintile']
        if any(keyword in content for keyword in ses_keywords):
            # Check if limitation note already exists
            if 'SES data' not in content and 'socioeconomic status data not available' not in content:
                # Find limitations section or create one
                if '## Limitations' in content or '# Limitations' in content:
                    # Add to existing limitations section
                    content = re.sub(
                        r'(##? Limitations.*?)(\n##|\n#|$)',
                        r'\1\n' + ses_limitation_note + r'\2',
                        content,
                        flags=re.DOTALL
                    )
                else:
                    # Add near end of document
                    content += '\n\n## Limitations\n' + ses_limitation_note
                changes_made += 1
        
        # Write updated content
        if content != original_content:
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✓ Updated {changes_made} SES references in {doc_path}")
        else:
            logger.info(f"No SES updates needed for {doc_path}")


def validate_documentation_updates(
    doc_files: List[Union[str, Path]]
) -> bool:
    """
    Validate that documentation updates have been applied.
    
    Parameters:
    -----------
    doc_files : List[Union[str, Path]]
        List of documentation files to validate
        
    Returns:
    --------
    bool
        True if all updates are present, False otherwise
    """
    logger.info("Validating documentation updates...")
    
    validation_passed = True
    
    for doc_file in doc_files:
        doc_path = Path(doc_file)
        
        if not doc_path.exists():
            logger.warning(f"Documentation file not found: {doc_path}")
            continue
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for cost-related content and proxy disclaimers
        has_cost_content = any(keyword in content.lower() for keyword in ['cost', 'economic'])
        has_cost_disclaimer = any(phrase in content.lower() for phrase in ['proxy estimate', 'proxy calculation'])
        
        if has_cost_content and not has_cost_disclaimer:
            logger.error(f"Cost content found without proxy disclaimer in {doc_path}")
            validation_passed = False
        elif has_cost_content and has_cost_disclaimer:
            logger.info(f"✓ Cost proxy disclaimer present in {doc_path}")
        
        # Check for SES-related content and limitation notes (use word boundaries)
        import re
        has_ses_content = bool(re.search(r'\b(socioeconomic|ses|deprivation)\b', content, re.IGNORECASE))
        has_ses_limitation = any(phrase in content.lower() for phrase in ['ses data', 'socioeconomic status data not available', 'data not available'])
        
        if has_ses_content and not has_ses_limitation:
            logger.error(f"SES content found without limitation note in {doc_path}")
            validation_passed = False
        elif has_ses_content and has_ses_limitation:
            logger.info(f"✓ SES limitation note present in {doc_path}")
    
    if validation_passed:
        logger.info("✓ All documentation updates validated successfully")
    else:
        logger.error("❌ Some documentation updates are missing")
    
    return validation_passed


def find_documentation_files(
    root_dir: Union[str, Path] = "."
) -> List[Path]:
    """
    Find all documentation files that may need updates.
    
    Parameters:
    -----------
    root_dir : Union[str, Path], default="."
        Root directory to search
        
    Returns:
    --------
    List[Path]
        List of documentation files found
    """
    root_dir = Path(root_dir)
    doc_files = []
    
    # Look for markdown files
    for md_file in root_dir.glob("**/*.md"):
        # Skip backup files and certain directories
        if (md_file.name.endswith('.backup') or 
            'node_modules' in str(md_file) or
            '.git' in str(md_file)):
            continue
        doc_files.append(md_file)
    
    return doc_files


def main():
    """Command-line interface for cost documentation updater."""
    parser = argparse.ArgumentParser(
        description="Update documentation to reflect proxy costs and SES limitations"
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='Specific documentation files to update'
    )
    parser.add_argument(
        '--root-dir',
        default='.',
        help='Root directory to search for documentation files'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate that updates have been applied'
    )
    
    args = parser.parse_args()
    
    try:
        # Find documentation files
        if args.files:
            doc_files = [Path(f) for f in args.files]
        else:
            doc_files = find_documentation_files(args.root_dir)
        
        logger.info(f"Found {len(doc_files)} documentation files")
        
        if args.validate_only:
            # Validate only
            is_valid = validate_documentation_updates(doc_files)
            return 0 if is_valid else 1
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No files will be modified")
            
            for doc_file in doc_files:
                if doc_file.exists():
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    has_cost = any(keyword in content.lower() for keyword in ['cost', 'economic'])
                    has_ses = any(keyword in content.lower() for keyword in ['socioeconomic', 'ses'])
                    
                    if has_cost or has_ses:
                        logger.info(f"Would update {doc_file}: cost={has_cost}, ses={has_ses}")
            
            return 0
        
        # Apply updates
        logger.info("Updating documentation for proxy costs and SES limitations...")
        
        update_cost_proxy_documentation(doc_files, backup=not args.no_backup)
        update_ses_limitation_documentation(doc_files, backup=not args.no_backup)
        
        # Validate updates
        is_valid = validate_documentation_updates(doc_files)
        
        if is_valid:
            logger.info("✓ Successfully updated all documentation")
            return 0
        else:
            logger.error("❌ Some documentation updates may be incomplete")
            return 1
            
    except Exception as e:
        logger.error(f"Error during documentation update: {e}")
        return 1


if __name__ == "__main__":
    exit(main())