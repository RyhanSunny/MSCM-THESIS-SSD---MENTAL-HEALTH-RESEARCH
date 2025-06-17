#!/usr/bin/env python3
"""
voice_polisher.py - Convert passive voice to active voice in documentation

Converts academic writing from passive/third-person to active first-person (we/I)
for clearer, more direct communication.
"""

import re
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoicePolisher:
    """Convert passive voice to active voice in documents"""
    
    def __init__(self, create_backups: bool = True):
        self.create_backups = create_backups
        self.conversion_rules = self._build_conversion_rules()
        
    def _build_conversion_rules(self) -> List[Tuple[re.Pattern, str]]:
        """Build regex patterns for voice conversion"""
        rules = [
            # Passive voice patterns
            (r'\b(D|d)ata (were|was) analyzed\b', r'We analyzed data'),
            (r'\b(R|r)esults (were|are) presented\b', r'We present results'),
            (r'\b(T|t)he cohort (was|were) selected\b', r'We selected the cohort'),
            (r'\b(A|a)nalyses (were|are) conducted\b', r'We conducted analyses'),
            (r'\b(M|m)ethods (were|are) implemented\b', r'We implemented methods'),
            (r'\b(V|v)ariables (were|are) measured\b', r'We measured variables'),
            (r'\b(M|m)odels (were|are) fitted\b', r'We fitted models'),
            (r'\b(T|t)ests (were|are) performed\b', r'We performed tests'),
            (r'\b(E|e)stimates (were|are) calculated\b', r'We calculated estimates'),
            (r'\b(C|c)onfounders (were|are) adjusted\b', r'We adjusted for confounders'),
            
            # Third person to first person
            (r'\bThis study (will )?examine[sd]?\b', r'We \1examine'),
            (r'\bThe study (will )?investigate[sd]?\b', r'We \1investigate'),
            (r'\bThe researchers? (will )?collect(ed)?\b', r'We \1collect\2'),
            (r'\bThe team (will )?decide[sd]?\b', r'We \1decide'),
            (r'\bThe analysis (will )?include[sd]?\b', r'Our analysis \1includes'),
            (r'\bThe pipeline (will )?process(es|ed)?\b', r'Our pipeline \1processes'),
            (r'\bThe approach (will )?use[sd]?\b', r'Our approach \1uses'),
            
            # Common passive constructions
            (r'\b(is|are|was|were) (being )?analyzed\b', r'we analyze'),
            (r'\b(is|are|was|were) (being )?calculated\b', r'we calculate'),
            (r'\b(is|are|was|were) (being )?determined\b', r'we determine'),
            (r'\b(is|are|was|were) (being )?evaluated\b', r'we evaluate'),
            (r'\b(is|are|was|were) (being )?examined\b', r'we examine'),
            (r'\b(is|are|was|were) (being )?investigated\b', r'we investigate'),
            (r'\b(is|are|was|were) (being )?measured\b', r'we measure'),
            (r'\b(is|are|was|were) (being )?observed\b', r'we observe'),
            (r'\b(is|are|was|were) (being )?performed\b', r'we perform'),
            (r'\b(is|are|was|were) (being )?tested\b', r'we test'),
            
            # It is/was patterns
            (r'\bIt (is|was) found that\b', r'We found that'),
            (r'\bIt (is|was) observed that\b', r'We observed that'),
            (r'\bIt (is|was) determined that\b', r'We determined that'),
            (r'\bIt (is|was) concluded that\b', r'We concluded that'),
            
            # Can be patterns
            (r'\bcan be seen\b', r'we can see'),
            (r'\bcan be observed\b', r'we can observe'),
            (r'\bcan be determined\b', r'we can determine'),
            (r'\bcan be concluded\b', r'we can conclude'),
            
            # Has/have been patterns
            (r'\b(has|have) been analyzed\b', r'we analyzed'),
            (r'\b(has|have) been calculated\b', r'we calculated'),
            (r'\b(has|have) been determined\b', r'we determined'),
            (r'\b(has|have) been evaluated\b', r'we evaluated'),
            
            # Will be patterns
            (r'\bwill be analyzed\b', r'we will analyze'),
            (r'\bwill be calculated\b', r'we will calculate'),
            (r'\bwill be determined\b', r'we will determine'),
            (r'\bwill be evaluated\b', r'we will evaluate'),
        ]
        
        return [(re.compile(pattern, re.IGNORECASE), replacement) 
                for pattern, replacement in rules]
    
    def has_passive_voice(self, text: str) -> bool:
        """Check if text contains passive voice"""
        passive_indicators = [
            r'\b(is|are|was|were|been|being)\s+\w+ed\b',
            r'\b(is|are|was|were|been|being)\s+\w+en\b',
            r'\bby\s+the\s+\w+',
            r'\bit\s+(is|was)\s+\w+ed\s+that\b'
        ]
        
        for pattern in passive_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def convert_to_active(self, text: str) -> str:
        """Convert passive voice to active voice"""
        result = text
        
        for pattern, replacement in self.conversion_rules:
            result = pattern.sub(replacement, result)
        
        # Fix capitalization at sentence starts
        result = re.sub(r'(?<=[.!?]\s)we\b', 'We', result)
        result = re.sub(r'^we\b', 'We', result, flags=re.MULTILINE)
        
        return result
    
    def process_file(self, file_path: Path, return_summary: bool = False) -> Dict[str, Any]:
        """Process a single file to convert voice"""
        logger.info(f"Processing {file_path}...")
        
        file_path = Path(file_path)
        
        # Read content
        original_content = file_path.read_text(encoding='utf-8')
        
        # Create backup if requested
        if self.create_backups:
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
        
        # Process content
        processed_content = self._process_content(original_content)
        
        # Count changes
        changes = self._count_changes(original_content, processed_content)
        
        # Write back
        if changes > 0:
            file_path.write_text(processed_content, encoding='utf-8')
            logger.info(f"Updated {file_path} with {changes} changes")
        else:
            logger.info(f"No changes needed in {file_path}")
        
        if return_summary:
            return {
                'file': str(file_path),
                'changes': changes,
                'conversions': self._get_conversions(original_content, processed_content)
            }
        
        return {'file': str(file_path), 'changes': changes}
    
    def _process_content(self, content: str) -> str:
        """Process content preserving code blocks"""
        # Split into code and non-code sections
        code_blocks = []
        
        # Extract code blocks
        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"
        
        # Remove code blocks temporarily
        content_no_code = re.sub(
            r'```[\s\S]*?```|`[^`]+`',
            save_code_block,
            content
        )
        
        # Convert voice in non-code content
        converted = self.convert_to_active(content_no_code)
        
        # Restore code blocks
        for i, block in enumerate(code_blocks):
            converted = converted.replace(f"__CODE_BLOCK_{i}__", block)
        
        return converted
    
    def _count_changes(self, original: str, processed: str) -> int:
        """Count number of changes made"""
        # Simple approximation: count different lines
        orig_lines = original.splitlines()
        proc_lines = processed.splitlines()
        
        changes = 0
        for orig, proc in zip(orig_lines, proc_lines):
            if orig != proc:
                changes += 1
        
        return changes
    
    def _get_conversions(self, original: str, processed: str) -> List[Dict[str, str]]:
        """Get list of specific conversions made"""
        conversions = []
        
        orig_lines = original.splitlines()
        proc_lines = processed.splitlines()
        
        for i, (orig, proc) in enumerate(zip(orig_lines, proc_lines)):
            if orig != proc and orig.strip() and proc.strip():
                conversions.append({
                    'line': i + 1,
                    'original': orig.strip(),
                    'converted': proc.strip()
                })
        
        return conversions[:10]  # Return first 10 conversions
    
    def process_directory(self, directory: Path, 
                         extensions: List[str] = None) -> Dict[str, Any]:
        """Process all files in directory"""
        logger.info(f"Processing directory {directory}...")
        
        directory = Path(directory)
        if extensions is None:
            extensions = ['.md', '.tex', '.txt']
        
        results = {
            'processed': [],
            'skipped': [],
            'total_changes': 0,
            'errors': []
        }
        
        for ext in extensions:
            for file_path in directory.rglob(f'*{ext}'):
                try:
                    # Skip backup files
                    if '.backup' in str(file_path):
                        results['skipped'].append(str(file_path))
                        continue
                    
                    # Process file
                    file_result = self.process_file(file_path)
                    results['processed'].append(file_result)
                    results['total_changes'] += file_result['changes']
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results['errors'].append({
                        'file': str(file_path),
                        'error': str(e)
                    })
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate summary report of conversions"""
        report = f"""
# Voice Conversion Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- Files processed: {len(results['processed'])}
- Files skipped: {len(results['skipped'])}
- Total changes: {results['total_changes']}
- Errors: {len(results['errors'])}

## Files Modified

"""
        
        for file_result in results['processed']:
            if file_result['changes'] > 0:
                report += f"- {file_result['file']}: {file_result['changes']} changes\n"
        
        if results['errors']:
            report += "\n## Errors\n\n"
            for error in results['errors']:
                report += f"- {error['file']}: {error['error']}\n"
        
        return report


def main():
    """Polish voice in key documents"""
    polisher = VoicePolisher(create_backups=True)
    
    # Process key documents
    key_files = [
        "SSD THESIS final METHODOLOGIES blueprint (1).md",
        "docs/Methods_Supplement.md",
        "reports/week2_analysis_report.md"
    ]
    
    total_changes = 0
    
    print("\n=== Voice Polishing ===")
    print("Converting passive voice to active (we/I)...\n")
    
    for file_name in key_files:
        file_path = Path(file_name)
        if file_path.exists():
            result = polisher.process_file(file_path, return_summary=True)
            print(f"✓ {file_name}: {result['changes']} changes")
            total_changes += result['changes']
            
            # Show example conversions
            if result['conversions']:
                print("  Example conversions:")
                for conv in result['conversions'][:3]:
                    print(f"    Line {conv['line']}: '{conv['original'][:50]}...' → '{conv['converted'][:50]}...'")
        else:
            print(f"✗ {file_name}: Not found")
    
    print(f"\nTotal changes: {total_changes}")
    print("\nVoice polishing complete!")


if __name__ == "__main__":
    main()