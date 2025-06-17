#!/usr/bin/env python3
"""
test_voice_polisher.py - Tests for narrative voice conversion

Tests conversion from passive to active voice using we/I pronouns.
"""

import pytest
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestVoicePolisher:
    """Test narrative voice conversion to active"""
    
    def test_passive_voice_detection(self):
        """Test detection of passive voice constructions"""
        from voice_polisher import VoicePolisher
        
        polisher = VoicePolisher()
        
        # Test passive voice examples
        passive_text = "Data were analyzed using Python."
        assert polisher.has_passive_voice(passive_text)
        
        active_text = "We analyzed data using Python."
        assert not polisher.has_passive_voice(active_text)
    
    def test_voice_conversion(self):
        """Test conversion from passive to active voice"""
        from voice_polisher import VoicePolisher
        
        polisher = VoicePolisher()
        
        # Test conversions
        conversions = [
            ("Data were analyzed", "We analyzed data"),
            ("The cohort was selected", "We selected the cohort"),
            ("Results are presented", "We present results"),
            ("This study will evaluate", "We will evaluate"),
            ("The team decided", "We decided")
        ]
        
        for passive, expected_active in conversions:
            result = polisher.convert_to_active(passive)
            assert expected_active.lower() in result.lower()
    
    def test_third_person_conversion(self, tmp_path):
        """Test conversion of third person to first person"""
        from voice_polisher import VoicePolisher
        
        polisher = VoicePolisher()
        
        test_file = tmp_path / "test.md"
        test_file.write_text("""
        # Methods
        This study examines the effect of SSD.
        The researchers collected data from CPCSSN.
        The analysis includes propensity scores.
        """)
        
        polisher.process_file(test_file)
        
        content = test_file.read_text()
        assert "We examine" in content
        assert "We collected" in content
        assert "We include" in content or "Our analysis includes" in content
    
    def test_preserve_code_blocks(self, tmp_path):
        """Test that code blocks are not modified"""
        from voice_polisher import VoicePolisher
        
        polisher = VoicePolisher()
        
        test_file = tmp_path / "test.md"
        test_file.write_text("""
        Data were analyzed using:
        ```python
        # This code was written
        data = pd.read_csv('file.csv')
        results were calculated
        ```
        Results were significant.
        """)
        
        polisher.process_file(test_file)
        
        content = test_file.read_text()
        # Check text outside code is converted
        assert "We analyzed" in content
        assert "We found significant results" in content or "Results were significant" not in content
        # Check code block is preserved
        assert "data = pd.read_csv('file.csv')" in content
    
    def test_batch_processing(self, tmp_path):
        """Test processing multiple files"""
        from voice_polisher import VoicePolisher
        
        polisher = VoicePolisher()
        
        # Create test files
        for i in range(3):
            test_file = tmp_path / f"doc{i}.md"
            test_file.write_text(f"Document {i}: Data were analyzed.")
        
        # Process all files
        results = polisher.process_directory(tmp_path)
        
        assert len(results['processed']) == 3
        assert results['total_changes'] > 0
        
        # Check files were modified
        for i in range(3):
            content = (tmp_path / f"doc{i}.md").read_text()
            assert "We analyzed" in content
    
    def test_backup_creation(self, tmp_path):
        """Test that backups are created before modification"""
        from voice_polisher import VoicePolisher
        
        polisher = VoicePolisher(create_backups=True)
        
        test_file = tmp_path / "test.md"
        original_content = "Data were analyzed."
        test_file.write_text(original_content)
        
        polisher.process_file(test_file)
        
        # Check backup exists
        backup_files = list(tmp_path.glob("*.backup"))
        assert len(backup_files) == 1
        assert backup_files[0].read_text() == original_content
    
    def test_summary_report(self, tmp_path):
        """Test generation of conversion summary"""
        from voice_polisher import VoicePolisher
        
        polisher = VoicePolisher()
        
        test_file = tmp_path / "test.md"
        test_file.write_text("""
        Data were analyzed.
        Results are presented.
        The study shows significance.
        """)
        
        summary = polisher.process_file(test_file, return_summary=True)
        
        assert summary['file'] == str(test_file)
        assert summary['changes'] >= 2
        assert len(summary['conversions']) >= 2