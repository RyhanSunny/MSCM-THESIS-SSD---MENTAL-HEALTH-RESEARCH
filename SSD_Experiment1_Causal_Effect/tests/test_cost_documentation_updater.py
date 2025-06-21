"""
Test for updating documentation to reflect proxy cost data and SES limitations.

Following TDD principles as per CLAUDE.md requirements.
"""

import pytest
from pathlib import Path
import sys
import tempfile
import shutil
import re

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cost_documentation_updater import (
    update_cost_proxy_documentation,
    update_ses_limitation_documentation,
    validate_documentation_updates
)


class TestCostDocumentationUpdater:
    """Test documentation updates for proxy costs and SES limitations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create mock blueprint file
        self.blueprint_path = self.test_dir / "blueprint.md"
        with open(self.blueprint_path, 'w') as f:
            f.write("""
# Test Blueprint

## H4 — MH SSD Severity Mediation
healthcare utilization costs at 24 months

## Cost Analysis
Medical costs are calculated based on encounter data.

## Confounders
We adjust for socioeconomic status indicators.
""")
        
        # Create mock source file with cost calculations
        self.source_path = self.test_dir / "outcome_flag.py"
        with open(self.source_path, 'w') as f:
            f.write("""
# Cost proxies from config (in CAD)
COST_PC_VISIT = 100  # Primary care visit
COST_ED_VISIT = 500  # Emergency department visit

# Calculate proxy costs
utilization["medical_costs"] = (
    utilization["pc_visits"] * COST_PC_VISIT +
    utilization["ed_visits"] * COST_ED_VISIT
)
""")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_update_cost_proxy_documentation_function_exists(self):
        """Test that update_cost_proxy_documentation function exists."""
        # This should fail initially (TDD)
        from cost_documentation_updater import update_cost_proxy_documentation
        assert callable(update_cost_proxy_documentation)
    
    def test_update_cost_proxy_documentation_adds_proxy_disclaimers(self):
        """Test that function adds proxy cost disclaimers to documentation."""
        # Check original content
        with open(self.blueprint_path, 'r') as f:
            content_before = f.read()
        assert "proxy cost" not in content_before.lower()
        
        # Update documentation
        update_cost_proxy_documentation([self.blueprint_path])
        
        # Check updated content
        with open(self.blueprint_path, 'r') as f:
            content_after = f.read()
        
        # Should contain proxy cost disclaimers
        assert ("proxy cost" in content_after.lower() or "proxy estimate" in content_after.lower())
        assert ("estimated" in content_after.lower() or 
                "approximate" in content_after.lower() or 
                "proxy" in content_after.lower())
    
    def test_update_ses_limitation_documentation_function_exists(self):
        """Test that update_ses_limitation_documentation function exists."""
        from cost_documentation_updater import update_ses_limitation_documentation
        assert callable(update_ses_limitation_documentation)
    
    def test_update_ses_limitation_documentation_adds_limitations(self):
        """Test that function adds SES limitation disclaimers."""
        # Check original content
        with open(self.blueprint_path, 'r') as f:
            content_before = f.read()
        assert "ses limitation" not in content_before.lower()
        
        # Update documentation
        update_ses_limitation_documentation([self.blueprint_path])
        
        # Check updated content
        with open(self.blueprint_path, 'r') as f:
            content_after = f.read()
        
        # Should contain SES limitation disclaimers
        assert "socioeconomic" in content_after.lower()
        assert ("limitation" in content_after.lower() or 
                "unavailable" in content_after.lower() or
                "not available" in content_after.lower())
    
    def test_validate_documentation_updates_function_exists(self):
        """Test that validate_documentation_updates function exists."""
        from cost_documentation_updater import validate_documentation_updates
        assert callable(validate_documentation_updates)
    
    def test_validate_documentation_updates_passes_when_complete(self):
        """Test validation passes when all updates are complete."""
        # First update documentation
        update_cost_proxy_documentation([self.blueprint_path])
        update_ses_limitation_documentation([self.blueprint_path])
        
        # Validation should pass
        result = validate_documentation_updates([self.blueprint_path])
        assert result == True
    
    def test_validate_documentation_updates_fails_when_incomplete(self):
        """Test validation fails when updates are missing."""
        # Don't update documentation - validation should fail
        result = validate_documentation_updates([self.blueprint_path])
        assert result == False
    
    def test_backup_created_before_modification(self):
        """Test that backup files are created before modification."""
        # Update documentation
        update_cost_proxy_documentation([self.blueprint_path], backup=True)
        
        # Check backup was created
        backup_path = self.blueprint_path.with_suffix('.md.backup')
        assert backup_path.exists()
        
        # Check backup contains original content
        with open(backup_path, 'r') as f:
            backup_content = f.read()
        assert "proxy cost" not in backup_content.lower()
    
    def test_preserves_original_structure(self):
        """Test that updates preserve original document structure."""
        # Count lines before
        with open(self.blueprint_path, 'r') as f:
            lines_before = len(f.readlines())
        
        # Update documentation
        update_cost_proxy_documentation([self.blueprint_path])
        update_ses_limitation_documentation([self.blueprint_path])
        
        # Should not drastically change document length
        with open(self.blueprint_path, 'r') as f:
            lines_after = len(f.readlines())
        
        # Allow for some increase but not massive changes
        assert lines_after >= lines_before
        assert lines_after <= lines_before * 2  # Max 100% increase


# Run the failing test to ensure TDD compliance
if __name__ == "__main__":
    test = TestCostDocumentationUpdater()
    test.setup_method()
    try:
        test.test_update_cost_proxy_documentation_function_exists()
        print("❌ TEST SHOULD FAIL (TDD requirement)")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"✓ Test fails as expected: {e}")
        print("Now implementing cost_documentation_updater module...")
    finally:
        test.teardown_method()