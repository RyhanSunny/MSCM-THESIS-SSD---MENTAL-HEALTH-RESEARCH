#!/usr/bin/env python3
"""
Comprehensive Pipeline Audit Script
This script audits all phases of the SSD pipeline for completeness,
identifies gaps, placeholders, and ensures alignment with blueprint.
"""

import os
import re
import ast
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class PipelineAuditor:
    def __init__(self):
        self.base_path = Path(".")
        self.src_path = self.base_path / "src"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "gaps": [],
            "placeholders": [],
            "false_assumptions": [],
            "documentation_issues": [],
            "hypothesis_coverage": {}
        }
        
    def audit_script(self, filepath: Path) -> Dict:
        """Audit a single Python script"""
        audit_result = {
            "exists": filepath.exists(),
            "has_todos": False,
            "has_placeholders": False,
            "has_dummy_data": False,
            "has_hypothesis_mapping": False,
            "has_proper_imports": False,
            "has_config_integration": False,
            "has_error_handling": False,
            "has_documentation": False,
            "issues": []
        }
        
        if not filepath.exists():
            audit_result["issues"].append("File does not exist")
            return audit_result
            
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Check for TODOs and placeholders
        if re.search(r'TODO|FIXME|XXX', content, re.IGNORECASE):
            audit_result["has_todos"] = True
            audit_result["issues"].append("Contains TODO/FIXME comments")
            
        if re.search(r'placeholder|dummy|fake|mock', content, re.IGNORECASE):
            audit_result["has_placeholders"] = True
            audit_result["issues"].append("Contains placeholder code")
            
        # Check for dummy data or hardcoded values
        if re.search(r'np\.random\.random|dummy_data|test_data|example', content):
            audit_result["has_dummy_data"] = True
            audit_result["issues"].append("May contain dummy/test data")
            
        # Check for hypothesis mapping
        if re.search(r'[Hh]ypothesis.*H[1-6]|hypothesis_support', content):
            audit_result["has_hypothesis_mapping"] = True
            
        # Check for proper imports
        if re.search(r'from utils\.global_seeds|from src\.config_loader', content):
            audit_result["has_proper_imports"] = True
            
        # Check for config integration
        if re.search(r'load_config\(\)|get_config\(', content):
            audit_result["has_config_integration"] = True
        else:
            audit_result["issues"].append("Missing config integration")
            
        # Check for error handling
        if re.search(r'try:|except:|raise|logger\.error', content):
            audit_result["has_error_handling"] = True
        else:
            audit_result["issues"].append("Limited error handling")
            
        # Check for documentation
        if re.search(r'"""[\s\S]+?"""', content):
            audit_result["has_documentation"] = True
        else:
            audit_result["issues"].append("Missing docstring")
            
        # Check for specific implementation issues
        self._check_specific_issues(filepath, content, audit_result)
        
        return audit_result
        
    def _check_specific_issues(self, filepath: Path, content: str, audit_result: Dict):
        """Check for script-specific implementation issues"""
        filename = filepath.name
        
        # Check temporal consistency (should use 2018-2020)
        if re.search(r'2015-\d{2}-\d{2}|"2015|\'2015', content):
            audit_result["issues"].append("Contains 2015 dates (should be 2018-2020)")
            
        # Script-specific checks
        if filename == "01_cohort_builder.py":
            if not re.search(r'REF_DATE.*2018', content):
                audit_result["issues"].append("REF_DATE should be 2018-01-01")
            if not re.search(r'Long.?COVID|U07\.1', content, re.IGNORECASE):
                audit_result["issues"].append("Missing Long-COVID flag implementation")
                
        elif filename == "02_exposure_flag.py":
            if not re.search(r'drug_atc\.csv', content):
                audit_result["issues"].append("Missing drug ATC code integration")
            if not re.search(r'is_normal_lab', content):
                audit_result["issues"].append("Missing lab normality check")
                
        elif filename == "03_mediator_autoencoder.py":
            if not re.search(r'sparse.*regularizer|activity_regularizer', content):
                audit_result["issues"].append("Missing sparse regularization")
            if not re.search(r'56.*features|feature_cols.*56', content):
                audit_result["issues"].append("Should use 56 features as specified")
                
        elif filename == "05_ps_match.py":
            if not re.search(r'gpu_hist|tree_method.*gpu', content):
                audit_result["issues"].append("Missing GPU acceleration check")
            if not re.search(r'love.*plot|Love.*[Pp]lot', content):
                audit_result["issues"].append("Missing Love plot generation")
                
        elif filename == "06_causal_estimators.py":
            if not re.search(r'TMLE|tmle', content):
                audit_result["issues"].append("Missing TMLE implementation")
            if not re.search(r'[Dd]ouble.*ML|DML', content):
                audit_result["issues"].append("Missing Double ML implementation")
                
    def audit_phase(self, phase_name: str, expected_scripts: List[str]) -> Dict:
        """Audit a complete phase"""
        phase_result = {
            "complete": True,
            "scripts": {},
            "missing_scripts": [],
            "issues": []
        }
        
        for script in expected_scripts:
            script_path = self.src_path / script
            audit_result = self.audit_script(script_path)
            phase_result["scripts"][script] = audit_result
            
            if not audit_result["exists"]:
                phase_result["missing_scripts"].append(script)
                phase_result["complete"] = False
            elif audit_result["issues"]:
                phase_result["complete"] = False
                
        return phase_result
        
    def check_hypothesis_coverage(self):
        """Check if all hypotheses are properly covered"""
        hypothesis_map = {
            "H1": ["01_cohort_builder.py", "02_exposure_flag.py", "06_lab_flag.py"],
            "H2": ["07_referral_sequence.py"],
            "H3": ["02_exposure_flag.py", "04_outcome_flag.py"],
            "H4": ["03_mediator_autoencoder.py"],
            "H5": ["06_causal_estimators.py"],
            "H6": ["06_causal_estimators.py"]
        }
        
        for hyp, scripts in hypothesis_map.items():
            covered = True
            coverage_details = []
            
            for script in scripts:
                script_path = self.src_path / script
                if script_path.exists():
                    with open(script_path, 'r') as f:
                        content = f.read()
                    if hyp in content or f"Hypothesis.*{hyp}" in content:
                        coverage_details.append(f"{script}: Explicitly mentioned")
                    else:
                        coverage_details.append(f"{script}: Implicit coverage")
                else:
                    covered = False
                    coverage_details.append(f"{script}: MISSING")
                    
            self.results["hypothesis_coverage"][hyp] = {
                "covered": covered,
                "details": coverage_details
            }
            
    def check_documentation(self):
        """Check documentation completeness"""
        required_docs = [
            "README.md",
            "data_derived/cohort_report.md",
            "config/config.yaml",
            "requirements.txt",
            "Dockerfile",
            "Makefile"
        ]
        
        for doc in required_docs:
            doc_path = self.base_path / doc
            if not doc_path.exists():
                self.results["documentation_issues"].append(f"Missing: {doc}")
            else:
                # Check for completeness
                with open(doc_path, 'r') as f:
                    content = f.read()
                if len(content) < 100:  # Arbitrary threshold
                    self.results["documentation_issues"].append(f"Incomplete: {doc}")
                    
    def run_full_audit(self):
        """Run complete pipeline audit"""
        print("Starting comprehensive pipeline audit...")
        
        # Define phases and expected scripts
        phases = {
            "Phase_1_Infrastructure": [
                "config_loader.py", 
                "artefact_tracker.py",
                "helpers/lab_utils.py"
            ],
            "Phase_2_Data_Preparation": [
                "01_cohort_builder.py",
                "02_exposure_flag.py",
                "03_mediator_autoencoder.py",
                "04_outcome_flag.py",
                "05_confounder_flag.py",
                "06_lab_flag.py",
                "07_missing_data.py",
                "07_referral_sequence.py",
                "07a_misclassification_adjust.py",
                "08_patient_master_table.py"
            ],
            "Phase_3_Causal_Analysis": [
                "05_ps_match.py",
                "12_temporal_adjust.py",
                "06_causal_estimators.py"
            ],
            "Phase_4_Sensitivity": [
                "13_evalue_calc.py",
                "14_placebo_tests.py",
                "15_robustness.py",
                "finegray_competing.py",
                "death_rates_analysis.py"
            ]
        }
        
        # Audit each phase
        for phase_name, scripts in phases.items():
            print(f"\nAuditing {phase_name}...")
            self.results["phases"][phase_name] = self.audit_phase(phase_name, scripts)
            
        # Check hypothesis coverage
        print("\nChecking hypothesis coverage...")
        self.check_hypothesis_coverage()
        
        # Check documentation
        print("\nChecking documentation...")
        self.check_documentation()
        
        # Identify critical gaps
        self._identify_critical_gaps()
        
        # Save results
        output_path = Path("audit_results.json")
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Print summary
        self._print_summary()
        
    def _identify_critical_gaps(self):
        """Identify critical gaps in implementation"""
        # Check for false assumptions
        critical_checks = [
            ("Assuming 250,025 rows", "cohort size assumption"),
            ("baseline_rate = 10", "arbitrary baseline rate"),
            ("COST_PC_VISIT = 100", "cost proxy values"),
            ("sensitivity=0.82", "assumed misclassification rates")
        ]
        
        for pattern, issue in critical_checks:
            for phase_result in self.results["phases"].values():
                for script, audit in phase_result["scripts"].items():
                    if audit["exists"]:
                        script_path = self.src_path / script
                        with open(script_path, 'r') as f:
                            if pattern in f.read():
                                self.results["false_assumptions"].append({
                                    "script": script,
                                    "assumption": pattern,
                                    "issue": issue
                                })
                                
    def _print_summary(self):
        """Print audit summary"""
        print("\n" + "="*60)
        print("PIPELINE AUDIT SUMMARY")
        print("="*60)
        
        # Phase summary
        for phase, result in self.results["phases"].items():
            status = "✅ COMPLETE" if result["complete"] else "❌ INCOMPLETE"
            print(f"\n{phase}: {status}")
            
            if result["missing_scripts"]:
                print(f"  Missing scripts: {', '.join(result['missing_scripts'])}")
                
            # Count issues
            total_issues = sum(len(s["issues"]) for s in result["scripts"].values())
            if total_issues > 0:
                print(f"  Total issues: {total_issues}")
                
        # Hypothesis coverage
        print("\nHypothesis Coverage:")
        for hyp, coverage in self.results["hypothesis_coverage"].items():
            status = "✅" if coverage["covered"] else "❌"
            print(f"  {hyp}: {status}")
            
        # Critical issues
        if self.results["false_assumptions"]:
            print(f"\n⚠️  False Assumptions: {len(self.results['false_assumptions'])}")
            
        if self.results["documentation_issues"]:
            print(f"\n⚠️  Documentation Issues: {len(self.results['documentation_issues'])}")
            
        print("\nAudit complete. Results saved to audit_results.json")

if __name__ == "__main__":
    auditor = PipelineAuditor()
    auditor.run_full_audit()