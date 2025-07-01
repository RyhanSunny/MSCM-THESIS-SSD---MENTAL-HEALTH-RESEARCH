#!/usr/bin/env python3
"""
target_trial_emulation.py - Document target trial emulation framework

Creates documentation for the hypothetical RCT that our observational
study aims to emulate, following best practices from Hernán & Robins.

Following CLAUDE.md requirements:
- Evidence-based implementation
- Clear documentation
- Version numbering and timestamps

Author: Ryhan Suny (Toronto Metropolitan University)  
Version: 1.0
Date: 2025-07-01
"""

import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_target_trial_protocol(output_dir: Path = Path("docs")) -> dict:
    """
    Create target trial emulation protocol documentation.
    
    Returns dict with trial protocol details.
    """
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().isoformat()
    
    # Define ideal RCT protocol
    target_trial = {
        "metadata": {
            "title": "Target Trial Protocol: SSD Patterns and Healthcare Utilization",
            "created": timestamp,
            "version": "1.0",
            "author": "Ryhan Suny, MSc"
        },
        
        "ideal_rct": {
            "title": "Randomized Controlled Trial of SSD Pattern Monitoring vs Usual Care",
            "design": "Parallel-group, open-label RCT",
            "duration": "24 months follow-up",
            
            "eligibility": {
                "inclusion": [
                    "Age ≥18 years",
                    "Active mental health diagnosis (anxiety, depression, or related)",
                    "≥30 months enrollment in primary care",
                    "Consent to participate"
                ],
                "exclusion": [
                    "Terminal illness",
                    "Severe cognitive impairment",
                    "Unable to provide informed consent",
                    "Planned relocation within 24 months"
                ]
            },
            
            "randomization": {
                "method": "1:1 block randomization stratified by:",
                "strata": [
                    "Age group (<40, 40-65, >65)",
                    "Sex (Male/Female)",
                    "Baseline utilization (High/Low)",
                    "Anxiety diagnosis (Yes/No)"
                ]
            },
            
            "interventions": {
                "treatment": {
                    "name": "SSD Pattern Alert System",
                    "components": [
                        "Real-time monitoring of lab results",
                        "Alert when ≥3 normal labs in 12 months",
                        "Flag unresolved referrals (≥2 NYD)",
                        "Monitor psychotropic persistence (>90 days)",
                        "Integrated care referral for high SSDSI"
                    ]
                },
                "control": {
                    "name": "Usual Care",
                    "description": "Standard mental health care without SSD monitoring"
                }
            },
            
            "outcomes": {
                "primary": [
                    "Total healthcare encounters at 24 months",
                    "Emergency department visits at 12 months"
                ],
                "secondary": [
                    "Mental health service utilization",
                    "Healthcare costs (estimated)",
                    "Patient satisfaction",
                    "Quality of life scores"
                ]
            },
            
            "sample_size": {
                "calculation": "Based on 80% power, α=0.05, IRR=1.4",
                "per_arm": 2500,
                "total": 5000,
                "attrition": "20% expected"
            }
        },
        
        "emulation_strategy": {
            "design": "New-user cohort design with active comparator",
            
            "eligibility_alignment": {
                "inclusion": "Applied same criteria to CPCSSN cohort",
                "exclusion": "Removed patients with <30 months data",
                "n_eligible": 256746
            },
            
            "treatment_assignment": {
                "exposed": "Patients with SSD patterns (n=143,579)",
                "unexposed": "Patients without SSD patterns (n=113,167)",
                "definition": "OR logic: ≥3 normal labs OR ≥2 referrals OR >90 days meds"
            },
            
            "baseline_alignment": {
                "index_date": "First qualifying SSD pattern",
                "grace_period": "12-month exposure window",
                "washout": "No prior SSD patterns"
            },
            
            "confounding_control": {
                "measured": [
                    "Demographics (age, sex)",
                    "Comorbidities (Charlson score)",
                    "Baseline utilization",
                    "Mental health diagnoses",
                    "Site effects"
                ],
                "methods": [
                    "Propensity score matching",
                    "Inverse probability weighting",
                    "Doubly robust estimation (TMLE)"
                ]
            },
            
            "causal_assumptions": {
                "exchangeability": "Conditional on measured confounders",
                "positivity": "Common support verified (ESS >80%)",
                "consistency": "Well-defined exposure",
                "no_interference": "Individual-level treatment"
            },
            
            "sensitivity_analyses": [
                "E-value for unmeasured confounding",
                "MC-SIMEX for exposure misclassification",
                "Alternative exposure definitions (AND logic)",
                "Competing risk of death"
            ]
        },
        
        "key_differences": {
            "randomization": "Observational vs randomized assignment",
            "blinding": "Impossible in observational setting",
            "adherence": "Not controlled (natural variation)",
            "measurement": "EMR-based vs protocol-driven"
        },
        
        "validity_threats": {
            "internal": [
                "Unmeasured confounding (health anxiety)",
                "Selection bias (EMR completeness)",
                "Information bias (coding accuracy)"
            ],
            "external": [
                "CPCSSN representativeness",
                "Pre-pandemic data only",
                "Canadian healthcare context"
            ]
        }
    }
    
    # Save as JSON
    output_path = output_dir / "target_trial_emulation.json"
    with open(output_path, 'w') as f:
        json.dump(target_trial, f, indent=2)
    
    # Create markdown documentation
    _create_markdown_documentation(target_trial, output_dir)
    
    logger.info(f"Target trial emulation protocol saved to {output_path}")
    return target_trial


def _create_markdown_documentation(protocol: dict, output_dir: Path):
    """Create markdown version of protocol (≤50 lines)."""
    md_content = f"""# Target Trial Emulation Protocol

Generated: {protocol['metadata']['created']}

## Ideal Randomized Controlled Trial

### Title
{protocol['ideal_rct']['title']}

### Design
- Type: {protocol['ideal_rct']['design']}
- Duration: {protocol['ideal_rct']['duration']}
- Randomization: {protocol['ideal_rct']['randomization']['method']}

### Eligibility Criteria

**Inclusion:**
{chr(10).join(f"- {c}" for c in protocol['ideal_rct']['eligibility']['inclusion'])}

**Exclusion:**
{chr(10).join(f"- {c}" for c in protocol['ideal_rct']['eligibility']['exclusion'])}

### Interventions

**Treatment:** {protocol['ideal_rct']['interventions']['treatment']['name']}
{chr(10).join(f"- {c}" for c in protocol['ideal_rct']['interventions']['treatment']['components'])}

**Control:** {protocol['ideal_rct']['interventions']['control']['name']}
- {protocol['ideal_rct']['interventions']['control']['description']}

## Observational Study Emulation

### Treatment Assignment
- Exposed: {protocol['emulation_strategy']['treatment_assignment']['exposed']}
- Unexposed: {protocol['emulation_strategy']['treatment_assignment']['unexposed']}

### Causal Assumptions
{chr(10).join(f"- **{k.title()}**: {v}" for k, v in protocol['emulation_strategy']['causal_assumptions'].items())}

### Methods for Confounding Control
{chr(10).join(f"- {m}" for m in protocol['emulation_strategy']['confounding_control']['methods'])}

### Sensitivity Analyses
{chr(10).join(f"- {s}" for s in protocol['emulation_strategy']['sensitivity_analyses'])}
"""
    
    output_path = output_dir / "target_trial_emulation.md"
    with open(output_path, 'w') as f:
        f.write(md_content)


if __name__ == "__main__":
    # Create target trial documentation
    protocol = create_target_trial_protocol()
    print("✓ Target trial emulation protocol created")