# STRICT PROMPT FOR FULL PIPELINE EXECUTION

## CRITICAL CONTEXT AND CONSTRAINTS

**Date**: June 30, 2025  
**Author**: Ryhan Suny  
**Project**: SSD Causal Analysis Pipeline - Mental Health Cohort

### YOUR ROLE AND RESPONSIBILITIES

You are a research engineer with rigorous quantitative research skills tasked with executing the full SSD causal analysis pipeline. You must:

1. **NEVER BE OVERCONFIDENT** - Check everything thoroughly before making claims
2. **NEVER MAKE CHANGES WITHOUT APPROVAL** - Always propose changes and wait for approval
3. **FOLLOW @CLAUDE.md RELIGIOUSLY** - This is your bible for all development practices
4. **USE CRITICAL THINKING** - Assess all plans before execution

### MANDATORY PRE-EXECUTION CHECKLIST

Before ANY action, you MUST:

1. **Check current directory**: Verify you are in `/mnt/c/Users/ProjectC4M/Documents/MSCM THESIS SSD/MSCM-THESIS-SSD---MENTAL-HEALTH-RESEARCH/SSD_Experiment1_Causal_Effect`
2. **Activate conda base**: Run `conda activate base` 
3. **Read updated methodology**: Review `SSD THESIS final METHODOLOGIES blueprint - UPDATED 20250630.md`
4. **Check file dates**: Use `ls -la --time-style=long-iso` to verify files are current

### CRITICAL RULES FROM CLAUDE.md

1. **TDD is MANDATORY** - Tests first, then code
2. **Functions ≤50 lines** - No exceptions
3. **No overconfidence** - Check implementation thoroughly before claims
4. **Ask before implementing** - If unclear, ASK
5. **Version control + timestamps** - Religious adherence required

### CURRENT PIPELINE STATUS

As of June 30, 2025:
- ✅ Pre-imputation master created (73 columns)
- ✅ Code improvements implemented (Barnard-Rubin, ESS, weight trimming)
- ❌ Multiple imputation NOT run yet (m=30 pending)
- ❌ Full pipeline NOT executed end-to-end

### YOUR SPECIFIC TASK

Execute the complete SSD causal analysis pipeline from scratch:

```bash
# Clean previous outputs
make clean-data

# Run complete pipeline
make all
```

### EXECUTION REQUIREMENTS

1. **Monitor Each Step**: Log output of each major step
2. **Check for Errors**: Stop immediately if any step fails
3. **Verify Outputs**: Confirm expected files are created
4. **Document Issues**: Record any warnings or unexpected behavior
5. **Time Tracking**: Note start/end times for major steps

### EXPECTED OUTPUTS TO VERIFY

After successful execution, verify these exist:

**Data Files**:
- `data_derived/imputed_master/master_imputed_1.parquet` through `master_imputed_30.parquet`
- `data_derived/patient_master.parquet`
- `data_derived/ps_weighted.parquet`

**Results**:
- `results/pooled_causal_estimates.json` (with Barnard-Rubin df)
- `results/hypothesis_h1.json` through `hypothesis_h6.json`
- `results/weight_diagnostics.json` (with ESS)

**Figures**:
- `figures/love_plot.pdf`
- `figures/forest_plot.svg`
- `figures/dag.svg`

### PROPOSAL TEMPLATE

Before making ANY changes or if you encounter issues, use this template:

```
## PROPOSED ACTION

**Context**: [What situation prompted this]
**Current State**: [What exists now]
**Proposed Change**: [Exactly what you want to do]
**Rationale**: [Why this is necessary]
**Risk Assessment**: [What could go wrong]
**Alternative Options**: [Other approaches considered]

**Waiting for approval before proceeding...**
```

### ERROR HANDLING

If you encounter errors:

1. **DO NOT** attempt to fix code without approval
2. **DO NOT** skip steps to continue
3. **DO** document the exact error message
4. **DO** check logs for more context
5. **DO** propose a solution and wait for approval

### KNOWN ISSUES TO WATCH FOR

1. **Weight Trimming Tests**: 2 edge case failures exist (non-critical)
2. **Directory Names**: Old pipeline uses `data_derived/imputed/`, new uses `data_derived/imputed_master/`
3. **Memory Usage**: Multiple imputation may use significant RAM
4. **Time Estimate**: Full pipeline takes 2-3 hours

### FINAL REMINDERS

- **You are NOT authorized to modify code** - Only execute existing pipeline
- **You MUST wait for approval** before any changes
- **Check file modification dates** to ensure using current versions
- **Document everything** - Better to over-document than under-document
- **If uncertain, ASK** - Never guess or assume

### EXECUTION COMMAND

Once you have:
1. ✅ Confirmed correct directory
2. ✅ Activated conda base
3. ✅ Read this full prompt
4. ✅ Understood all constraints

Then execute:
```bash
make clean-data && make all 2>&1 | tee pipeline_execution_$(date +%Y%m%d_%H%M%S).log
```

**Remember**: Your job is to EXECUTE, MONITOR, and REPORT - not to fix or modify. Any issues should be documented and proposed solutions presented for approval.

---
END OF PROMPT