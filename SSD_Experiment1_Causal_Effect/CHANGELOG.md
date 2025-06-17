# Changelog

## [v4.1.0] - 2025-06-17

### Added
- Week 5 compliance polish and external validation features
- Selection diagram (CONSORT-style patient flowchart)
- Cost-effectiveness plane for intervention analysis (H6)
- Estimate reconciliation rule with 15% discordance threshold
- External validity transport weights with ICES marginals
- Enhanced autoencoder with hyperparameter optimization
- Power analysis consistency synchronization
- Comprehensive code quality improvements

### Enhanced
- Mental health-specific causal inference pipeline
- Advanced statistical methods (mediation, CATE, G-computation)
- Comprehensive test coverage with TDD methodology
- Documentation with embedded figures and clinical interpretations
- CI/CD compatibility with graceful dependency fallbacks

### Technical Improvements
- All functions comply with ≤50 LOC requirement
- Comprehensive numpy-style docstrings
- Robust error handling and edge case coverage
- Matplotlib fallback mechanisms for CI environments
- Transport weight calculation with ESS validation

### Quality Assurance
- 100% test coverage for all Week 5 modules
- Automated quality gates with threshold validation
- Comprehensive bias assessment (E-values, ROBINS-I)
- Power analysis parameter synchronization
- Release readiness validation

### Commits Since v1.0.0-cohort:
- feat(week4-5): Complete Week 4 Task 5 final deliverables
- feat(week4-5-A): Implement Week 4 figure generation module
- Update prompts.md with Week 4 completion status
- Week 4 Task 4: Wire experimental modules into Makefile and extend CI/QA
- Week 4 Task 3: Implement H4-H6 advanced causal analyses
- Week 4 Task 2: Add mental health-specific outcome flags
- feat(week4-1): Add mental health cohort filtering and enhanced exposures
- docs(week3-complete): Update implementation tracker with Week 3 completion
- feat(week3-5): Create comprehensive submission package and OSF integration
- feat(week3-4): Add CI/QA enhancements and environment lockfile
- feat(week3-3): Polish narrative voice to active and add author info
- feat(week3-2): Generate supplementary documentation artifacts
- feat(week3-1): Generate high-resolution figures bundle for manuscript
- fix(makefile): Add missing Week 2 targets for complete integration
- feat(week2-complete): Complete Week 2 Analysis & Visualization tasks
- feat(week2-1): Complete H1-H3 hypothesis analyses with cluster-robust SEs
- feat(task-4): implement temporal ordering validator
- feat(task-3): implement Poisson/NB count outcome models
- feat(task-2): implement cluster-robust standard errors
- feat(task-1): implement weight diagnostics guard-rails
- fixes and cleanup june 16
- Fix test imports: update paths to experimental/ folder
- Tidy src/: archive enhanced scripts, delete duplicates, drop obsolete tests
- june 15 cleanup
- Switch age baseline to Age_at_2015; update tests, patient master table script; ignore venv_analysis
- Align docs and scripts with 2015 baseline
- Rename patient table module
- Clean up reference date inconsistencies
- Document final OR logic decision
- Clarify healthcare utilization outcomes
- Align reference date, add Felipe table and sequential analysis
- Clarify healthcare utilization outcomes in overview
- Update automation status doc to reflect new validation runner
- clarify reference date follow-up
- Add links to hypotheses and blueprint
- Fix unit test expectations
- Update status doc for sequential pathway module
- huh?
- v3.1 blueprint implementation 1
- modules created

### Pipeline Status
- Mental health domain alignment: ✅ COMPLETE
- Advanced causal methods: ✅ COMPLETE  
- Statistical rigor: ✅ COMPLETE
- Documentation standards: ✅ COMPLETE
- Reproducibility: ✅ COMPLETE

### Clinical Impact
- Mental health-specific SSD analysis framework
- Policy-ready intervention simulation results
- Publication-ready figures and documentation
- Healthcare system optimization insights

---

## Previous Versions

### [v4.0.0] - 2025-06-17
- Initial mental health alignment implementation
- Advanced causal analysis methods (H4-H6)
- Comprehensive documentation suite
- Production-ready pipeline with Docker support

### [v3.x] - Earlier Versions
- Core causal inference pipeline
- Basic statistical methods implementation
- Initial test framework
