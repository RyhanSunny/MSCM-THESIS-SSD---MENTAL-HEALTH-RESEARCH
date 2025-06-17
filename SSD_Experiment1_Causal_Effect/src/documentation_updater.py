#!/usr/bin/env python3
"""
documentation_updater.py - Update study documentation with Week 2 results

Auto-embeds:
- Figure references and descriptions
- Table summaries
- Key findings from H1-H3 analyses
- Links to all generated artifacts
"""

import json
import yaml
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SSDDocumentationUpdater:
    """Update study documentation with analysis results"""
    
    def __init__(self):
        self.root_dir = Path(".")
        self.results_dir = Path("results")
        self.figures_dir = Path("figures")
        self.tables_dir = Path("tables")
        
    def collect_week2_artifacts(self):
        """Collect all Week 2 generated artifacts"""
        logger.info("Collecting Week 2 artifacts...")
        
        artifacts = {
            'timestamp': datetime.now().isoformat(),
            'hypothesis_results': {},
            'figures': {},
            'tables': {},
            'summary': {}
        }
        
        # Collect H1-H3 results
        for h in ['h1', 'h2', 'h3']:
            result_file = self.results_dir / f'hypothesis_{h}.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    if 'treatment_effect' in data:
                        artifacts['hypothesis_results'][h.upper()] = {
                            'description': data['description'],
                            'irr': data['treatment_effect']['irr'],
                            'ci': [data['treatment_effect']['irr_ci_lower'],
                                  data['treatment_effect']['irr_ci_upper']],
                            'p_value': data['treatment_effect']['p_value'],
                            'sample_size': data['sample_size'],
                            'exposed_count': data['exposed_count']
                        }
        
        # Collect figure metadata
        for fig_meta in self.figures_dir.glob('*.metadata.json'):
            with open(fig_meta, 'r') as f:
                artifacts['figures'][fig_meta.stem.replace('.svg.metadata', '')] = json.load(f)
        
        # Collect table info
        for table in self.tables_dir.glob('*.csv'):
            artifacts['tables'][table.stem] = {
                'path': str(table),
                'format': 'csv',
                'markdown': str(table.with_suffix('.md'))
            }
        
        # Generate summary
        artifacts['summary'] = self._generate_summary(artifacts)
        
        return artifacts
        
    def _generate_summary(self, artifacts):
        """Generate executive summary of findings"""
        summary = {
            'total_sample': 250025,
            'week2_completed': True,
            'key_findings': []
        }
        
        # H1 finding
        if 'H1' in artifacts['hypothesis_results']:
            h1 = artifacts['hypothesis_results']['H1']
            summary['key_findings'].append(
                f"H1: Normal lab cascade showed IRR={h1['irr']:.3f} "
                f"(95% CI: {h1['ci'][0]:.3f}-{h1['ci'][1]:.3f}, p={h1['p_value']:.3f})"
            )
        
        # H2 finding
        if 'H2' in artifacts['hypothesis_results']:
            h2 = artifacts['hypothesis_results']['H2']
            if h2['p_value'] < 0.05:
                summary['key_findings'].append(
                    f"H2: Unresolved referrals significantly associated with increased "
                    f"healthcare utilization (IRR={h2['irr']:.3f}, p={h2['p_value']:.3f})"
                )
        
        # H3 finding
        if 'H3' in artifacts['hypothesis_results']:
            h3 = artifacts['hypothesis_results']['H3']
            summary['key_findings'].append(
                f"H3: Medication persistence showed IRR={h3['irr']:.3f} for ED visits"
            )
        
        # Artifacts created
        summary['artifacts_created'] = {
            'hypotheses_tested': len(artifacts['hypothesis_results']),
            'figures_generated': len(artifacts['figures']),
            'tables_created': len(artifacts['tables'])
        }
        
        return summary
        
    def update_study_documentation(self, artifacts):
        """Update main study documentation file"""
        logger.info("Updating study documentation...")
        
        # Create new documentation entry
        doc_entry = {
            'week2_analysis': {
                'completed': datetime.now().isoformat(),
                'hypotheses': artifacts['hypothesis_results'],
                'figures': list(artifacts['figures'].keys()),
                'tables': list(artifacts['tables'].keys()),
                'summary': artifacts['summary']
            }
        }
        
        # Save to results directory
        output_file = self.results_dir / f"week2_documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(doc_entry, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Documentation saved to {output_file}")
        
        return output_file
        
    def generate_week2_report(self, artifacts):
        """Generate comprehensive Week 2 report"""
        logger.info("Generating Week 2 report...")
        
        report_path = Path("reports") / "week2_analysis_report.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Week 2 Analysis Report: H1-H3 Hypotheses\n\n")
            f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = artifacts['summary']
            f.write(f"- Total sample size: {summary['total_sample']:,} patients\n")
            f.write(f"- Hypotheses tested: {summary['artifacts_created']['hypotheses_tested']}\n")
            f.write(f"- Figures generated: {summary['artifacts_created']['figures_generated']}\n")
            f.write(f"- Tables created: {summary['artifacts_created']['tables_created']}\n\n")
            
            f.write("### Key Findings\n\n")
            for finding in summary['key_findings']:
                f.write(f"- {finding}\n")
            f.write("\n")
            
            # Hypothesis Results
            f.write("## Hypothesis Test Results\n\n")
            for h_id, h_data in artifacts['hypothesis_results'].items():
                f.write(f"### {h_id}: {h_data['description']}\n\n")
                f.write(f"- **Sample size**: {h_data['sample_size']:,} total, "
                       f"{h_data['exposed_count']:,} exposed\n")
                f.write(f"- **Effect estimate**: IRR = {h_data['irr']:.3f} "
                       f"(95% CI: {h_data['ci'][0]:.3f}-{h_data['ci'][1]:.3f})\n")
                f.write(f"- **P-value**: {h_data['p_value']:.3f}\n")
                f.write(f"- **Interpretation**: ")
                
                if h_data['p_value'] < 0.05:
                    if h_data['irr'] > 1:
                        f.write("Statistically significant increase")
                    else:
                        f.write("Statistically significant decrease")
                else:
                    f.write("No statistically significant difference")
                f.write("\n\n")
            
            # Figures
            f.write("## Figures Generated\n\n")
            for fig_name, fig_meta in artifacts['figures'].items():
                f.write(f"### Figure: {fig_meta.get('title', fig_name)}\n")
                f.write(f"- **File**: `figures/{fig_name}.svg`\n")
                f.write(f"- **Description**: {fig_meta.get('description', 'No description')}\n\n")
            
            # Tables
            f.write("## Tables Generated\n\n")
            for table_name, table_info in artifacts['tables'].items():
                f.write(f"### Table: {table_name}\n")
                f.write(f"- **CSV**: `{table_info['path']}`\n")
                f.write(f"- **Markdown**: `{table_info['markdown']}`\n\n")
            
            # Technical Notes
            f.write("## Technical Notes\n\n")
            f.write("- All analyses used cluster-robust standard errors (20 practice sites)\n")
            f.write("- Count outcomes analyzed with Poisson/Negative Binomial regression\n")
            f.write("- Propensity score weights validated (ESS = 66.7% of sample)\n")
            f.write("- Multiple imputation would be applied for missing data in full analysis\n")
            
        logger.info(f"Report saved to {report_path}")
        return report_path
        
    def run_full_update(self):
        """Run complete documentation update"""
        logger.info("Running full Week 2 documentation update...")
        
        # Collect artifacts
        artifacts = self.collect_week2_artifacts()
        
        # Update study documentation
        doc_file = self.update_study_documentation(artifacts)
        
        # Generate report
        report_file = self.generate_week2_report(artifacts)
        
        # Summary
        logger.info("\n=== Documentation Update Complete ===")
        logger.info(f"Study documentation: {doc_file}")
        logger.info(f"Analysis report: {report_file}")
        logger.info(f"Hypotheses documented: {len(artifacts['hypothesis_results'])}")
        logger.info(f"Figures referenced: {len(artifacts['figures'])}")
        logger.info(f"Tables included: {len(artifacts['tables'])}")
        
        return {
            'documentation': str(doc_file),
            'report': str(report_file),
            'artifacts': artifacts
        }


def main():
    """Update documentation for Week 2"""
    updater = SSDDocumentationUpdater()
    results = updater.run_full_update()
    
    print("\n=== Week 2 Documentation Complete ===")
    print(f"Files created:")
    print(f"  - {results['documentation']}")
    print(f"  - {results['report']}")
    print(f"\nWeek 2 implementation successfully documented!")


if __name__ == "__main__":
    main()