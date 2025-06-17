#!/usr/bin/env python3
"""
week5_figures.py - Selection & cost-effectiveness figures

Week 5 Task F: Selection & cost-effectiveness figures
Generates selection diagram and cost-effectiveness plane for H6.

Author: Ryhan Suny <sunnyrayhan2@gmail.com>
Date: 2025-06-17
Version: 4.0.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Handle matplotlib import gracefully
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Ellipse
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Using SVG placeholders.")


def create_selection_diagram(cohort_stats: Dict[str, int], 
                           output_path: Path,
                           format: str = 'svg') -> Path:
    """
    Create CONSORT-style patient selection flowchart
    
    Parameters:
    -----------
    cohort_stats : Dict[str, int]
        Patient counts at each selection step
    output_path : Path
        Output file path
    format : str
        Output format ('svg' or 'png')
        
    Returns:
    --------
    Path
        Path to created diagram
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not MATPLOTLIB_AVAILABLE:
        # Create SVG placeholder
        svg_content = _create_selection_diagram_svg_placeholder(cohort_stats)
        with open(output_path, 'w') as f:
            f.write(svg_content)
        logger.info(f"Created selection diagram placeholder: {output_path}")
        return output_path
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Define positions for flowchart boxes
    boxes = [
        {'text': f"Total patients screened\nn = {cohort_stats.get('total_patients', 'N/A'):,}", 
         'pos': (5, 18), 'width': 3, 'height': 1.5},
        {'text': f"After age filter (≥18 years)\nn = {cohort_stats.get('after_age_filter', 'N/A'):,}", 
         'pos': (5, 15.5), 'width': 3, 'height': 1.5},
        {'text': f"After observation period filter\nn = {cohort_stats.get('after_observation_filter', 'N/A'):,}", 
         'pos': (5, 13), 'width': 3, 'height': 1.5},
        {'text': f"Mental health subset\nn = {cohort_stats.get('mental_health_subset', 'N/A'):,}", 
         'pos': (5, 10.5), 'width': 3, 'height': 1.5},
    ]
    
    # Draw boxes and text
    for box in boxes:
        rect = patches.Rectangle(
            (box['pos'][0] - box['width']/2, box['pos'][1] - box['height']/2),
            box['width'], box['height'],
            linewidth=2, edgecolor='black', facecolor='lightblue'
        )
        ax.add_patch(rect)
        ax.text(box['pos'][0], box['pos'][1], box['text'], 
               ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw arrows between boxes
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    for i in range(len(boxes)-1):
        ax.annotate('', xy=(5, boxes[i+1]['pos'][1] + boxes[i+1]['height']/2),
                   xytext=(5, boxes[i]['pos'][1] - boxes[i]['height']/2),
                   arrowprops=arrow_props)
    
    # Add final split for exposed/control groups
    if 'exposed_group' in cohort_stats and 'control_group' in cohort_stats:
        exposed_box = {'text': f"Exposed group\nn = {cohort_stats['exposed_group']:,}", 
                      'pos': (2.5, 7), 'width': 2.5, 'height': 1.5}
        control_box = {'text': f"Control group\nn = {cohort_stats['control_group']:,}", 
                      'pos': (7.5, 7), 'width': 2.5, 'height': 1.5}
        
        for box in [exposed_box, control_box]:
            rect = patches.Rectangle(
                (box['pos'][0] - box['width']/2, box['pos'][1] - box['height']/2),
                box['width'], box['height'],
                linewidth=2, edgecolor='black', facecolor='lightgreen'
            )
            ax.add_patch(rect)
            ax.text(box['pos'][0], box['pos'][1], box['text'], 
                   ha='center', va='center', fontsize=10, weight='bold')
        
        # Arrows to split groups
        ax.annotate('', xy=(2.5, 7.75), xytext=(4.5, 9.25), arrowprops=arrow_props)
        ax.annotate('', xy=(7.5, 7.75), xytext=(5.5, 9.25), arrowprops=arrow_props)
    
    # Add matched pairs if available
    if 'matched_pairs' in cohort_stats:
        matched_box = {'text': f"Matched pairs\nn = {cohort_stats['matched_pairs']:,} pairs", 
                      'pos': (5, 4), 'width': 3, 'height': 1.5}
        rect = patches.Rectangle(
            (matched_box['pos'][0] - matched_box['width']/2, 
             matched_box['pos'][1] - matched_box['height']/2),
            matched_box['width'], matched_box['height'],
            linewidth=2, edgecolor='black', facecolor='yellow'
        )
        ax.add_patch(rect)
        ax.text(matched_box['pos'][0], matched_box['pos'][1], matched_box['text'], 
               ha='center', va='center', fontsize=10, weight='bold')
        
        # Arrows from both groups to matched pairs
        ax.annotate('', xy=(4.25, 4.75), xytext=(2.5, 6.25), arrowprops=arrow_props)
        ax.annotate('', xy=(5.75, 4.75), xytext=(7.5, 6.25), arrowprops=arrow_props)
    
    # Add title
    ax.text(5, 19.5, 'Patient Selection Flowchart', 
           ha='center', va='center', fontsize=16, weight='bold')
    
    # Save figure
    plt.tight_layout()
    if format.lower() == 'png':
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created selection diagram: {output_path}")
    return output_path


def _create_selection_diagram_svg_placeholder(cohort_stats: Dict[str, int]) -> str:
    """Create SVG placeholder for selection diagram when matplotlib unavailable"""
    total = cohort_stats.get('total_patients', 'N/A')
    final = cohort_stats.get('after_observation_filter', 'N/A')
    
    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="600" height="800" xmlns="http://www.w3.org/2000/svg">
    <rect width="600" height="800" fill="white"/>
    
    <!-- Title -->
    <text x="300" y="50" text-anchor="middle" font-size="20" font-weight="bold">
        Patient Selection Flowchart (CONSORT Style)
    </text>
    
    <!-- Total patients box -->
    <rect x="200" y="100" width="200" height="80" fill="lightblue" stroke="black" stroke-width="2"/>
    <text x="300" y="130" text-anchor="middle" font-size="14" font-weight="bold">
        Total patients screened
    </text>
    <text x="300" y="150" text-anchor="middle" font-size="14">
        n = {total}
    </text>
    
    <!-- Arrow -->
    <line x1="300" y1="180" x2="300" y2="220" stroke="black" stroke-width="2" marker-end="url(#arrowhead)"/>
    
    <!-- Final cohort box -->
    <rect x="200" y="240" width="200" height="80" fill="lightgreen" stroke="black" stroke-width="2"/>
    <text x="300" y="270" text-anchor="middle" font-size="14" font-weight="bold">
        Final analysis cohort
    </text>
    <text x="300" y="290" text-anchor="middle" font-size="14">
        n = {final}
    </text>
    
    <!-- Arrow marker definition -->
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="10" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="black"/>
        </marker>
    </defs>
    
    <!-- Note -->
    <text x="300" y="400" text-anchor="middle" font-size="12" fill="gray">
        Generated with matplotlib fallback (matplotlib not available)
    </text>
</svg>"""
    return svg_content


def create_cost_effectiveness_plane(scenarios: List[Dict[str, Any]], 
                                   output_path: Path,
                                   willingness_to_pay: float = 50000,
                                   include_confidence_intervals: bool = False,
                                   format: str = 'svg') -> Path:
    """
    Create cost-effectiveness plane for intervention scenarios
    
    Parameters:
    -----------
    scenarios : List[Dict[str, Any]]
        List of intervention scenarios with cost and effectiveness
    output_path : Path
        Output file path
    willingness_to_pay : float
        Willingness-to-pay threshold ($/QALY)
    include_confidence_intervals : bool
        Whether to include uncertainty ellipses
    format : str
        Output format ('svg' or 'png')
        
    Returns:
    --------
    Path
        Path to created diagram
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not MATPLOTLIB_AVAILABLE:
        # Create SVG placeholder
        svg_content = _create_cost_effectiveness_svg_placeholder(scenarios)
        with open(output_path, 'w') as f:
            f.write(svg_content)
        logger.info(f"Created cost-effectiveness plane placeholder: {output_path}")
        return output_path
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract costs and effectiveness
    costs = [s.get('cost', 0) for s in scenarios]
    effectiveness = [s.get('effectiveness', 0) for s in scenarios]
    names = [s.get('name', f'Scenario {i+1}') for i, s in enumerate(scenarios)]
    
    # Determine plot ranges
    max_cost = max(max(costs), abs(min(costs))) * 1.2
    max_eff = max(max(effectiveness), abs(min(effectiveness))) * 1.2
    
    # Set up axes
    ax.set_xlim(-max_eff, max_eff)
    ax.set_ylim(-max_cost, max_cost)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Color map for quadrants
    colors = []
    for cost, eff in zip(costs, effectiveness):
        quadrant = _classify_quadrant(cost, eff)
        if quadrant == 'NW':  # Dominant (less cost, more effective)
            colors.append('green')
        elif quadrant == 'SE':  # Dominated (more cost, less effective)
            colors.append('red')
        elif quadrant == 'NE':  # Trade-off (more cost, more effective)
            colors.append('orange')
        else:  # SW (less cost, less effective)
            colors.append('blue')
    
    # Plot scenarios
    scatter = ax.scatter(effectiveness, costs, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Add confidence intervals if requested
    if include_confidence_intervals:
        for i, scenario in enumerate(scenarios):
            if all(key in scenario for key in ['cost_ci_lower', 'cost_ci_upper', 
                                             'effectiveness_ci_lower', 'effectiveness_ci_upper']):
                width = scenario['effectiveness_ci_upper'] - scenario['effectiveness_ci_lower']
                height = scenario['cost_ci_upper'] - scenario['cost_ci_lower']
                ellipse = Ellipse((effectiveness[i], costs[i]), width, height, 
                                alpha=0.3, facecolor=colors[i])
                ax.add_patch(ellipse)
    
    # Add willingness-to-pay threshold line
    if willingness_to_pay > 0:
        eff_range = np.linspace(0, max_eff, 100)
        wtp_line = willingness_to_pay * eff_range
        ax.plot(eff_range, wtp_line, 'k--', alpha=0.5, 
               label=f'WTP threshold (${willingness_to_pay:,.0f}/QALY)')
    
    # Add labels for each scenario
    for i, name in enumerate(names):
        ax.annotate(name, (effectiveness[i], costs[i]), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=9, ha='left')
    
    # Labels and title
    ax.set_xlabel('Effectiveness (QALYs gained)', fontsize=12)
    ax.set_ylabel('Incremental Cost ($)', fontsize=12)
    ax.set_title('Cost-Effectiveness Plane\nMental Health Intervention Analysis', 
                fontsize=14, weight='bold')
    
    # Add quadrant labels
    ax.text(max_eff*0.8, max_cost*0.8, 'More costly\nMore effective', 
           ha='center', va='center', fontsize=10, alpha=0.7, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))
    ax.text(-max_eff*0.8, max_cost*0.8, 'Less costly\nMore effective\n(Dominant)', 
           ha='center', va='center', fontsize=10, alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))
    ax.text(max_eff*0.8, -max_cost*0.8, 'More costly\nLess effective\n(Dominated)', 
           ha='center', va='center', fontsize=10, alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
    ax.text(-max_eff*0.8, -max_cost*0.8, 'Less costly\nLess effective', 
           ha='center', va='center', fontsize=10, alpha=0.7,
           bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3))
    
    # Add legend
    if willingness_to_pay > 0:
        ax.legend(loc='upper right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    if format.lower() == 'png':
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created cost-effectiveness plane: {output_path}")
    return output_path


def _create_cost_effectiveness_svg_placeholder(scenarios: List[Dict[str, Any]]) -> str:
    """Create SVG placeholder for cost-effectiveness plane when matplotlib unavailable"""
    scenario_text = ""
    for i, scenario in enumerate(scenarios):
        name = scenario.get('name', f'Scenario {i+1}')
        cost = scenario.get('cost', 0)
        eff = scenario.get('effectiveness', 0)
        scenario_text += f"    <text x='50' y='{150 + i*30}' font-size='12'>{name}: Cost=${cost}, Eff={eff}</text>\n"
    
    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
    <rect width="600" height="400" fill="white"/>
    
    <!-- Title -->
    <text x="300" y="30" text-anchor="middle" font-size="18" font-weight="bold">
        Cost-Effectiveness Plane
    </text>
    <text x="300" y="50" text-anchor="middle" font-size="14">
        Mental Health Intervention Analysis
    </text>
    
    <!-- Axes -->
    <line x1="100" y1="200" x2="500" y2="200" stroke="black" stroke-width="2"/>
    <line x1="300" y1="100" x2="300" y2="300" stroke="black" stroke-width="2"/>
    
    <!-- Axis labels -->
    <text x="500" y="220" text-anchor="middle" font-size="12">Effectiveness →</text>
    <text x="280" y="100" text-anchor="middle" font-size="12" transform="rotate(-90 280 100)">Cost →</text>
    
    <!-- Quadrant labels -->
    <text x="400" y="130" text-anchor="middle" font-size="10" fill="orange">More costly, More effective</text>
    <text x="200" y="130" text-anchor="middle" font-size="10" fill="green">Less costly, More effective (Dominant)</text>
    <text x="400" y="270" text-anchor="middle" font-size="10" fill="red">More costly, Less effective (Dominated)</text>
    <text x="200" y="270" text-anchor="middle" font-size="10" fill="blue">Less costly, Less effective</text>
    
    <!-- Scenario data -->
{scenario_text}
    
    <!-- Note -->
    <text x="300" y="350" text-anchor="middle" font-size="12" fill="gray">
        Generated with matplotlib fallback (matplotlib not available)
    </text>
</svg>"""
    return svg_content


def _classify_quadrant(cost: float, effectiveness: float) -> str:
    """
    Classify cost-effectiveness into quadrants
    
    Parameters:
    -----------
    cost : float
        Incremental cost
    effectiveness : float
        Incremental effectiveness
        
    Returns:
    --------
    str
        Quadrant classification (NE, NW, SE, SW)
    """
    if cost >= 0 and effectiveness >= 0:
        return 'NE'  # Northeast: More costly, more effective
    elif cost < 0 and effectiveness >= 0:
        return 'NW'  # Northwest: Less costly, more effective (dominant)
    elif cost >= 0 and effectiveness < 0:
        return 'SE'  # Southeast: More costly, less effective (dominated)
    else:
        return 'SW'  # Southwest: Less costly, less effective


def calculate_icer(delta_cost: float, delta_effectiveness: float) -> float:
    """
    Calculate Incremental Cost-Effectiveness Ratio (ICER)
    
    Parameters:
    -----------
    delta_cost : float
        Incremental cost difference
    delta_effectiveness : float
        Incremental effectiveness difference
        
    Returns:
    --------
    float
        ICER value (cost per unit effectiveness)
    """
    if delta_effectiveness == 0:
        return float('inf')
    
    return delta_cost / delta_effectiveness


def embed_figures_in_documentation(doc_path: Path,
                                  selection_diagram_path: Path,
                                  cost_plane_path: Path) -> Path:
    """
    Embed new figures into Week 4 documentation
    
    Parameters:
    -----------
    doc_path : Path
        Path to documentation file
    selection_diagram_path : Path
        Path to selection diagram
    cost_plane_path : Path
        Path to cost-effectiveness plane
        
    Returns:
    --------
    Path
        Updated documentation path
    """
    with open(doc_path, 'r') as f:
        content = f.read()
    
    # Prepare figure section
    figures_section = f"""
### Week 5 Additional Figures

#### Patient Selection Flowchart
![Selection Diagram]({selection_diagram_path.name})

**Figure**: CONSORT-style patient selection flowchart showing the progression from initial screening to final analysis cohort. This diagram illustrates the application of inclusion/exclusion criteria and the resulting sample sizes at each stage of the selection process.

#### Cost-Effectiveness Analysis
![Cost-Effectiveness Plane]({cost_plane_path.name})

**Figure**: Cost-effectiveness plane for mental health intervention scenarios (H6 analysis). Each point represents a different intervention strategy plotted by incremental cost (y-axis) and incremental effectiveness (x-axis). The four quadrants represent different cost-effectiveness relationships relative to current practice.

"""
    
    # Insert figures section
    if '<!-- WEEK5_FIGURES_PLACEHOLDER -->' in content:
        content = content.replace('<!-- WEEK5_FIGURES_PLACEHOLDER -->', figures_section)
    else:
        # Add at the end of existing figures section
        figures_end = content.find('### New Figures Section')
        if figures_end != -1:
            insert_pos = content.find('\n## ', figures_end)
            if insert_pos != -1:
                content = content[:insert_pos] + figures_section + content[insert_pos:]
            else:
                content += figures_section
        else:
            content += figures_section
    
    # Write updated content
    with open(doc_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Embedded figures in documentation: {doc_path}")
    return doc_path


def update_figure_list_in_readme(readme_path: Path, 
                                new_figures: List[str]) -> Path:
    """
    Update figure list in README with new figures
    
    Parameters:
    -----------
    readme_path : Path
        Path to README file
    new_figures : List[str]
        List of new figure descriptions
        
    Returns:
    --------
    Path
        Updated README path
    """
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Find figure list section
    figure_list_start = content.find('## Generated Figures')
    if figure_list_start == -1:
        # Create new section
        figure_section = "\n## Generated Figures\n\n"
        for figure in new_figures:
            figure_section += f"- {figure}\n"
        content += figure_section
    else:
        # Add to existing list
        figure_list_end = content.find('<!-- FIGURE_LIST_END -->')
        if figure_list_end == -1:
            # Find end of existing list
            next_section = content.find('\n## ', figure_list_start + 1)
            insert_pos = next_section if next_section != -1 else len(content)
        else:
            insert_pos = figure_list_end
        
        # Add new figures
        new_entries = ""
        for figure in new_figures:
            new_entries += f"- {figure}\n"
        
        content = content[:insert_pos] + new_entries + content[insert_pos:]
    
    # Write updated content
    with open(readme_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Updated figure list in README: {readme_path}")
    return readme_path


def main():
    """Main execution for generating Week 5 figures"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Week 5 figures')
    parser.add_argument('--output-dir', type=Path, default=Path('figures'),
                       help='Output directory for figures')
    parser.add_argument('--format', choices=['svg', 'png'], default='svg',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock cohort statistics for demonstration
    cohort_stats = {
        'total_patients': 350000,
        'after_age_filter': 280000,
        'after_observation_filter': 250025,
        'mental_health_subset': 45000,
        'exposed_group': 12500,
        'control_group': 37500,
        'matched_pairs': 8500
    }
    
    # Mock cost-effectiveness scenarios
    ce_scenarios = [
        {'name': 'Baseline', 'cost': 0, 'effectiveness': 0},
        {'name': 'Enhanced Screening', 'cost': 150, 'effectiveness': 0.12},
        {'name': 'Reduced Testing', 'cost': -75, 'effectiveness': -0.05},
        {'name': 'AI-Assisted Diagnosis', 'cost': 200, 'effectiveness': 0.18},
        {'name': 'Telemedicine', 'cost': -50, 'effectiveness': 0.08}
    ]
    
    # Generate figures
    selection_path = create_selection_diagram(
        cohort_stats=cohort_stats,
        output_path=args.output_dir / f'selection_diagram.{args.format}',
        format=args.format
    )
    
    cost_plane_path = create_cost_effectiveness_plane(
        scenarios=ce_scenarios,
        output_path=args.output_dir / f'cost_plane.{args.format}',
        format=args.format
    )
    
    print(f"Generated figures:")
    print(f"  Selection diagram: {selection_path}")
    print(f"  Cost-effectiveness plane: {cost_plane_path}")


if __name__ == "__main__":
    main()