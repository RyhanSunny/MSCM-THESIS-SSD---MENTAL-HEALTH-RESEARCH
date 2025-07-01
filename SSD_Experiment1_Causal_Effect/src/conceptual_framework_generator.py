#!/usr/bin/env python3
"""
conceptual_framework_generator.py - Create conceptual framework diagram for SSD study

Creates a high-quality conceptual framework diagram showing theoretical relationships
between mental health vulnerability, SSD patterns, and healthcare utilization.

Following CLAUDE.md requirements:
- TDD approach (tests first)
- Functions ≤50 lines
- Evidence-based implementation
- Version numbering and timestamps

Author: Ryhan Suny (Toronto Metropolitan University)
Version: 1.0
Date: 2025-07-01
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_conceptual_framework(output_dir: Path = Path("figures")) -> str:
    """
    Create conceptual framework diagram showing theoretical pathways.
    
    Returns path to saved figure.
    """
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define color scheme
    colors = {
        'vulnerability': '#FFE5B4',  # Peach
        'exposure': '#B4D4FF',      # Light blue
        'mediator': '#C8E6C9',      # Light green
        'outcome': '#FFB4B4',       # Light red
        'moderator': '#E1BEE7'      # Light purple
    }
    
    # Create boxes
    _add_conceptual_boxes(ax, colors)
    
    # Add arrows showing relationships
    _add_conceptual_arrows(ax)
    
    # Add title and labels
    plt.title('Conceptual Framework: SSD Patterns in Mental Health Population', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    _add_framework_legend(ax, colors)
    
    # Save figure
    output_path = output_dir / f'conceptual_framework_{timestamp}.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    
    logger.info(f"Conceptual framework saved to {output_path}")
    plt.close()
    
    return str(output_path)


def _add_conceptual_boxes(ax, colors):
    """Add conceptual framework boxes (≤50 lines)."""
    # Box style
    box_style = "round,pad=0.3"
    
    # 1. Mental Health Vulnerability (Left)
    vulnerability_box = FancyBboxPatch(
        (0.5, 4), 2.5, 2,
        boxstyle=box_style,
        facecolor=colors['vulnerability'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(vulnerability_box)
    ax.text(1.75, 5, 'Pre-existing\nMental Health\nConditions', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 2. SSD Pattern Exposures (Center-left)
    exposures = [
        ('Normal Lab\nResults\n(≥3 panels)', 7.5),
        ('Unresolved\nReferrals\n(≥2 NYD)', 5),
        ('Psychotropic\nPersistence\n(>90 days)', 2.5)
    ]
    
    for text, y in exposures:
        box = FancyBboxPatch(
            (4, y-0.75), 2.5, 1.5,
            boxstyle=box_style,
            facecolor=colors['exposure'],
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(5.25, y, text, ha='center', va='center', fontsize=10)
    
    # 3. SSDSI Mediator (Center)
    mediator_box = FancyBboxPatch(
        (7.5, 4), 2.5, 2,
        boxstyle=box_style,
        facecolor=colors['mediator'],
        edgecolor='black',
        linewidth=3
    )
    ax.add_patch(mediator_box)
    ax.text(8.75, 5, 'SSD Severity\nIndex\n(SSDSI)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 4. Outcomes (Right)
    outcome_box = FancyBboxPatch(
        (11, 4), 2.5, 2,
        boxstyle=box_style,
        facecolor=colors['outcome'],
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(outcome_box)
    ax.text(12.25, 5, 'Healthcare\nUtilization\n↑ ED Visits\n↑ MH Services', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 5. Effect Modifiers (Bottom)
    modifiers = ['Age<40', 'Female', 'Anxiety', 'High Baseline']
    for i, mod in enumerate(modifiers):
        x = 4 + i * 2
        box = FancyBboxPatch(
            (x, 0.5), 1.8, 0.8,
            boxstyle=box_style,
            facecolor=colors['moderator'],
            edgecolor='black',
            linewidth=1
        )
        ax.add_patch(box)
        ax.text(x + 0.9, 0.9, mod, ha='center', va='center', fontsize=9)


def _add_conceptual_arrows(ax):
    """Add arrows showing relationships (≤50 lines)."""
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    arrow_kwargs = dict(arrowstyle=arrow_style, color="black", lw=2)
    
    # Vulnerability → Exposures
    ax.add_patch(FancyArrowPatch((3, 5), (4, 7.5), **arrow_kwargs))
    ax.add_patch(FancyArrowPatch((3, 5), (4, 5), **arrow_kwargs))
    ax.add_patch(FancyArrowPatch((3, 5), (4, 2.5), **arrow_kwargs))
    
    # Exposures → SSDSI
    ax.add_patch(FancyArrowPatch((6.5, 7.5), (7.5, 5.5), **arrow_kwargs))
    ax.add_patch(FancyArrowPatch((6.5, 5), (7.5, 5), **arrow_kwargs))
    ax.add_patch(FancyArrowPatch((6.5, 2.5), (7.5, 4.5), **arrow_kwargs))
    
    # SSDSI → Outcome (mediation)
    mediation_arrow = FancyArrowPatch(
        (10, 5), (11, 5),
        arrowstyle=arrow_style,
        color="blue",
        lw=3
    )
    ax.add_patch(mediation_arrow)
    ax.text(10.5, 5.3, 'Mediation\n≥55%', ha='center', fontsize=9, color='blue')
    
    # Direct effects (dashed)
    for y in [7.5, 5, 2.5]:
        direct_arrow = FancyArrowPatch(
            (6.5, y), (11, 5),
            arrowstyle=arrow_style,
            color="gray",
            lw=1.5,
            linestyle='dashed'
        )
        ax.add_patch(direct_arrow)
    
    # Effect modification arrows
    for x in [5, 7, 9, 11]:
        mod_arrow = FancyArrowPatch(
            (x, 1.3), (x, 2),
            arrowstyle=arrow_style,
            color="purple",
            lw=1
        )
        ax.add_patch(mod_arrow)
    
    # Feedback loop
    feedback = FancyArrowPatch(
        (12, 3.5), (2, 3.5),
        arrowstyle=arrow_style,
        color="red",
        lw=1.5,
        linestyle='dotted',
        connectionstyle="arc3,rad=-.3"
    )
    ax.add_patch(feedback)
    ax.text(7, 2, 'Amplification Loop', ha='center', fontsize=9, 
            color='red', style='italic')


def _add_framework_legend(ax, colors):
    """Add legend explaining components (≤50 lines)."""
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, fc=colors['vulnerability'], 
                          edgecolor='black', label='Mental Health Vulnerability'),
        mpatches.Rectangle((0, 0), 1, 1, fc=colors['exposure'], 
                          edgecolor='black', label='SSD Pattern Exposures'),
        mpatches.Rectangle((0, 0), 1, 1, fc=colors['mediator'], 
                          edgecolor='black', label='Mediator (SSDSI)'),
        mpatches.Rectangle((0, 0), 1, 1, fc=colors['outcome'], 
                          edgecolor='black', label='Healthcare Outcomes'),
        mpatches.Rectangle((0, 0), 1, 1, fc=colors['moderator'], 
                          edgecolor='black', label='Effect Modifiers')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    # Add hypothesis labels
    ax.text(5.25, 8.5, 'H1', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="yellow"))
    ax.text(5.25, 6, 'H2', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="yellow"))
    ax.text(5.25, 3.5, 'H3', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="yellow"))
    ax.text(8.75, 3, 'H4', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="yellow"))
    ax.text(7, 0.3, 'H5', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.3", facecolor="yellow"))


if __name__ == "__main__":
    # Generate conceptual framework
    output_path = create_conceptual_framework()
    print(f"✓ Conceptual framework diagram created: {output_path}")