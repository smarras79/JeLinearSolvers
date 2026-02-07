#!/usr/bin/env python3
"""
Generate PDF report for Mixed Precision ILU Preconditioning Study
"""

import json
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def load_results(json_file="/home/claude/results.json"):
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_plots(results_data):
    """Create comparison plots"""
    precisions = ['Float64', 'Float32', 'Float16']
    
    # Extract data
    noprecond_iters = [results_data['results'][p]['noprecond_iters'] for p in precisions]
    ilu_iters = [results_data['results'][p]['ilu_iters'] for p in precisions]
    noprecond_time = [results_data['results'][p]['noprecond_time'] for p in precisions]
    ilu_time = [results_data['results'][p]['ilu_time'] for p in precisions]
    errors = [results_data['results'][p]['ilu_error'] for p in precisions]
    speedups = [results_data['results'][p]['iter_reduction'] for p in precisions]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Plot 1: Iteration Comparison
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(precisions))
    width = 0.35
    ax1.bar(x - width/2, noprecond_iters, width, label='No Precond', color='#e74c3c')
    ax1.bar(x + width/2, ilu_iters, width, label='ILU Precond', color='#3498db')
    ax1.set_xlabel('Precision', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Iterations', fontsize=11, fontweight='bold')
    ax1.set_title('Iteration Count Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(precisions)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Time Comparison
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(x - width/2, noprecond_time, width, label='No Precond', color='#e74c3c')
    ax2.bar(x + width/2, ilu_time, width, label='ILU Precond', color='#3498db')
    ax2.set_xlabel('Precision', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('Solution Time Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(precisions)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Speedup Factor
    ax3 = plt.subplot(2, 2, 3)
    colors_speedup = ['#27ae60' if s > 1.0 else '#e67e22' for s in speedups]
    ax3.bar(precisions, speedups, color=colors_speedup, alpha=0.8)
    ax3.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='No Improvement')
    ax3.set_xlabel('Precision', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
    ax3.set_title('ILU Iteration Reduction', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Relative Error
    ax4 = plt.subplot(2, 2, 4)
    ax4.semilogy(precisions, errors, marker='o', markersize=10, linewidth=2, color='#9b59b6')
    ax4.set_xlabel('Precision', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Relative Error (log scale)', fontsize=11, fontweight='bold')
    ax4.set_title('Solution Accuracy', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return '/home/claude/comparison_plots.png'

def generate_pdf(results_data, output_file="/mnt/user-data/outputs/ilu_preconditioning_report.pdf"):
    """Generate PDF report"""
    
    doc = SimpleDocTemplate(output_file, pagesize=letter,
                           topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # ==== TITLE PAGE ====
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("Mixed Precision ILU Preconditioning Study", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("GPU-Accelerated Sparse Linear Solver Performance Analysis", 
                          ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14,
                                       alignment=TA_CENTER, textColor=colors.HexColor('#7f8c8d'))))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Metadata
    meta = results_data['metadata']
    metadata_table = Table([
        ['Date:', meta['date']],
        ['Problem Type:', meta['problem_type'].replace('_', ' ').title()],
        ['Problem Size:', f"{meta['problem_size']} unknowns"],
        ['Grid:', meta['grid_size']],
        ['Method:', 'Hybrid ILU (CPU solve + GPU matvec)']
    ], colWidths=[2*inch, 3.5*inch])
    
    metadata_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(metadata_table)
    story.append(PageBreak())
    
    # ==== RESULTS SECTION ====
    story.append(Paragraph("Performance Results", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Results table
    precisions = ['Float64', 'Float32', 'Float16']
    table_data = [
        ['Precision', 'No-Prec\nIters', 'ILU\nIters', 'Speedup', 'Time (ms)', 'Rel. Error', 'Converged']
    ]
    
    for prec in precisions:
        r = results_data['results'][prec]
        converged_symbol = '✓' if r['ilu_converged'] else '✗'
        table_data.append([
            prec,
            str(r['noprecond_iters']),
            str(r['ilu_iters']),
            f"{r['iter_reduction']:.2f}×",
            f"{r['ilu_time']:.2f}",
            f"{r['ilu_error']:.2e}",
            converged_symbol
        ])
    
    results_table = Table(table_data, colWidths=[1.0*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.0*inch, 1.1*inch, 0.9*inch])
    
    results_table.setStyle(TableStyle([
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ==== VISUAL COMPARISONS ====
    story.append(Paragraph("Visual Comparison", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Generate and add plots
    plot_file = create_plots(results_data)
    img = Image(plot_file, width=6.5*inch, height=4.33*inch)
    story.append(img)
    story.append(PageBreak())
    
    # ==== ANALYSIS AND CONCLUSIONS ====
    story.append(Paragraph("Analysis and Key Findings", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Calculate insights
    avg_speedup = np.mean([results_data['results'][p]['iter_reduction'] for p in precisions])
    best_precision = max(precisions, key=lambda p: results_data['results'][p]['iter_reduction'])
    best_speedup = results_data['results'][best_precision]['iter_reduction']
    
    # Memory savings
    matrix_nnz = results_data['results']['Float64']['matrix_nnz']
    ilu_nnz = results_data['results']['Float64']['ilu_nnz']
    matrix_size = results_data['results']['Float64']['matrix_size']
    memory_savings = 100 * (1 - ilu_nnz / (matrix_size * matrix_size))
    
    findings = [
        f"<b>ILU Effectiveness:</b> Average iteration reduction of {avg_speedup:.2f}× across all precisions.",
        f"<b>Best Performance:</b> {best_precision} achieved {best_speedup:.2f}× speedup in iterations.",
        f"<b>Memory Efficiency:</b> ILU factorization achieves {memory_savings:.1f}% memory savings compared to dense storage.",
        f"<b>Sparsity:</b> Matrix has {matrix_nnz:,} non-zeros; ILU factors have {ilu_nnz:,} non-zeros total.",
        "<b>Hybrid Approach:</b> CPU performs exact sparse triangular solves while GPU handles matrix-vector products.",
        "<b>Precision Trade-off:</b> Lower precision (Float16) provides faster computation with acceptable accuracy for many applications.",
    ]
    
    for finding in findings:
        story.append(Paragraph(f"• {finding}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Recommendations", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    recommendations = [
        "Use <b>Float32</b> for balance between speed and accuracy in most engineering applications.",
        "ILU(0) preconditioner significantly reduces iterations for convection-diffusion problems.",
        "Hybrid CPU/GPU approach leverages strengths of both: exact triangular solves (CPU) + fast matvec (GPU).",
        "For production systems, monitor convergence and adjust tolerances based on precision used.",
    ]
    
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    # ==== FOOTER ====
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        f"<i>Report generated on {meta['date']}</i>",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9,
                      textColor=colors.grey, alignment=TA_CENTER)
    ))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF report generated: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Load results
    results_data = load_results()
    
    # Generate PDF
    pdf_file = generate_pdf(results_data)
    
    print("\n" + "="*70)
    print("PDF REPORT GENERATION COMPLETE")
    print("="*70)
    print(f"Output: {pdf_file}")
