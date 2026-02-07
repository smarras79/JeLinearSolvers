#!/bin/bash
# Combined script to run ILU study and generate PDF report

echo "======================================================================"
echo "Mixed Precision ILU Preconditioning Study with PDF Report Generation"
echo "======================================================================"
echo ""

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
python3 -c "import reportlab, matplotlib, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Python dependencies..."
    pip install reportlab matplotlib numpy --break-system-packages --quiet
fi

# Run Julia simulation
echo ""
echo "Step 1: Running Julia simulation..."
echo "----------------------------------------------------------------------"
julia /home/claude/axb_gmres_iLU_with_pdf.jl

# Check if Julia completed successfully
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Julia simulation failed!"
    exit 1
fi

# Generate PDF report
echo ""
echo "Step 2: Generating PDF report..."
echo "----------------------------------------------------------------------"
python3 /home/claude/generate_pdf_report.py

# Check if PDF was generated
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ SUCCESS: PDF report generated successfully!"
    echo "======================================================================"
    echo ""
    echo "Output file: /mnt/user-data/outputs/ilu_preconditioning_report.pdf"
    echo ""
else
    echo ""
    echo "ERROR: PDF generation failed!"
    exit 1
fi
