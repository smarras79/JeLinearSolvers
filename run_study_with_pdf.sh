#!/bin/bash
# Combined script to run ILU study and generate PDF report
#
# Usage: ./run_study_with_pdf.sh [options]
#   --maxiter, -m N       Maximum solver iterations (default: 2500)
#   --rtol, -r VAL        Relative convergence tolerance (default: 1e-8)
#   --precision, -p TYPE  Precision: Float64, Float32, Float16 (default: Float64)
#
# Examples:
#   ./run_study_with_pdf.sh --maxiter 5000 --rtol 1e-10 --precision Float32
#   ./run_study_with_pdf.sh -m 1000 -r 1e-6 -p Float64

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

# Run Julia simulation (pass through all CLI arguments)
echo ""
echo "Step 1: Running Julia simulation..."
echo "----------------------------------------------------------------------"
julia --project=. ./axb_gmres_iLU_sparse_hybrid.jl "$@"

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
python3 ./generate_pdf_report.py

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
