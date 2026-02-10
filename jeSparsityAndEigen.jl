using MatrixMarket
using SparseArrays
using Arpack
using LinearAlgebra
using Plots

# 1. Define the filename
filename = "sparse_Abx_data_A.mtx"
lplot_sparsity = false

# 2. Read the matrix
# mmread automatically detects the format and returns a SparseMatrixCSC
println("Reading matrix from $filename ...")
A = MatrixMarket.mmread(filename)

# Display basic info about the matrix
n_rows, n_cols = size(A)
n_nz = nnz(A)
println("Matrix Dimensions: $n_rows x $n_cols")
println("Non-zeros: $n_nz")
println("Density: $(round(n_nz / (n_rows * n_cols) * 100, digits=4))%")

# 3. Plot the sparsity pattern
# The 'spy' function is optimized for this exact purpose.
# We reverse the y-axis so row 1 is at the top (standard matrix visualization).

if lplot_sparsity
    
    println("Generating sparsity plot...")
    p = spy(A,
            markersize = 1,          # Adjust based on matrix size (smaller for large matrices)
            markerstrokewidth = 0,   # Removes outline on points for cleaner look
            color = :blues,          # Color gradient
            title = "Sparsity Pattern of A",
            xlabel = "Column Index",
            ylabel = "Row Index",
            legend = false,
            yflip = true             # Ensure row 1 is at the top
            )

    # 4. Save the figure
    output_file = "sparsity_pattern.png"
    savefig(p, output_file)
    println("Plot saved to $output_file")

    # If running interactively, this will display the plot window
    #display(p)
end

# 1. Read the Matrix
filename = "A.mtx"
println("Reading matrix from $filename ...")
A = MatrixMarket.mmread(filename)

# Check matrix properties
N = size(A, 1)
println("Matrix size: $N x $N")
println("Non-zeros: $(nnz(A))")

# 2. Check Symmetry
# Spectral element matrices are often symmetric positive definite.
# Knowing this helps the solver select the optimal algorithm (Lanczos vs Arnoldi).
is_sym = issymmetric(A)
println("Matrix is symmetric: $is_sym")

# 3. Estimate Eigenvalues using 'eigs'
# 'nev': Number of eigenvalues to compute
# 'which': Which eigenvalues to target (:LM = Largest Magnitude, :SM = Smallest Magnitude, :LR = Largest Real, etc.)
# 'tol': Convergence tolerance
println("Computing top 6 eigenvalues with largest magnitude...")

evals, evecs = eigs(A; 
    nev = 6,                 # Calculate only the top 6 eigenvalues
    which = :LM,             # Target: Largest Magnitude
    tol = 1e-6,              # Tolerance
    maxiter = 300            # Max iterations
)

# 4. Display Results
println("\n--- Results ---")
for (i, val) in enumerate(evals)
    println("Eigenvalue $i: $val")
    # To access the eigenvector: evecs[:, i]
end

# Example: Check the residual for the first eigenpair to verify accuracy
# Residual = || A*v - lambda*v ||
lambda_1 = evals[1]
v_1 = evecs[:, 1]
residual = norm(A * v_1 - lambda_1 * v_1)
println("\nResidual for first eigenpair: $residual")
