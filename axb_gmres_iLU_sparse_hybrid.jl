using SparseArrays, LinearAlgebra, Random, Printf, Dates
using CUDA, CUDA.CUSPARSE
using Krylov 
using MatrixMarket
using Plots

# ===== ROBUST NATIVE GPU ILU PRECONDITIONER =====
struct GPUSparseILU{T}
    LU::CuSparseMatrixCSR{T, Int32}
    temp_z::CuVector{T} 
end

function GPUSparseILU(A_gpu::CuSparseMatrixCSR{T}) where T
    n = size(A_gpu, 1)
    # Use a copy to avoid mutating the original matrix A
    LU = copy(A_gpu)
    
    # Perform the incomplete LU factorization on the GPU
    # ilu02! is the most robust entry point for modern CUSPARSE
    CUSPARSE.ilu02!(LU) 
    
    temp_z = CuVector{T}(undef, n)
    return GPUSparseILU{T}(LU, temp_z)
end

function LinearAlgebra.ldiv!(y, P::GPUSparseILU, x)
    # L * z = x then U * y = z
    # These wrappers trigger optimized NVIDIA triangular solve kernels
    ldiv!(P.temp_z, UnitLowerTriangular(P.LU), x)
    ldiv!(y, UpperTriangular(P.LU), P.temp_z)
    return y
end

# ===== FILE FORMAT DETECTION =====
function detect_format(filepath)
    if endswith(lowercase(filepath), ".mtx")
        return :matrixmarket
    end
    open(filepath) do f
        line = readline(f)
        if startswith(line, "%%MatrixMarket")
            return :matrixmarket
        end
    end
    return :text_sparse
end

# ===== SPARSE MATRIX LOADING =====
# Load a sparse matrix from file. Supports:
#   - MatrixMarket (.mtx) format
#   - Plain text sparse coordinate (COO) format:
#       Comment lines start with '#' or '%'
#       Each data line: row col value (1-indexed)
#       If minimum index is 0, indices are shifted to 1-based automatically.
function load_sparse_matrix(filepath, PREC)
    if !isfile(filepath)
        error("Matrix file not found: $filepath")
    end

    fmt = detect_format(filepath)
    println("  Loading matrix from: $filepath (format: $fmt)")

    if fmt == :matrixmarket
        A_raw = MatrixMarket.mmread(filepath)
        I, J, V = findnz(A_raw)
        return sparse(I, J, Vector{PREC}(V), size(A_raw)...)
    end

    # Plain text COO format
    rows = Int[]
    cols = Int[]
    vals = PREC[]

    open(filepath) do f
        for line in eachline(f)
            stripped = strip(line)
            if isempty(stripped) || startswith(stripped, "#") || startswith(stripped, "%")
                continue
            end
            parts = split(stripped)
            length(parts) >= 3 || error("Expected 'row col value' but got: $stripped")
            push!(rows, parse(Int, parts[1]))
            push!(cols, parse(Int, parts[2]))
            push!(vals, parse(PREC, parts[3]))
        end
    end

    isempty(rows) && error("No data found in matrix file: $filepath")

    # Handle 0-based indexing: shift to 1-based if needed
    if minimum(rows) == 0 || minimum(cols) == 0
        println("  Detected 0-based indices, shifting to 1-based.")
        rows .+= 1
        cols .+= 1
    end

    return sparse(rows, cols, vals, maximum(rows), maximum(cols))
end

# ===== VECTOR LOADING =====
# Load a vector from file. Supports:
#   - MatrixMarket (.mtx) format
#   - Plain text format:
#       Comment lines start with '#' or '%'
#       Each data line: one value per line, OR index value per line
function load_vector(filepath, PREC)
    if !isfile(filepath)
        error("Vector file not found: $filepath")
    end

    fmt = detect_format(filepath)
    println("  Loading vector from: $filepath (format: $fmt)")

    if fmt == :matrixmarket
        return Vector{PREC}(vec(MatrixMarket.mmread(filepath)))
    end

    # Plain text format
    values = Tuple{Int, PREC}[]
    line_count = 0
    has_indices = nothing

    open(filepath) do f
        for line in eachline(f)
            stripped = strip(line)
            if isempty(stripped) || startswith(stripped, "#") || startswith(stripped, "%")
                continue
            end
            parts = split(stripped)
            line_count += 1

            if has_indices === nothing
                has_indices = length(parts) >= 2
            end

            if has_indices && length(parts) >= 2
                idx = parse(Int, parts[1])
                val = parse(PREC, parts[2])
                push!(values, (idx, val))
            else
                val = parse(PREC, parts[1])
                push!(values, (line_count, val))
            end
        end
    end

    isempty(values) && error("No data found in vector file: $filepath")

    # Handle 0-based indexing
    if minimum(first.(values)) == 0
        println("  Detected 0-based indices, shifting to 1-based.")
        values = [(idx + 1, val) for (idx, val) in values]
    end

    n = maximum(first.(values))
    b = zeros(PREC, n)
    for (idx, val) in values
        b[idx] = val
    end
    return b
end

# ===== UNIFIED DATA LOADING =====
# Load a sparse linear system from files. The reference solution x is optional.
# Supports both MatrixMarket (.mtx) and plain text sparse formats.
function load_system(path_A, path_b, PREC; path_x=nothing)
    println("Loading sparse system from files...")
    A = load_sparse_matrix(path_A, PREC)
    b = load_vector(path_b, PREC)
    n = size(A, 1)

    if size(A, 1) != size(A, 2)
        error("Matrix A must be square, got $(size(A, 1))x$(size(A, 2))")
    end
    if length(b) != n
        error("Dimension mismatch: A is $(n)x$(n) but b has length $(length(b))")
    end

    xt = if path_x !== nothing && isfile(path_x)
        load_vector(path_x, PREC)
    else
        if path_x !== nothing
            println("  Warning: Reference solution file not found: $path_x (skipping)")
        end
        nothing
    end

    @printf("  System loaded: A is %dx%d with %d nonzeros\n", n, n, nnz(A))
    return A, b, xt, n
end

# ===== MAIN SOLVER WITH PRE-CONDITIONING STRATEGY =====
function solve_with_ilu(A_cpu, b_cpu, x_true, PREC)
    n = length(b_cpu)
    println("\n" * "="^70)
    @printf("Robust Solver: %d unknowns | Precision: %s\n", n, PREC)
    println("="^70)

    # --- 1. GPU TRANSFER ---
    println("Transferring system to GPU...")
    A_gpu = CuSparseMatrixCSR(A_cpu)
    b_gpu = CuArray(b_cpu)

    # --- 2. BUILD ILU(0) PRECONDITIONER DIRECTLY ---
    # Apply CUSPARSE ilu02! on the original matrix (no scaling/shifting).
    # If ilu02! fails (zero pivot), retry with a minimal diagonal perturbation
    # applied only to the preconditioner copy — the solve still uses original A.
    println("Building GPU-Native ILU(0) preconditioner...")
    P = try
        GPUSparseILU(A_gpu)
    catch e
        @printf("  ILU(0) failed on original matrix: %s\n", e)
        epsilon = PREC == Float64 ? 1e-10 : PREC(1e-6)
        @printf("  Retrying with diagonal perturbation eps=%.1e ...\n", epsilon)
        A_shifted_cpu = A_cpu + epsilon * sparse(PREC(1)*I, n, n)
        A_shifted_gpu = CuSparseMatrixCSR(A_shifted_cpu)
        GPUSparseILU(A_shifted_gpu)
    end

    # --- 3. LIVE CONVERGENCE PLOTTING ---
    p = plot(title="GMRES Convergence (ILU(0) preconditioned)",
             xlabel="Iteration", ylabel="Relative Residual",
             yaxis=:log10, label="||r||/||b||", lw=2, grid=true,
             color=:crimson)
    display(p)

    println("\nStarting GMRES (Restart Memory=150)...")

    t_solve = @elapsed begin
        x_gpu, stats = gmres(A_gpu, b_gpu;
                             M=P, ldiv=true,
                             restart=true, memory=150,
                             itmax=2500,
                             atol=1e-8, rtol=1e-8,
                             verbose=1,
                             callback = (solver) -> begin
                                 it = solver.stats.niter
                                 if it > 0 && it % 10 == 0 && !isempty(solver.stats.residuals)
                                     res = solver.stats.residuals[end] / solver.stats.residuals[1]
                                     push!(p, it, res)
                                     gui(p)
                                 end
                                 return false # Continue solving
                             end)
    end

    # --- 4. RESULTS ---
    x_cpu = Array(x_gpu)
    savefig("convergence_report.png")

    @printf("\nSolve Time: %.2f s | Iterations: %d\n", t_solve, stats.niter)
    if x_true !== nothing
        err = norm(x_cpu - x_true) / norm(x_true)
        @printf("Final Relative Error vs True: %.2e\n", err)
    else
        res = norm(A_cpu * x_cpu - b_cpu) / norm(b_cpu)
        @printf("Final Relative Residual ||Ax-b||/||b||: %.2e\n", res)
    end
    @printf("Solver Status: %s\n", stats.solved ? "CONVERGED" : "FAILED")

    return x_cpu, stats
end

# ===== EXECUTION =====
# Usage:
#   julia axb_gmres_iLU_sparse_hybrid.jl <file_A> <file_b> [file_x]
#
# Supported file formats:
#   - MatrixMarket (.mtx)
#   - Plain text sparse coordinate format:
#       Matrix files: each line is "row col value" (1-indexed)
#       Vector files: each line is "value" or "index value"
#       Lines starting with '#' or '%' are treated as comments.
#       0-based indices are auto-detected and shifted to 1-based.
#
# If no arguments are given, falls back to a synthetic Laplacian test case.

PRECISION = Float64

A, b, xt, n_actual = if length(ARGS) >= 2
    path_A = ARGS[1]
    path_b = ARGS[2]
    path_x = length(ARGS) >= 3 ? ARGS[3] : nothing
    load_system(path_A, path_b, PRECISION; path_x=path_x)
else
    # Laplacian Test Case (fallback when no files are provided)
    println("No input files specified. Running synthetic Laplacian test case.")
    nx = 585; n_t = nx^2
    I_idx, J_idx, V_idx = Int[], Int[], PRECISION[]
    for i in 1:n_t
        push!(I_idx, i); push!(J_idx, i); push!(V_idx, 4.0)
    end
    sparse(I_idx, J_idx, V_idx, n_t, n_t), ones(PRECISION, n_t), nothing, n_t
end

x_final, stats = solve_with_ilu(A, b, xt, PRECISION)

println("Saving solution to x_out.mtx...")
MatrixMarket.mmwrite("x_out.mtx", sparse(reshape(x_final, :, 1)))
