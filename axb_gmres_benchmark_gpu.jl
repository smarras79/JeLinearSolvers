# axb_gmres_benchmark_gpu.jl
#
# GPU GMRES benchmark: unpreconditioned vs ILU(0)-preconditioned.
#
# Target: numerically non-symmetric sparse matrices with symmetric
# sparsity pattern and O(M) nonzeros (very sparse, ~3-5 nnz per row).
#
# ILU(0) is computed entirely on GPU via CUSPARSE ilu02!, which
# stores L and U in the same CSR structure as A (zero fill-in).
# Triangular solves use native CUSPARSE kernels (exact, not iterative).
#
# Usage:
#   julia --project=. axb_gmres_benchmark_gpu.jl [options] [A.mtx b.mtx [x.mtx]]
#
# Options: --maxiter N, --rtol VAL, --precision TYPE (see cli_args.jl)
#
# Timing protocol:
#   - Warm-up solve excludes JIT/compilation overhead
#   - CUDA.@sync ensures GPU work completes before timer stops
#   - Multiple runs (nruns=3); best wall-clock time reported
#   - ILU setup time reported separately from solve time

using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
using Krylov
using Printf, Dates, JSON
using MatrixMarket

include("cli_args.jl")

# ─────────────────────────────────────────────────────────────────────
# GPU ILU(0) Preconditioner via CUSPARSE csrilu02
# ─────────────────────────────────────────────────────────────────────
# CUSPARSE ilu02! computes an incomplete LU factorization with zero
# fill-in directly on the GPU.  L (unit lower) and U (upper) are
# packed into a single CSR matrix.  The triangular solves use
# optimized CUSPARSE kernels — no Jacobi approximation needed.
#
# Requirement: the CSR matrix must have storage for every (i,j) such
# that BOTH (i,j) and (j,i) belong to the sparsity pattern.  For
# matrices that already have symmetric sparsity (e.g. FD/FE stencils)
# this holds automatically.
# ─────────────────────────────────────────────────────────────────────
struct GPUILU0{T}
    LU::CuSparseMatrixCSR{T, Int32}
    temp::CuVector{T}
end

function GPUILU0(A_gpu::CuSparseMatrixCSR{T}) where T
    LU = copy(A_gpu)
    CUSPARSE.ilu02!(LU)
    return GPUILU0{T}(LU, CuVector{T}(undef, size(A_gpu, 1)))
end

function LinearAlgebra.ldiv!(y, P::GPUILU0, x)
    # Forward solve: L z = x   (L = unit lower triangular part of LU)
    ldiv!(P.temp, UnitLowerTriangular(P.LU), x)
    # Backward solve: U y = z  (U = upper triangular part of LU)
    ldiv!(y, UpperTriangular(P.LU), P.temp)
    return y
end

# ─────────────────────────────────────────────────────────────────────
# Test Problem: 2D Convection-Diffusion (non-symmetric values,
# symmetric 5-point stencil sparsity, ~5 nnz per row ≈ O(M))
# ─────────────────────────────────────────────────────────────────────
# PDE:  -ν ∇²u + c · ∇u = f   on [0,1]²
#
# The 5-point FD stencil connects each interior node to its four
# neighbours, giving a symmetric sparsity pattern.  The convection
# term (central differences) makes the VALUES non-symmetric:
#     a_{k, k-1} = -1/h² - cx/(2h)
#     a_{k, k+1} = -1/h² + cx/(2h)
# so a_{k, k-1} ≠ a_{k+1, k} when cx ≠ 0.
# ─────────────────────────────────────────────────────────────────────
function create_nonsym_convdiff(M, PREC)
    nx = ny = Int(ceil(sqrt(M)))
    n = nx * ny
    h = one(PREC) / PREC(nx + 1)

    # Convection velocities — non-zero values break value symmetry
    cx = PREC(100)
    cy = PREC(50)

    rows = Int[]
    cols = Int[]
    vals = PREC[]

    for j in 1:ny, i in 1:nx
        k = i + (j - 1) * nx

        # Diagonal: diffusion (4/h²) + upwind stabilisation
        diag_val = PREC(4) / h^2 + abs(cx) / h + abs(cy) / h
        push!(rows, k); push!(cols, k); push!(vals, diag_val)

        # West (i-1)
        if i > 1
            push!(rows, k); push!(cols, k - 1)
            push!(vals, -one(PREC) / h^2 - cx / (2h))
        end
        # East (i+1)
        if i < nx
            push!(rows, k); push!(cols, k + 1)
            push!(vals, -one(PREC) / h^2 + cx / (2h))
        end
        # South (j-1)
        if j > 1
            push!(rows, k); push!(cols, k - nx)
            push!(vals, -one(PREC) / h^2 - cy / (2h))
        end
        # North (j+1)
        if j < ny
            push!(rows, k); push!(cols, k + nx)
            push!(vals, -one(PREC) / h^2 + cy / (2h))
        end
    end

    A = sparse(rows, cols, vals, n, n)
    x_true = ones(PREC, n)
    b = A * x_true
    return A, b, x_true, n
end

# ─────────────────────────────────────────────────────────────────────
# Utility: ensure symmetric sparsity pattern
# ─────────────────────────────────────────────────────────────────────
# If loading from file, the matrix might be missing a few (j,i)
# entries where (i,j) exists.  This adds structural zeros so that
# the pattern of A equals the pattern of A^T — a requirement for
# CUSPARSE ilu02.
# ─────────────────────────────────────────────────────────────────────
function ensure_symmetric_sparsity(A::SparseMatrixCSC{T}) where T
    I_a, J_a, V_a = findnz(A)
    # Append the transposed index pairs with zero values.
    # sparse() sums duplicates, so original entries keep their
    # values (a_{ij} + 0 = a_{ij}) while missing (j,i) entries
    # get a structural zero.
    I_all = vcat(I_a, J_a)
    J_all = vcat(J_a, I_a)
    V_all = vcat(V_a, zeros(T, length(V_a)))
    return sparse(I_all, J_all, V_all, size(A)...)
end

# ─────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────
function benchmark_gmres(A_cpu, b_cpu, x_true, PREC;
                         maxiter::Int = 500,
                         rtol::Float64 = 1e-8,
                         gmres_memory::Int = 100,
                         nruns::Int = 3)

    n = size(A_cpu, 1)
    nnz_A = SparseArrays.nnz(A_cpu)

    # ── Header ──────────────────────────────────────────────────────
    println("\n" * "="^72)
    println("  GPU GMRES BENCHMARK: unpreconditioned vs ILU(0)")
    println("="^72)
    @printf("  Matrix size:    %d x %d\n", n, n)
    @printf("  Nonzeros:       %d  (%.1f per row, density %.4f%%)\n",
            nnz_A, nnz_A / n, 100 * nnz_A / (Float64(n)^2))
    @printf("  Precision:      %s\n", PREC)
    @printf("  GMRES memory:   %d vectors (restart = true)\n", gmres_memory)
    @printf("  Max iterations: %d\n", maxiter)
    @printf("  Tolerance:      rtol = %.1e, atol = %.1e\n", rtol, rtol)
    @printf("  Timed runs:     %d (best wall-clock time reported)\n", nruns)

    is_sym_vals = issymmetric(A_cpu)
    @printf("  Value symmetry: %s\n", is_sym_vals ? "symmetric" : "non-symmetric")

    # ── GPU transfer ────────────────────────────────────────────────
    println("\nTransferring to GPU (CSR format)...")
    CUDA.@sync A_gpu = CuSparseMatrixCSR(A_cpu)
    CUDA.@sync b_gpu = CuArray(b_cpu)
    println("  done.")

    solve_rtol = PREC(rtol)
    solve_atol = PREC(rtol)

    # ── Warm-up (exclude JIT compilation from timings) ──────────────
    println("\nWarm-up solve (5 iterations, no preconditioner)...")
    CUDA.@sync gmres(A_gpu, b_gpu;
                     itmax=5, restart=true, memory=20, verbose=0)
    println("  done.")

    # ════════════════════════════════════════════════════════════════
    #  [1/2] GMRES WITHOUT preconditioner
    # ════════════════════════════════════════════════════════════════
    println("\n" * "-"^72)
    println("  [1/2] GMRES — no preconditioner")
    println("-"^72)

    best_t_nopre = Inf
    stats_nopre  = nothing
    x_nopre      = nothing

    for run in 1:nruns
        CUDA.@sync t = @elapsed begin
            xr, sr = gmres(A_gpu, b_gpu;
                           atol=solve_atol, rtol=solve_rtol,
                           restart=true, memory=gmres_memory,
                           itmax=maxiter, verbose=0, history=true)
        end
        if t < best_t_nopre
            best_t_nopre = t
            stats_nopre  = sr
            x_nopre      = xr
        end
        @printf("    run %d/%d: %.4f s  (%d iters, %s)\n",
                run, nruns, t, sr.niter,
                sr.solved ? "converged" : "NOT converged")
    end

    err_nopre = (x_true !== nothing
                 ? norm(Array(x_nopre) - x_true) / norm(x_true) : NaN)
    res_nopre = (!isempty(stats_nopre.residuals)
                 ? stats_nopre.residuals[end] : NaN)

    @printf("\n  Best time:       %10.4f s  (%10.2f ms)\n",
            best_t_nopre, best_t_nopre * 1000)
    @printf("  Iterations:      %10d\n", stats_nopre.niter)
    @printf("  Final residual:  %10.2e\n", res_nopre)
    @printf("  Relative error:  %10.2e\n", err_nopre)
    @printf("  Converged:       %10s\n",
            stats_nopre.solved ? "yes" : "NO")

    # ════════════════════════════════════════════════════════════════
    #  [2/2] GMRES WITH ILU(0) preconditioner  (CUSPARSE ilu02)
    # ════════════════════════════════════════════════════════════════
    println("\n" * "-"^72)
    println("  [2/2] GMRES — ILU(0) preconditioner (CUSPARSE)")
    println("-"^72)

    # ── ILU(0) setup (timed separately) ─────────────────────────────
    println("  Computing ILU(0) on GPU...")
    CUDA.@sync t_ilu_setup = @elapsed begin
        P = GPUILU0(A_gpu)
    end
    @printf("  ILU(0) setup time: %.4f s  (%.2f ms)\n",
            t_ilu_setup, t_ilu_setup * 1000)

    # Sanity check: apply preconditioner to a test vector
    test_x = CUDA.ones(PREC, n)
    test_y = similar(test_x)
    ldiv!(test_y, P, test_x)
    test_vals = Array(test_y)
    ilu_ok = !(any(isnan, test_vals) || any(isinf, test_vals))

    if !ilu_ok
        println("  WARNING: ILU(0) produces NaN/Inf — preconditioned solve skipped.")
        println("  (Matrix may need diagonal scaling or a shift; see axb_gmres_iLU_sparse_hybrid.jl)")
        println("="^72)
        return Dict(
            "metadata" => Dict("error" => "ILU(0) diverged"),
            "no_preconditioner" => Dict(
                "iterations"  => stats_nopre.niter,
                "solve_time_ms" => best_t_nopre * 1000,
                "converged"   => stats_nopre.solved))
    end
    println("  ILU(0) sanity check: OK")

    # ── Warm-up with preconditioner ─────────────────────────────────
    println("  Warm-up solve (5 iterations, ILU(0))...")
    CUDA.@sync gmres(A_gpu, b_gpu; M=P, ldiv=true,
                     itmax=5, restart=true, memory=20, verbose=0)

    # ── Timed runs ──────────────────────────────────────────────────
    best_t_ilu = Inf
    stats_ilu  = nothing
    x_ilu      = nothing

    for run in 1:nruns
        CUDA.@sync t = @elapsed begin
            xr, sr = gmres(A_gpu, b_gpu;
                           M=P, ldiv=true,
                           atol=solve_atol, rtol=solve_rtol,
                           restart=true, memory=gmres_memory,
                           itmax=maxiter, verbose=0, history=true)
        end
        if t < best_t_ilu
            best_t_ilu = t
            stats_ilu  = sr
            x_ilu      = xr
        end
        @printf("    run %d/%d: %.4f s  (%d iters, %s)\n",
                run, nruns, t, sr.niter,
                sr.solved ? "converged" : "NOT converged")
    end

    err_ilu = (x_true !== nothing
               ? norm(Array(x_ilu) - x_true) / norm(x_true) : NaN)
    res_ilu = (!isempty(stats_ilu.residuals)
               ? stats_ilu.residuals[end] : NaN)

    @printf("\n  Best solve time:     %10.4f s  (%10.2f ms)\n",
            best_t_ilu, best_t_ilu * 1000)
    @printf("  Total (setup+solve): %10.4f s  (%10.2f ms)\n",
            t_ilu_setup + best_t_ilu, (t_ilu_setup + best_t_ilu) * 1000)
    @printf("  Iterations:          %10d\n", stats_ilu.niter)
    @printf("  Final residual:      %10.2e\n", res_ilu)
    @printf("  Relative error:      %10.2e\n", err_ilu)
    @printf("  Converged:           %10s\n",
            stats_ilu.solved ? "yes" : "NO")

    # ════════════════════════════════════════════════════════════════
    #  COMPARISON SUMMARY
    # ════════════════════════════════════════════════════════════════
    iter_ratio    = stats_ilu.niter > 0  ? stats_nopre.niter / stats_ilu.niter           : NaN
    solve_speedup = best_t_ilu > 0       ? best_t_nopre / best_t_ilu                     : NaN
    total_speedup = (t_ilu_setup + best_t_ilu) > 0 ? best_t_nopre / (t_ilu_setup + best_t_ilu) : NaN

    println("\n" * "="^72)
    println("  COMPARISON SUMMARY")
    println("="^72)
    @printf("  %-32s %12s %12s\n", "", "No Precond", "ILU(0)")
    println("  " * "-"^56)
    @printf("  %-32s %10d %12d\n",
            "Iterations", stats_nopre.niter, stats_ilu.niter)
    @printf("  %-32s %9.2f ms %9.2f ms\n",
            "Solve time (best of $nruns)",
            best_t_nopre * 1000, best_t_ilu * 1000)
    @printf("  %-32s %12s %9.2f ms\n",
            "ILU(0) setup time", "---", t_ilu_setup * 1000)
    @printf("  %-32s %9.2f ms %9.2f ms\n",
            "Total time",
            best_t_nopre * 1000, (t_ilu_setup + best_t_ilu) * 1000)
    @printf("  %-32s %10.2e %12.2e\n",
            "Final residual", res_nopre, res_ilu)
    @printf("  %-32s %10.2e %12.2e\n",
            "Relative error ||x-x*||/||x*||", err_nopre, err_ilu)
    @printf("  %-32s %10s %12s\n",
            "Converged",
            stats_nopre.solved ? "yes" : "NO",
            stats_ilu.solved   ? "yes" : "NO")
    println("  " * "-"^56)
    @printf("  Iteration reduction:         %.2fx\n", iter_ratio)
    @printf("  Solve-time speedup:          %.2fx\n", solve_speedup)
    @printf("  Total-time speedup:          %.2fx  (incl. ILU setup)\n",
            total_speedup)
    println("="^72)

    # ── Build results dictionary ────────────────────────────────────
    return Dict(
        "metadata" => Dict(
            "date"              => string(Dates.now()),
            "matrix_size"       => n,
            "matrix_nnz"        => nnz_A,
            "nnz_per_row"       => nnz_A / n,
            "precision"         => string(PREC),
            "gmres_memory"      => gmres_memory,
            "maxiter"           => maxiter,
            "rtol"              => rtol,
            "nruns"             => nruns,
            "symmetric_values"  => is_sym_vals
        ),
        "no_preconditioner" => Dict(
            "iterations"       => stats_nopre.niter,
            "solve_time_ms"    => best_t_nopre * 1000,
            "final_residual"   => res_nopre,
            "relative_error"   => err_nopre,
            "converged"        => stats_nopre.solved,
            "residual_history" => collect(stats_nopre.residuals)
        ),
        "ilu0_preconditioner" => Dict(
            "iterations"       => stats_ilu.niter,
            "setup_time_ms"    => t_ilu_setup * 1000,
            "solve_time_ms"    => best_t_ilu * 1000,
            "total_time_ms"    => (t_ilu_setup + best_t_ilu) * 1000,
            "final_residual"   => res_ilu,
            "relative_error"   => err_ilu,
            "converged"        => stats_ilu.solved,
            "residual_history" => collect(stats_ilu.residuals)
        ),
        "speedup" => Dict(
            "iteration_reduction" => iter_ratio,
            "solve_time_speedup"  => solve_speedup,
            "total_time_speedup"  => total_speedup
        )
    )
end

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
opts = parse_commandline_args(;
    default_maxiter   = 500,
    default_rtol      = 1e-8,
    default_precision = Float64)

PREC = opts.precision

println("="^72)
println("  GPU GMRES PRECONDITIONED BENCHMARK")
println("  $(Dates.now())")
println("="^72)

# ── Determine data source ──────────────────────────────────────────
if length(opts.positional) >= 2
    file_A = opts.positional[1]
    file_b = opts.positional[2]
    file_x = length(opts.positional) >= 3 ? opts.positional[3] : nothing
elseif isfile("sparse_Abx_data_A.mtx")
    file_A = "sparse_Abx_data_A.mtx"
    file_b = "sparse_Abx_data_b.mtx"
    file_x = isfile("sparse_Abx_data_x.mtx") ? "sparse_Abx_data_x.mtx" : nothing
else
    file_A = nothing
    file_b = nothing
    file_x = nothing
end

A_cpu, b_cpu, x_true, n = if file_A !== nothing
    # ── Load from Matrix Market files ──────────────────────────────
    isfile(file_A) || error("File not found: $file_A")
    isfile(file_b) || error("File not found: $file_b")

    println("\nLoading matrix from files...")
    println("  A: $file_A")
    println("  b: $file_b")

    A_raw = MatrixMarket.mmread(file_A)
    I_idx, J_idx, V = findnz(A_raw)
    A = sparse(I_idx, J_idx, Vector{PREC}(V), size(A_raw)...)
    b = Vector{PREC}(vec(MatrixMarket.mmread(file_b)))

    xt = if file_x !== nothing && isfile(file_x)
        println("  x: $file_x")
        Vector{PREC}(vec(MatrixMarket.mmread(file_x)))
    else
        nothing
    end

    # Ensure symmetric sparsity for CUSPARSE ilu02!
    nnz_before = SparseArrays.nnz(A)
    A_sym = ensure_symmetric_sparsity(A)
    nnz_after = SparseArrays.nnz(A_sym)
    if nnz_after > nnz_before
        @printf("  Symmetrised sparsity: %d -> %d stored entries (+%d structural zeros)\n",
                nnz_before, nnz_after, nnz_after - nnz_before)
    else
        println("  Sparsity pattern already symmetric.")
    end

    A_sym, b, xt, size(A, 1)
else
    # ── Generate test problem ──────────────────────────────────────
    println("\nNo matrix files found. Generating convection-diffusion test problem...")
    M = 10000   # 100x100 grid -> 10 000 unknowns, ~50 000 nnz
    A, b, xt, actual_n = create_nonsym_convdiff(M, PREC)
    @printf("  Grid:       %d x %d  (%d unknowns)\n",
            Int(sqrt(actual_n)), Int(sqrt(actual_n)), actual_n)
    @printf("  Nonzeros:   %d  (%.1f per row)\n",
            SparseArrays.nnz(A), SparseArrays.nnz(A) / actual_n)
    @printf("  Symmetric values: %s\n", issymmetric(A) ? "yes" : "no")
    A, b, xt, actual_n
end

# ── Run benchmark ──────────────────────────────────────────────────
results = benchmark_gmres(A_cpu, b_cpu, x_true, PREC;
                          maxiter      = opts.maxiter,
                          rtol         = opts.rtol,
                          gmres_memory = 100,
                          nruns        = 3)

# ── Save results ───────────────────────────────────────────────────
output_file = "results_benchmark.json"
open(output_file, "w") do f
    JSON.print(f, results, 2)
end
println("\nResults saved to $output_file")
