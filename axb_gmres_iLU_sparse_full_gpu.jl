using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
using IncompleteLU, Krylov, LinearOperators
using Printf
using Dates
using JSON
using MatrixMarket

include("cli_args.jl")

# ===== CREATE REALISTIC TEST PROBLEM =====
function create_convection_diffusion(n, PREC)
    """Convection-diffusion: harder problem that benefits from ILU"""
    nx = ny = Int(sqrt(n))
    actual_n = nx * ny
    
    rows = Int[]
    cols = Int[]
    vals = PREC[]
    
    # Convection velocity
    cx, cy = PREC(10.0), PREC(10.0)
    h = PREC(1.0) / nx
    
    for j in 1:ny
        for i in 1:nx
            k = i + (j-1)*nx
            
            # Diagonal (diffusion + stabilization)
            push!(rows, k); push!(cols, k); push!(vals, PREC(4.0)/h^2 + abs(cx)/h + abs(cy)/h)
            
            # Convection-diffusion stencil
            if i > 1
                push!(rows, k); push!(cols, k-1); push!(vals, -PREC(1.0)/h^2 - cx/(2*h))
            end
            if i < nx
                push!(rows, k); push!(cols, k+1); push!(vals, -PREC(1.0)/h^2 + cx/(2*h))
            end
            if j > 1
                push!(rows, k); push!(cols, k-nx); push!(vals, -PREC(1.0)/h^2 - cy/(2*h))
            end
            if j < ny
                push!(rows, k); push!(cols, k+nx); push!(vals, -PREC(1.0)/h^2 + cy/(2*h))
            end
        end
    end
    
    A = sparse(rows, cols, vals, actual_n, actual_n)
    x_true = ones(PREC, actual_n)
    b = A * x_true
    
    return A, b, x_true, actual_n
end

function create_2d_laplacian(n, PREC)
    nx = ny = Int(sqrt(n))
    actual_n = nx * ny
    
    rows = Int[]
    cols = Int[]
    vals = PREC[]
    
    println("Creating 2D Laplacian ($(nx)×$(ny) grid, $actual_n unknowns)...")
    
    for j in 1:ny
        for i in 1:nx
            k = i + (j-1)*nx
            
            push!(rows, k); push!(cols, k); push!(vals, PREC(4.0))
            
            if i > 1
                push!(rows, k); push!(cols, k-1); push!(vals, PREC(-1.0))
            end
            if i < nx
                push!(rows, k); push!(cols, k+1); push!(vals, PREC(-1.0))
            end
            if j > 1
                push!(rows, k); push!(cols, k-nx); push!(vals, PREC(-1.0))
            end
            if j < ny
                push!(rows, k); push!(cols, k+nx); push!(vals, PREC(-1.0))
            end
        end
    end
    
    A = sparse(rows, cols, vals, actual_n, actual_n)
    x_true = ones(PREC, actual_n)
    b = A * x_true
    
    return A, b, x_true, actual_n
end

# ===== FULLY GPU ILU PRECONDITIONER =====
struct FullyGPUILU{T,TM}
    L_gpu::TM              # Sparse L on GPU
    U_gpu::TM              # Sparse U on GPU
    L_diag_inv::CuVector{T}  # Precomputed 1/diag(L)
    U_diag_inv::CuVector{T}  # Precomputed 1/diag(U)
    temp1::CuVector{T}
    temp2::CuVector{T}
    jacobi_iters::Int      # Iterations for triangular solve
    omega::T               # Relaxation parameter
end

function FullyGPUILU(L_cpu::SparseMatrixCSC{T}, U_cpu::SparseMatrixCSC{T};
                     jacobi_iters=100, omega=T(0.9)) where T
    n = size(L_cpu, 1)
    threshold = eps(T) * T(1000)

    # Extract diagonals efficiently (single O(nnz) pass each)
    L_diag = diag(L_cpu)
    U_diag = diag(U_cpu)

    # Fix near-zero diagonals via sparse addition (no element-by-element mutation)
    L_fix = [abs(d) < threshold ? T(1.0) - d : T(0.0) for d in L_diag]
    U_fix = [abs(d) < threshold ? T(1.0) - d : T(0.0) for d in U_diag]
    L_fixed = any(!iszero, L_fix) ? L_cpu + spdiagm(0 => L_fix) : L_cpu
    U_fixed = any(!iszero, U_fix) ? U_cpu + spdiagm(0 => U_fix) : U_cpu

    # Compute inverse diagonals from the corrected values
    L_diag_inv_cpu = T(1) ./ (L_diag .+ L_fix)
    U_diag_inv_cpu = T(1) ./ (U_diag .+ U_fix)

    # Transfer to GPU as sparse
    L_gpu = CuSparseMatrixCSC(L_fixed)
    U_gpu = CuSparseMatrixCSC(U_fixed)
    L_diag_inv = CuArray(L_diag_inv_cpu)
    U_diag_inv = CuArray(U_diag_inv_cpu)
    
    FullyGPUILU{T,typeof(L_gpu)}(
        L_gpu, U_gpu,
        L_diag_inv, U_diag_inv,
        CUDA.zeros(T, n),
        CUDA.zeros(T, n),
        jacobi_iters,
        omega
    )
end

# Damped Jacobi iteration for sparse triangular solve on GPU
function solve_lower_jacobi!(y, L_sparse, L_diag_inv, x, temp, niter, omega)
    """
    Solve L*y = x using damped Jacobi iteration on GPU
    L*y = x  =>  (D + L_strict)*y = x
    Jacobi: y^{k+1} = D^{-1}(x - L_strict*y^k)
    Damped: y^{k+1} = (1-ω)*y^k + ω*D^{-1}(x - L*y^k)
    """
    # Initialize: y = D^{-1} * x
    y .= L_diag_inv .* x
    
    for iter in 1:niter
        # temp = L * y
        mul!(temp, L_sparse, y)
        # temp = x - L*y (residual)
        temp .= x .- temp
        # y = (1-ω)*y + ω*D^{-1}*residual
        y .= (1 - omega) .* y .+ omega .* (L_diag_inv .* temp)
    end
    
    return y
end

function solve_upper_jacobi!(y, U_sparse, U_diag_inv, x, temp, niter, omega)
    """
    Solve U*y = x using damped Jacobi iteration on GPU
    """
    # Initialize: y = D^{-1} * x
    y .= U_diag_inv .* x
    
    for iter in 1:niter
        # temp = U * y
        mul!(temp, U_sparse, y)
        # temp = x - U*y
        temp .= x .- temp
        # y = (1-ω)*y + ω*D^{-1}*residual
        y .= (1 - omega) .* y .+ omega .* (U_diag_inv .* temp)
    end
    
    return y
end

function LinearAlgebra.ldiv!(y, P::FullyGPUILU, x)
    """
    Apply ILU preconditioner: solve (L*U)*y = x
    All operations on GPU with sparse matrices
    """
    # Forward solve: L * temp = x (iterative on GPU)
    solve_lower_jacobi!(P.temp1, P.L_gpu, P.L_diag_inv, x, P.temp2, 
                       P.jacobi_iters, P.omega)
    
    # Backward solve: U * y = temp (iterative on GPU)
    solve_upper_jacobi!(y, P.U_gpu, P.U_diag_inv, P.temp1, P.temp2,
                       P.jacobi_iters, P.omega)
    
    return y
end

# ===== MIXED PRECISION ILU SOLVER =====
function solve_with_gpu_ilu(A_cpu, b_cpu, x_true, PREC, jacobi_iters, omega;
                            maxiter::Int=200, rtol::Float64=1e-8)
    n = length(b_cpu)

    println("\n" * "="^70)
    println("PRECISION: $PREC | maxiter: $maxiter | rtol: $rtol")
    println("="^70)
    
    nnz_A = SparseArrays.nnz(A_cpu)
    density = nnz_A / (n * n) * 100
    
    println("Matrix statistics:")
    println("  Size: $n × $n")
    println("  Nonzeros: $nnz_A")
    println("  Density: $(round(density, digits=2))%")
    
    # ===== COMPUTE ILU FACTORIZATION (on CPU) =====
    # τ controls drop tolerance: entries below τ are dropped from L,U
    # τ=0.0 keeps ALL fill-in (essentially full LU, hangs on large matrices)
    ilu_tau = PREC(0.01)
    println("\nComputing ILU factorization on CPU (τ=$ilu_tau)...")

    L_cpu, U_cpu = try
        fact = ilu(A_cpu, τ=ilu_tau)
        fact.L, fact.U
    catch e
        println("  WARNING: ILU failed: $e, using diagonal preconditioner")
        d_vals = diag(A_cpu)
        D_sqrt = sqrt.(max.(abs.(d_vals), eps(PREC)))
        sparse(Diagonal(D_sqrt)), sparse(Diagonal(D_sqrt))
    end
    
    nnz_L = SparseArrays.nnz(L_cpu)
    nnz_U = SparseArrays.nnz(U_cpu)
    
    println("  ✓ ILU completed")
    println("  L: $nnz_L nnz, U: $nnz_U nnz")
    println("  Memory vs dense: $(round(100*(1 - (nnz_L+nnz_U)/(n*n)), digits=1))% savings")
    
    # ===== TRANSFER EVERYTHING TO GPU =====
    println("\nTransferring to GPU...")
    A_gpu = CuSparseMatrixCSC(A_cpu)
    b_gpu = CuArray(b_cpu)

    # Build GPU ILU preconditioner; fall back to diagonal if Jacobi iteration diverges
    println("\nBuilding GPU ILU preconditioner...")
    P = FullyGPUILU(L_cpu, U_cpu, jacobi_iters=jacobi_iters, omega=omega)

    println("  - L, U: sparse on GPU")
    println("  - Triangular solve: Damped Jacobi ($jacobi_iters iters, ω=$omega)")

    # Test preconditioner — Jacobi triangular solve can diverge for some matrices
    test_x = CuArray(ones(PREC, n))
    test_y = similar(test_x)
    ldiv!(test_y, P, test_x)
    test_result = Array(test_y)

    use_ilu = true
    if any(isnan.(test_result)) || any(isinf.(test_result))
        println("  WARNING: GPU ILU preconditioner produces NaN/Inf (Jacobi iteration diverged)")
        println("  Falling back to diagonal (Jacobi) preconditioner...")
        # Simple diagonal preconditioner: M = diag(A)^{-1}
        a_diag = diag(A_cpu)
        d_cpu = [abs(v) > eps(PREC) ? PREC(1) / v : PREC(1) for v in a_diag]
        d_gpu = CuArray(d_cpu)
        # Wrap as a LinearOperator so Krylov.jl can use it
        use_ilu = false
    else
        println("  ✓ GPU ILU preconditioner test passed")
    end
    
    # ===== SOLVE WITHOUT PRECONDITIONER =====
    println("\n[1/2] Solving WITHOUT preconditioner...")

    solve_atol = PREC(rtol)
    solve_rtol = PREC(rtol)

    CUDA.@sync t_noprecond = @elapsed begin
        x_noprecond, stats_noprecond = gmres(A_gpu, b_gpu;
                                              atol=solve_atol,
                                              rtol=solve_rtol,
                                              restart=true,
                                              itmax=maxiter,
                                              verbose=0,
                                              history=true)
    end
    
    x_noprecond_cpu = Array(x_noprecond)
    err_noprecond = norm(x_noprecond_cpu - x_true) / norm(x_true)
    final_res_noprecond = length(stats_noprecond.residuals) > 0 ? 
                          stats_noprecond.residuals[end] : NaN
    
    println("  Iterations: $(stats_noprecond.niter)")
    println("  GPU time: $(round(t_noprecond*1000, digits=2)) ms")
    println("  Final residual: $(final_res_noprecond)")
    println("  Relative error: $(err_noprecond)")
    println("  Converged: $(stats_noprecond.solved)")
    
    # ===== SOLVE WITH PRECONDITIONER =====
    if use_ilu
        println("\n[2/2] Solving WITH fully GPU ILU preconditioner...")
    else
        println("\n[2/2] Solving WITH diagonal preconditioner (ILU diverged)...")
    end

    CUDA.@sync t_ilu = @elapsed begin
        if use_ilu
            x_ilu, stats_ilu = gmres(A_gpu, b_gpu;
                                     M=P,
                                     ldiv=true,
                                     atol=solve_atol,
                                     rtol=solve_rtol,
                                     restart=true,
                                     itmax=maxiter,
                                     verbose=0,
                                     history=true)
        else
            # Diagonal preconditioner: M*x = diag(A)^{-1} * x
            opM = LinearOperator(PREC, n, n, true, true,
                                 (y, v) -> (y .= d_gpu .* v))
            x_ilu, stats_ilu = gmres(A_gpu, b_gpu;
                                     M=opM,
                                     atol=solve_atol,
                                     rtol=solve_rtol,
                                     restart=true,
                                     itmax=maxiter,
                                     verbose=0,
                                     history=true)
        end
    end
    
    x_ilu_cpu = Array(x_ilu)
    err_ilu = norm(x_ilu_cpu - x_true) / norm(x_true)
    final_res_ilu = length(stats_ilu.residuals) > 0 ? 
                    stats_ilu.residuals[end] : NaN
    
    println("  Iterations: $(stats_ilu.niter)")
    println("  GPU time: $(round(t_ilu*1000, digits=2)) ms")
    println("  Final residual: $(final_res_ilu)")
    println("  Relative error: $(err_ilu)")
    println("  Converged: $(stats_ilu.solved)")
    
    # ===== SUMMARY =====
    println("\n" * "-"^70)
    println("PERFORMANCE SUMMARY")
    println("-"^70)
    
    iter_reduction = stats_ilu.niter > 0 ? stats_noprecond.niter / stats_ilu.niter : 1.0
    time_speedup = t_ilu > 0 ? t_noprecond / t_ilu : 1.0
    
    println("Iteration reduction: $(round(iter_reduction, digits=2))x")
    println("GPU time speedup: $(round(time_speedup, digits=2))x")
    
    if stats_ilu.niter < stats_noprecond.niter
        println("✓ GPU ILU preconditioning EFFECTIVE")
    else
        println("⚠ ILU did not reduce iterations")
    end
    
    return (
        noprecond_iters = stats_noprecond.niter,
        noprecond_time = t_noprecond,
        noprecond_error = err_noprecond,
        noprecond_converged = stats_noprecond.solved,
        ilu_iters = stats_ilu.niter,
        ilu_time = t_ilu,
        ilu_error = err_ilu,
        ilu_converged = stats_ilu.solved,
        iter_reduction = iter_reduction,
        time_speedup = time_speedup,
        matrix_size = n,
        matrix_nnz = nnz_A,
        ilu_nnz = nnz_L + nnz_U,
        jacobi_iters = jacobi_iters,
        omega = omega
    )
end

# ===== MAIN: MIXED PRECISION STUDY =====
opts = parse_commandline_args(; default_maxiter=200, default_rtol=1e-8, default_precision=Float64)

println("="^70)
println("FULLY GPU ILU PRECONDITIONING - MIXED PRECISION STUDY")
println("="^70)

# GPU-specific parameters
jacobi_iters = 100  # Increase for better accuracy (10, 50, 100)
omega = 0.9         # Damping factor (0.7-0.9)

# Determine data source: files or generated problem
use_files = length(opts.positional) >= 2

# If user specified --precision, run only that one; otherwise loop over all three
precision_was_specified = any(a -> a in ("--precision", "-p"), ARGS)
precisions_to_run = precision_was_specified ? [opts.precision] : [Float64, Float32, Float16]

println("\nGPU Solver Parameters:")
println("  Jacobi iterations: $jacobi_iters")
println("  Omega (damping): $omega")
println("  maxiter: $(opts.maxiter)")
println("  rtol: $(opts.rtol)")
println("  Precisions: $precisions_to_run")
if use_files
    println("  Input files: ", join(opts.positional, ", "))
end

results = Dict()

for PREC in precisions_to_run
    Random.seed!(42)

    if use_files
        # Load from Matrix Market files: A.mtx b.mtx [x.mtx]
        path_A = opts.positional[1]
        path_b = opts.positional[2]
        isfile(path_A) || error("File not found: $path_A")
        isfile(path_b) || error("File not found: $path_b")

        println("\nLoading Matrix Market files for $PREC ...")
        A_raw = MatrixMarket.mmread(path_A)
        I_idx, J_idx, V = findnz(A_raw)
        A_cpu = sparse(I_idx, J_idx, Vector{PREC}(V), size(A_raw)...)
        b_cpu = Vector{PREC}(vec(MatrixMarket.mmread(path_b)))
        actual_n = size(A_cpu, 1)

        if length(opts.positional) >= 3
            path_x = opts.positional[3]
            isfile(path_x) || error("File not found: $path_x")
            x_true = Vector{PREC}(vec(MatrixMarket.mmread(path_x)))
        else
            x_true = ones(PREC, actual_n)
        end
    else
        # Generate test problem
        n = 10000  # 100x100 grid
        problem_type = "convection_diffusion"
        if problem_type == "laplacian"
            A_cpu, b_cpu, x_true, actual_n = create_2d_laplacian(n, PREC)
        else
            A_cpu, b_cpu, x_true, actual_n = create_convection_diffusion(n, PREC)
        end
    end

    result = solve_with_gpu_ilu(A_cpu, b_cpu, x_true, PREC, jacobi_iters, PREC(omega);
                                maxiter=opts.maxiter, rtol=opts.rtol)
    results[PREC] = result

    println()
end

# ===== FINAL COMPARISON =====
println("="^70)
println("MIXED PRECISION COMPARISON - FULLY GPU ILU")
println("="^70)
println()

header = @sprintf("%-12s %10s %10s %10s %10s %10s", 
                  "Precision", "No-P Iter", "ILU Iter", "Speedup", "Time (ms)", "Rel Error")
println(header)
println("-"^70)

for PREC in precisions_to_run
    r = results[PREC]
    row = @sprintf("%-12s %10d %10d %10.2fx %10.2f %10.2e",
                   PREC,
                   r.noprecond_iters,
                   r.ilu_iters,
                   r.iter_reduction,
                   r.ilu_time * 1000,
                   r.ilu_error)
    println(row)
end

println("="^70)
println("\nKEY CHARACTERISTICS:")
println("• Fully GPU: ILU factors stored sparse on GPU")
println("• Triangular solve: Damped Jacobi iteration (GPU-parallelizable)")
println("• No CPU/GPU transfers during GMRES iterations")
println("• Trade-off: Approximate triangular solve vs GPU parallelism")
println("="^70)

# ===== EXPORT RESULTS FOR PDF GENERATION =====
println("\n" * "="^70)
println("GENERATING PDF REPORT...")
println("="^70)

# Prepare data for Python PDF generation
first_result = first(values(results))
problem_n = first_result.matrix_size
results_data = Dict(
    "metadata" => Dict(
        "date" => string(Dates.now()),
        "problem_type" => use_files ? join(opts.positional, ", ") : "convection_diffusion",
        "problem_size" => problem_n,
        "grid_size" => use_files ? "$(problem_n)" : "$(Int(sqrt(problem_n)))x$(Int(sqrt(problem_n)))",
        "method" => "Fully GPU ILU",
        "jacobi_iters" => jacobi_iters,
        "omega" => omega
    ),
    "results" => Dict(
        string(PREC) => Dict(
            "noprecond_iters" => r.noprecond_iters,
            "noprecond_time" => r.noprecond_time * 1000,  # Convert to ms
            "noprecond_error" => r.noprecond_error,
            "noprecond_converged" => r.noprecond_converged,
            "ilu_iters" => r.ilu_iters,
            "ilu_time" => r.ilu_time * 1000,  # Convert to ms
            "ilu_error" => r.ilu_error,
            "ilu_converged" => r.ilu_converged,
            "iter_reduction" => r.iter_reduction,
            "time_speedup" => r.time_speedup,
            "matrix_size" => r.matrix_size,
            "matrix_nnz" => r.matrix_nnz,
            "ilu_nnz" => r.ilu_nnz
        ) for (PREC, r) in results
    )
)

# Save to JSON for Python script
open("results_fullgpu.json", "w") do f
    JSON.print(f, results_data, 2)
end

println("Results saved to results_fullgpu.json")
println("Calling Python PDF generator...")
