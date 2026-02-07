using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
using IncompleteLU, Krylov
using Printf
using Dates
using JSON

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
    
    # Fix any zero diagonals
    L_fixed = copy(L_cpu)
    U_fixed = copy(U_cpu)
    
    for i in 1:n
        if abs(L_fixed[i,i]) < eps(T) * 1000
            L_fixed[i,i] = T(1.0)
        end
        if abs(U_fixed[i,i]) < eps(T) * 1000
            U_fixed[i,i] = T(1.0)
        end
    end
    
    # Extract diagonals on CPU
    L_diag_cpu = [L_fixed[i,i] for i in 1:n]
    U_diag_cpu = [U_fixed[i,i] for i in 1:n]
    L_diag_inv_cpu = T(1) ./ L_diag_cpu
    U_diag_inv_cpu = T(1) ./ U_diag_cpu
    
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
function solve_with_gpu_ilu(A_cpu, b_cpu, x_true, PREC, jacobi_iters, omega)
    n = length(b_cpu)
    
    println("\n" * "="^70)
    println("PRECISION: $PREC")
    println("="^70)
    
    nnz_A = SparseArrays.nnz(A_cpu)
    density = nnz_A / (n * n) * 100
    
    println("Matrix statistics:")
    println("  Size: $n × $n")
    println("  Nonzeros: $nnz_A")
    println("  Density: $(round(density, digits=2))%")
    
    # ===== COMPUTE ILU FACTORIZATION (on CPU) =====
    println("\nComputing ILU(0) factorization on CPU...")
    
    L_cpu, U_cpu = try
        fact = ilu(A_cpu, τ=PREC(0.0))
        fact.L, fact.U
    catch e
        println("  ⚠ ILU failed: $e, using diagonal preconditioner")
        D = [abs(A_cpu[i,i]) > eps(PREC) ? A_cpu[i,i] : PREC(1.0) for i in 1:n]
        D_sqrt = sqrt.(abs.(D))
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
    
    # Build fully GPU ILU preconditioner
    P = FullyGPUILU(L_cpu, U_cpu, jacobi_iters=jacobi_iters, omega=omega)
    
    println("  ✓ Fully GPU ILU preconditioner built")
    println("  - L, U: sparse on GPU")
    println("  - Triangular solve: Damped Jacobi ($jacobi_iters iters, ω=$omega)")
    println("  - All operations on GPU during solve")
    
    # Test preconditioner
    println("\nTesting GPU preconditioner...")
    test_x = CuArray(ones(PREC, n))
    test_y = similar(test_x)
    ldiv!(test_y, P, test_x)
    test_result = Array(test_y)
    
    if any(isnan.(test_result)) || any(isinf.(test_result))
        error("GPU Preconditioner produces NaN/Inf")
    end
    println("  ✓ GPU preconditioner test passed")
    
    # ===== SOLVE WITHOUT PRECONDITIONER =====
    println("\n[1/2] Solving WITHOUT preconditioner...")
    
    atol = PREC == Float64 ? 1e-8 : PREC == Float32 ? 1f-6 : Float16(1e-3)
    rtol = PREC == Float64 ? 1e-8 : PREC == Float32 ? 1f-6 : Float16(1e-3)
    
    CUDA.@sync t_noprecond = @elapsed begin
        x_noprecond, stats_noprecond = gmres(A_gpu, b_gpu; 
                                              atol=atol,
                                              rtol=rtol,
                                              restart=true,
                                              itmax=200,
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
    
    # ===== SOLVE WITH GPU ILU PRECONDITIONER =====
    println("\n[2/2] Solving WITH fully GPU ILU preconditioner...")
    
    CUDA.@sync t_ilu = @elapsed begin
        x_ilu, stats_ilu = gmres(A_gpu, b_gpu; 
                                 M=P,
                                 ldiv=true,
                                 atol=atol,
                                 rtol=rtol,
                                 restart=true,
                                 itmax=200,
                                 verbose=0,
                                 history=true)
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
println("="^70)
println("FULLY GPU ILU PRECONDITIONING - MIXED PRECISION STUDY")
println("="^70)

n = 10000  # 100×100 grid
problem_type = "convection_diffusion"  # or "laplacian"

# GPU-specific parameters
jacobi_iters = 100  # Increase for better accuracy (10, 50, 100)
omega = 0.9         # Damping factor (0.7-0.9)

println("\nGPU Solver Parameters:")
println("  Jacobi iterations: $jacobi_iters")
println("  Omega (damping): $omega")

results = Dict()

for PREC in [Float64, Float32, Float16]
    Random.seed!(42)
    
    if problem_type == "laplacian"
        A_cpu, b_cpu, x_true, actual_n = create_2d_laplacian(n, PREC)
    else
        A_cpu, b_cpu, x_true, actual_n = create_convection_diffusion(n, PREC)
    end
    
    result = solve_with_gpu_ilu(A_cpu, b_cpu, x_true, PREC, jacobi_iters, PREC(omega))
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

for PREC in [Float64, Float32, Float16]
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
results_data = Dict(
    "metadata" => Dict(
        "date" => string(Dates.now()),
        "problem_type" => problem_type,
        "problem_size" => n,
        "grid_size" => "$(Int(sqrt(n)))×$(Int(sqrt(n)))",
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
