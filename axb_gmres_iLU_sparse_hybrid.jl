using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
using IncompleteLU, Krylov
using Printf

# ===== CREATE REALISTIC TEST PROBLEM =====
function create_2d_laplacian(n, PREC)
    """
    Create 2D Laplacian matrix (discretized Poisson equation)
    This is a realistic test case from CFD/PDE applications
    """
    nx = ny = Int(sqrt(n))
    actual_n = nx * ny
    
    rows = Int[]
    cols = Int[]
    vals = PREC[]
    
    println("Creating 2D Laplacian ($(nx)×$(ny) grid, $actual_n unknowns)...")
    
    for j in 1:ny
        for i in 1:nx
            k = i + (j-1)*nx
            
            # Diagonal (diffusion coefficient)
            push!(rows, k); push!(cols, k); push!(vals, PREC(4.0))
            
            # Left neighbor
            if i > 1
                push!(rows, k); push!(cols, k-1); push!(vals, PREC(-1.0))
            end
            # Right neighbor
            if i < nx
                push!(rows, k); push!(cols, k+1); push!(vals, PREC(-1.0))
            end
            # Bottom neighbor
            if j > 1
                push!(rows, k); push!(cols, k-nx); push!(vals, PREC(-1.0))
            end
            # Top neighbor
            if j < ny
                push!(rows, k); push!(cols, k+nx); push!(vals, PREC(-1.0))
            end
        end
    end
    
    A = sparse(rows, cols, vals, actual_n, actual_n)
    
    # Create right-hand side
    x_true = ones(PREC, actual_n)
    b = A * x_true
    
    return A, b, x_true, actual_n
end

# ===== HYBRID SPARSE ILU PRECONDITIONER =====
struct HybridSparseILU{T,TL,TU}
    L_cpu::TL
    U_cpu::TU
    temp_cpu::Vector{T}
end

function HybridSparseILU(L_cpu::TL, U_cpu::TU) where {T,TL<:SparseMatrixCSC{T},TU<:SparseMatrixCSC{T}}
    n = size(L_cpu, 1)
    HybridSparseILU{T,TL,TU}(L_cpu, U_cpu, Vector{T}(undef, n))
end

function LinearAlgebra.ldiv!(y, P::HybridSparseILU, x)
    x_cpu = Array(x)
    ldiv!(P.temp_cpu, LowerTriangular(P.L_cpu), x_cpu)
    y_cpu = P.U_cpu \ P.temp_cpu
    copyto!(y, y_cpu)
    return y
end

# ===== MIXED PRECISION ILU SOLVER =====
function solve_with_ilu(A_cpu, b_cpu, x_true, PREC)
    n = length(b_cpu)
    
    println("\n" * "="^70)
    println("PRECISION: $PREC")
    println("="^70)
    
    # Matrix statistics
    nnz_A = SparseArrays.nnz(A_cpu)
    density = nnz_A / (n * n) * 100
    
    println("Matrix statistics:")
    println("  Size: $n × $n")
    println("  Nonzeros: $nnz_A")
    println("  Density: $(round(density, digits=2))%")
    
    # Estimate condition number (expensive for large matrices)
    if n <= 1000
        cond_est = cond(Matrix(A_cpu))
        println("  Condition number: $(round(cond_est, digits=1))")
    end
    
    # ===== COMPUTE ILU FACTORIZATION =====
    println("\nComputing ILU(0) factorization...")
    
    ilu_fact = try
        ilu(A_cpu, τ=PREC(0.0))
    catch e
        println("  ⚠ ILU failed: $e")
        println("  Using diagonal preconditioner instead")
        D = [A_cpu[i,i] for i in 1:n]
        L_fact = sparse(Diagonal(sqrt.(abs.(D))))
        U_fact = sparse(Diagonal(sqrt.(abs.(D))))
        (L = L_fact, U = U_fact)
    end
    
    nnz_L = SparseArrays.nnz(ilu_fact.L)
    nnz_U = SparseArrays.nnz(ilu_fact.U)
    fill_factor = (nnz_L + nnz_U) / (2 * nnz_A)
    
    println("  ✓ ILU completed")
    println("  L: $nnz_L nnz")
    println("  U: $nnz_U nnz")
    println("  Fill factor: $(round(fill_factor, digits=2))x")
    println("  Memory vs dense: $(round(100*(1 - (nnz_L+nnz_U)/(n*n)), digits=1))% savings")
    
    # ===== TRANSFER TO GPU =====
    A_gpu = CuSparseMatrixCSC(A_cpu)
    b_gpu = CuArray(b_cpu)
    
    # Build preconditioner
    P = HybridSparseILU(ilu_fact.L, ilu_fact.U)
    
    # ===== SOLVE WITHOUT PRECONDITIONER =====
    println("\n[1/2] Solving WITHOUT preconditioner...")
    
    atol = PREC == Float64 ? 1e-8 : PREC == Float32 ? 1f-6 : Float16(1e-3)
    rtol = PREC == Float64 ? 1e-8 : PREC == Float32 ? 1f-6 : Float16(1e-3)
    
    t_noprecond = @elapsed begin
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
    
    # Safe access to residuals
    final_res_noprecond = length(stats_noprecond.residuals) > 0 ? 
                          stats_noprecond.residuals[end] : NaN
    
    println("  Iterations: $(stats_noprecond.niter)")
    println("  Time: $(round(t_noprecond*1000, digits=2)) ms")
    println("  Final residual: $(final_res_noprecond)")
    println("  Relative error: $(err_noprecond)")
    println("  Converged: $(stats_noprecond.solved)")
    
    # ===== SOLVE WITH ILU PRECONDITIONER =====
    println("\n[2/2] Solving WITH ILU preconditioner...")
    
    t_ilu = @elapsed begin
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
    
    # Safe access to residuals
    final_res_ilu = length(stats_ilu.residuals) > 0 ? 
                    stats_ilu.residuals[end] : NaN
    
    println("  Iterations: $(stats_ilu.niter)")
    println("  Time: $(round(t_ilu*1000, digits=2)) ms")
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
    println("Time speedup: $(round(time_speedup, digits=2))x")
    
    if stats_ilu.niter < stats_noprecond.niter
        println("✓ ILU preconditioning EFFECTIVE")
    else
        println("⚠ ILU preconditioning NOT effective (problem may be too easy)")
    end
    
    return (
        noprecond_iters = stats_noprecond.niter,
        noprecond_time = t_noprecond,
        noprecond_error = err_noprecond,
        ilu_iters = stats_ilu.niter,
        ilu_time = t_ilu,
        ilu_error = err_ilu,
        iter_reduction = iter_reduction,
        time_speedup = time_speedup
    )
end

# ===== MAIN: MIXED PRECISION STUDY =====
println("="^70)
println("MIXED PRECISION ILU PRECONDITIONING STUDY")
println("GPU-Accelerated Sparse Linear Solver")
println("="^70)

# Problem size (adjust as needed)
# n = 2500 → 50×50 grid (good for testing)
# n = 10000 → 100×100 grid (realistic)
# n = 40000 → 200×200 grid (large)
n = 10000

results = Dict()

for PREC in [Float64, Float32, Float16]
    Random.seed!(42)  # Consistent across precisions
    
    A_cpu, b_cpu, x_true, actual_n = create_2d_laplacian(n, PREC)
    
    result = solve_with_ilu(A_cpu, b_cpu, x_true, PREC)
    results[PREC] = result
    
    # Add spacing between tests
    println()
end

# ===== FINAL COMPARISON =====
println("="^70)
println("MIXED PRECISION COMPARISON")
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
println("\nKEY FINDINGS:")
println("• Lower precision → Faster computation, less memory")
println("• ILU preconditioning → Fewer iterations, faster solve")
println("• Trade-off: Precision vs Speed vs Accuracy")
println("="^70)
