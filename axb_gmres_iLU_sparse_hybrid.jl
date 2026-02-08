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

# ===== DATA LOADING UTILITY =====
function load_system_from_files(path_A, path_b, path_x, PREC)
    if !isfile(path_A) error("File $path_A not found") end
    println("Loading Matrix Market files...")
    A_raw = MatrixMarket.mmread(path_A)
    # findnz handles any sparse format returned by MatrixMarket
    I, J, V = findnz(A_raw)
    A = sparse(I, J, Vector{PREC}(V), size(A_raw)...)
    
    b = Vector{PREC}(vec(MatrixMarket.mmread(path_b)))
    xt = Vector{PREC}(vec(MatrixMarket.mmread(path_x)))
    return A, b, xt, size(A, 1)
end

# ===== MAIN SOLVER WITH PRE-CONDITIONING STRATEGY =====
function solve_with_ilu(A_cpu, b_cpu, x_true, PREC)
    n = length(b_cpu)
    println("\n" * "="^70)
    @printf("Robust Solver: %d unknowns | Precision: %s\n", n, PREC)
    println("="^70)

    # --- 1. PRE-PROCESSING (Scaling and Shift) ---
    # Shift prevents zero pivots in ILU; Scaling improves conditioning
    epsilon = PREC == Float64 ? 1e-9 : 1f-7
    println("Applying Diagonal Scaling and Shift (ϵ = $epsilon)...")
    
    diag_A = diag(A_cpu)
    # d_i = 1 / sqrt(|A_ii| + epsilon)
    d = [1.0 / sqrt(abs(val) + epsilon) for val in diag_A]
    D = sparse(1:n, 1:n, PREC.(d))
    
    # Transform: A_scaled = D * (A + epsilon*I) * D
    A_scaled_cpu = D * (A_cpu + epsilon * I) * D
    b_scaled_cpu = D * b_cpu
    
    # --- 2. GPU TRANSFER & SETUP ---
    A_gpu = CuSparseMatrixCSR(A_scaled_cpu)
    b_gpu = CuArray(b_scaled_cpu)

    println("Building GPU-Native ILU(0)...")
    P = GPUSparseILU(A_gpu)

    # --- 3. LIVE CONVERGENCE PLOTTING ---
    p = plot(title="GMRES Convergence (Scaled & Shifted)", 
             xlabel="Iteration", ylabel="Relative Residual",
             yaxis=:log10, label="||r||/||b||", lw=2, grid=true,
             color=:crimson)
    display(p)

    println("\nStarting GMRES (Restart Memory=150)...")
    
    t_solve = @elapsed begin
        # We solve the scaled system: (D A D) y = (D b)
        y_gpu, stats = gmres(A_gpu, b_gpu; 
                             M=P, ldiv=true, 
                             restart=true, memory=150, 
                             itmax=2500, 
                             atol=1e-8, rtol=1e-8,
                             verbose=1,
                             callback = (solver) -> begin
                                 it = solver.stats.niter
                                 if it > 0 && it % 10 == 0 && !isempty(solver.stats.residuals)
                                     # Plot normalized residual
                                     res = solver.stats.residuals[end] / solver.stats.residuals[1]
                                     push!(p, it, res)
                                     gui(p)
                                 end
                                 return false # Continue solving
                             end)
    end

    # --- 4. RECOVER SOLUTION ---
    # Original solution x = D * y
    x_cpu = D * Array(y_gpu)
    savefig("convergence_report.png")
    
    err = norm(x_cpu - x_true) / norm(x_true)
    @printf("\nSolve Time: %.2f s | Iterations: %d\n", t_solve, stats.niter)
    @printf("Final Relative Error vs True: %.2e\n", err)
    @printf("Solver Status: %s\n", stats.solved ? "CONVERGED" : "FAILED")
    
    return x_cpu, stats
end

# ===== EXECUTION =====
mode = "readdata" 
PRECISION = Float64

A, b, xt, n_actual = if mode == "readdata"
    load_system_from_files("sparse_Abx_data_A.mtx", "sparse_Abx_data_b.mtx", "sparse_Abx_data_x.mtx", PRECISION)
else
    # Laplacian Test Case
    nx = 585; n_t = nx^2
    I_idx, J_idx, V_idx = Int[], Int[], PRECISION[]
    for i in 1:n_t
        push!(I_idx, i); push!(J_idx, i); push!(V_idx, 4.0)
    end
    sparse(I_idx, J_idx, V_idx, n_t, n_t), ones(PRECISION, n_t), ones(PRECISION, n_t), n_t
end

x_final, stats = solve_with_ilu(A, b, xt, PRECISION)

println("Saving solution to x_out.mtx...")
MatrixMarket.mmwrite("x_out.mtx", sparse(reshape(x_final, :, 1)))
