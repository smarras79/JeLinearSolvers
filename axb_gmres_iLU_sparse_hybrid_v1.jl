using SparseArrays, LinearAlgebra, Random, Printf, Dates
using CUDA, CUDA.CUSPARSE
using Krylov 
using MatrixMarket
using Plots

# ===== 1. PRECONDITIONER SETUP =====
struct GPUSparseILU{T}
    LU::CuSparseMatrixCSR{T, Int32}
    temp_z::CuVector{T} 
end

function GPUSparseILU(A_gpu::CuSparseMatrixCSR{T}) where T
    LU = copy(A_gpu)
    CUSPARSE.ilu02!(LU) 
    return GPUSparseILU{T}(LU, CuVector{T}(undef, size(A_gpu, 1)))
end

function LinearAlgebra.ldiv!(y, P::GPUSparseILU, x)
    ldiv!(P.temp_z, UnitLowerTriangular(P.LU), x)
    ldiv!(y, UpperTriangular(P.LU), P.temp_z)
    return y
end

# ===== 2. DATA LOADING =====
function load_system_from_files(path_A, path_b, path_x, PREC)
    println("Loading Matrix Market files...")
    A_raw = MatrixMarket.mmread(path_A)
    I, J, V = findnz(A_raw)
    A = sparse(I, J, Vector{PREC}(V), size(A_raw)...)
    b = Vector{PREC}(vec(MatrixMarket.mmread(path_b)))
    xt = Vector{PREC}(vec(MatrixMarket.mmread(path_x)))
    return A, b, xt
end

# ===== 3. FLEXIBLE SOLVER FUNCTION =====
function solve_linear_system(A_cpu, b_cpu, x_true, solver_type="idrs"; PREC=Float64)
    n = length(b_cpu)
    
    # --- PRE-PROCESSING: Scaling & Stabilization ---
    # Shift (epsilon) is crucial if the matrix is close to singular
    epsilon = PREC == Float64 ? 1e-7 : 1f-5 
    println("\nPreprocessing: Diagonal Scaling & Shift (ϵ = $epsilon)...")
    
    diag_A = diag(A_cpu)
    d = [1.0 / sqrt(abs(val) + epsilon) for val in diag_A]
    D = sparse(1:n, 1:n, PREC.(d))
    
    A_scaled_cpu = D * (A_cpu + epsilon * I) * D
    b_scaled_cpu = D * b_cpu
    
    # --- GPU SETUP ---
    A_gpu = CuSparseMatrixCSR(A_scaled_cpu)
    b_gpu = CuArray(b_scaled_cpu)
    P = GPUSparseILU(A_gpu)

    # --- SOLVER SELECTION ---
    println("Selected Solver: $solver_type")
    p = plot(title="Convergence: $solver_type", xlabel="Iteration", 
             ylabel="Relative Residual", yaxis=:log10, lw=2, grid=true)
    display(p)

    t_solve = @elapsed begin
        # Callback for real-time plotting
        cb = (solver) -> begin
            it = solver.stats.niter
            if it > 0 && it % 10 == 0 && !isempty(solver.stats.residuals)
                res = solver.stats.residuals[end] / solver.stats.residuals[1]
                push!(p, it, res)
                gui(p)
            end
            return false
        end

        # Dispatch based on user choice
        y_gpu, stats = if solver_type == "gmres"
            gmres(A_gpu, b_gpu; M=P, ldiv=true, restart=true, memory=150, itmax=2000, callback=cb, verbose=1)
        elseif solver_type == "bicgstab"
            bicgstab(A_gpu, b_gpu; M=P, ldiv=true, itmax=3000, callback=cb, verbose=1)
        elseif solver_type == "idrs"
            # IDR(s) with s=8 is very robust for non-symmetric systems
            idrs(A_gpu, b_gpu; M=P, ldiv=true, s=8, itmax=3000, callback=cb, verbose=1)
        else
            error("Unknown solver type: $solver_type")
        end
    end

    # --- RECOVER & EXPORT ---
    x_cpu = D * Array(y_gpu)
    err = norm(x_cpu - x_true) / norm(x_true)
    
    @printf("\nFinal Result: %s\n", stats.solved ? "CONVERGED" : "FAILED")
    @printf("Solve Time: %.2f s | Error: %.2e\n", t_solve, err)
    
    return x_cpu, stats
end

# ===== 4. EXECUTION =====
# Filenames (Adjust to your local paths)
fA, fb, fx = "sparse_Abx_data_A.mtx", "sparse_Abx_data_b.mtx", "sparse_Abx_data_x.mtx"
A_orig, b_orig, xt_orig = load_system_from_files(fA, fb, fx, Float64)

# CHOOSE YOUR SOLVER HERE: "gmres", "bicgstab", or "idrs"
choice = "idrs" 
x_final, stats = solve_linear_system(A_orig, b_orig, xt_orig, choice)

MatrixMarket.mmwrite("x_out.mtx", sparse(reshape(x_final, :, 1)))
