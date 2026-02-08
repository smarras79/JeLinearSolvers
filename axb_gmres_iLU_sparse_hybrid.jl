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

function GPUSparseILU(A_cpu::SparseMatrixCSC{T}) where T
    n = size(A_cpu, 1)
    A_gpu = CuSparseMatrixCSR(A_cpu)
    LU = copy(A_gpu)
    
    # Modern CUSPARSE call for ILU0
    CUSPARSE.ilu02!(LU) 
    
    temp_z = CuVector{T}(undef, n)
    return GPUSparseILU{T}(LU, temp_z)
end

function LinearAlgebra.ldiv!(y, P::GPUSparseILU, x)
    ldiv!(P.temp_z, UnitLowerTriangular(P.LU), x)
    ldiv!(y, UpperTriangular(P.LU), P.temp_z)
    return y
end

# ===== DATA LOADING =====
function load_system_from_files(path_A, path_b, path_x, PREC)
    println("Loading Matrix Market files...")
    A_raw = MatrixMarket.mmread(path_A)
    I, J, V = findnz(A_raw)
    A = sparse(I, J, Vector{PREC}(V), size(A_raw)...)
    b = Vector{PREC}(vec(MatrixMarket.mmread(path_b)))
    xt = Vector{PREC}(vec(MatrixMarket.mmread(path_x)))
    return A, b, xt, size(A, 1)
end

# ===== SOLVER WITH LIVE PLOTTING =====
function solve_with_ilu(A_cpu, b_cpu, x_true, PREC)
    n = length(b_cpu)
    println("\n" * "="^70)
    @printf("SYSTEM: %d unknowns | %s\n", n, PREC)
    println("="^70)
    
    println("Building GPU-Native ILU(0)...")
    P = GPUSparseILU(A_cpu)

    A_gpu = CuSparseMatrixCSR(A_cpu)
    b_gpu = CuArray(b_cpu)

    # Initialize Plot
    p = plot(title="GMRES Convergence: 342k Unknowns", 
             xlabel="Iteration", ylabel="Relative Residual",
             yaxis=:log10, label="||r||/||b||", lw=2,
             marker=:circle, markersize=2, grid=true)
    display(p)

    println("\nStarting GMRES Solve...")
    
    t_solve = @elapsed begin
        x_ilu, stats = gmres(A_gpu, b_gpu; 
                             M=P, ldiv=true, 
                             restart=true, memory=40,
                             itmax=1000, 
                             atol=1e-8, rtol=1e-8,
                             verbose=1,
                             callback = (solver) -> begin
                                 it = solver.stats.niter
                                 # Plot every 10 iterations
                                 if it > 0 && it % 10 == 0 && !isempty(solver.stats.residuals)
                                     res_norm = solver.stats.residuals[end] / solver.stats.residuals[1]
                                     push!(p, it, res_norm)
                                     gui(p)
                                 end
                                 return false # CRITICAL: Tells Krylov NOT to stop the solver
                             end)
    end

    savefig("convergence_log.png")
    
    x_cpu = Array(x_ilu)
    err = norm(x_cpu - x_true) / norm(x_true)
    
    @printf("\nSolve completed in: %.2f s\n", t_solve)
    @printf("Final Relative Error: %.2e\n", err)
    
    return x_cpu, stats
end

# ===== RUN =====
mode = "readdata" 
PRECISION = Float64

A, b, xt, n_actual = if mode == "readdata"
    load_system_from_files("sparse_Abx_data_A.mtx", "sparse_Abx_data_b.mtx", "sparse_Abx_data_x.mtx", PRECISION)
else
    # Simple Identity test
    n_t = 5000; I_t = sparse(1.0I, n_t, n_t)
    I_t, ones(n_t), ones(n_t), n_t
end

x_final, stats = solve_with_ilu(A, b, xt, PRECISION)

println("Writing solution to x_out.mtx...")
MatrixMarket.mmwrite("x_out.mtx", sparse(reshape(x_final, :, 1)))
