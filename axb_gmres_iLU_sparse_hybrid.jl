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
    
    # 1. Transfer and Convert to CSR
    A_gpu = CuSparseMatrixCSR(A_cpu)
    
    # 2. Factorization
    # We use a copy to preserve A, then apply the incomplete factorization.
    # If ilu0! or sv0! are missing, we use the direct C-binding wrapper
    LU = copy(A_gpu)
    
    # This call is the most robust across CUDA.jl 5.x versions
    CUSPARSE.ilu02!(LU) 
    
    temp_z = CuVector{T}(undef, n)
    return GPUSparseILU{T}(LU, temp_z)
end

function LinearAlgebra.ldiv!(y, P::GPUSparseILU, x)
    # Modern CUDA.jl handles the triangular solve dispatch via these wrappers
    # This avoids having to call low-level csrsv2 functions manually
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

    # Initialize Log Plot
    p = plot(title="GMRES Convergence (342k System)", 
             xlabel="Iteration", ylabel="Relative Residual",
             yaxis=:log10, label="||r||/||b||", lw=2)
    display(p)

    println("\nStarting GMRES Solve...")
    
    t_solve = @elapsed begin
        x_ilu, stats = gmres(A_gpu, b_gpu; 
                             M=P, ldiv=true, 
                             restart=true, memory=40, # Slightly higher memory for large systems
                             itmax=1000, 
                             atol=1e-8, rtol=1e-8,
                             verbose=1,
                             callback = (solver) -> begin
                                 if solver.stats.niter % 10 == 0
                                     push!(p, solver.stats.niter, solver.stats.residuals[end])
                                     gui(p)
                                 end
                             end)
    end

    # Save Results
    savefig("convergence_log.png")
    x_cpu = Array(x_ilu)
    err = norm(x_cpu - x_true) / norm(x_true)
    
    @printf("\nSolve completed in: %.2f s\n", t_solve)
    @printf("Final Relative Error: %.2e\n", err)
    
    return x_cpu, stats
end

# ===== EXECUTION =====
mode = "readdata" 
PRECISION = Float64

A, b, xt, n_actual = if mode == "readdata"
    load_system_from_files("sparse_Abx_data_A.mtx", "sparse_Abx_data_b.mtx", "sparse_Abx_data_x.mtx", PRECISION)
else
    # Small test if files are missing
    A_test = sprand(5000, 5000, 0.001) + I
    A_test, ones(5000), ones(5000), 5000
end

x_final, stats = solve_with_ilu(A, b, xt, PRECISION)

println("Writing solution to x_out.mtx...")
MatrixMarket.mmwrite("x_out.mtx", sparse(reshape(x_final, :, 1)))
