using SparseArrays, LinearAlgebra, Random, Printf, Dates
using CUDA, CUDA.CUSPARSE
using Krylov 
using MatrixMarket

# ===== COMPATIBLE NATIVE GPU ILU PRECONDITIONER =====
struct GPUSparseILU{T}
    LU::CuSparseMatrixCSR{T, Int32}
    temp_z::CuVector{T} 
end

function GPUSparseILU(A_cpu::SparseMatrixCSC{T}) where T
    n = size(A_cpu, 1)
    nnz_A = nnz(A_cpu)
    
    # GPU Memory Check
    available = CUDA.available_memory()
    # Matrix + LU copy + Int32 overhead
    estimated = (nnz_A * (sizeof(T) + 4) + n * 4) * 2.5 
    if estimated > available
        @warn "GPU Memory might be tight."
    end

    # 1. Transfer and Convert to CSR
    A_gpu = CuSparseMatrixCSR(A_cpu)
    
    # 2. Factorization
    # Using sv0! which is the standard name in many CUDA.jl versions
    LU = copy(A_gpu)
    CUSPARSE.sv0!(LU, 'N') # 'N' for non-transpose
    
    temp_z = CuVector{T}(undef, n)
    return GPUSparseILU{T}(LU, temp_z)
end

function LinearAlgebra.ldiv!(y, P::GPUSparseILU, x)
    # Using explicit Triangular wrappers for dispatch
    # L is UnitLowerTriangular, U is UpperTriangular
    ldiv!(P.temp_z, UnitLowerTriangular(P.LU), x)
    ldiv!(y, UpperTriangular(P.LU), P.temp_z)
    return y
end

# ===== DATA LOADING =====
function load_system_from_files(path_A, path_b, path_x, PREC)
    if !isfile(path_A) || !isfile(path_b) || !isfile(path_x)
        error("Matrix files not found.")
    end

    println("Loading Matrix Market files...")
    A_raw = MatrixMarket.mmread(path_A)
    actual_n = size(A_raw, 1)
    
    # Safe extraction of indices and values
    I, J, V = findnz(A_raw)
    A = sparse(I, J, Vector{PREC}(V), actual_n, actual_n)
    
    b = Vector{PREC}(vec(MatrixMarket.mmread(path_b)))
    xt = Vector{PREC}(vec(MatrixMarket.mmread(path_x)))
    
    return A, b, xt, actual_n
end

# ===== MAIN SOLVER FUNCTION =====
function solve_with_ilu(A_cpu, b_cpu, x_true, PREC)
    n = length(b_cpu)
    nnz_A = nnz(A_cpu)
    
    println("\n" * "="^70)
    @printf("SYSTEM: %d unknowns | %d non-zeros | %s\n", n, nnz_A, PREC)
    println("="^70)
    
    println("Building GPU-Native ILU(0)...")
    t_pre = @elapsed P = GPUSparseILU(A_cpu)
    @printf("  ✓ Preconditioner ready in %.3f s\n", t_pre)

    A_gpu = CuSparseMatrixCSR(A_cpu)
    b_gpu = CuArray(b_cpu)
    x_gpu_true = CuArray(x_true)

    atol = PREC == Float64 ? 1e-8 : 1f-6
    rtol = PREC == Float64 ? 1e-8 : 1f-6
    
    println("\nStarting GMRES Solve...")
    println("-"^70)
    
    t_solve = @elapsed begin
        x_ilu, stats = gmres(A_gpu, b_gpu; 
                             M=P, 
                             ldiv=true, 
                             restart=true, 
                             memory=30, 
                             itmax=1000, 
                             atol=atol, 
                             rtol=rtol, 
                             verbose=1)
    end
    println("-"^70)

    err = norm(x_ilu - x_gpu_true) / norm(x_gpu_true)
    @printf("Solve completed in: %.2f ms\n", t_solve * 1000)
    @printf("Iterations: %d\n", stats.niter)
    @printf("Converged: %s\n", stats.solved)

    return stats
end

# ===== RUN =====
mode = "readdata" 
PRECISION = Float64

A, b, xt, n_actual = if mode == "readdata"
    load_system_from_files("sparse_Abx_data_A.mtx", "sparse_Abx_data_b.mtx", "sparse_Abx_data_x.mtx", PRECISION)
else
    # Fallback for testing
    nx = 585 # approx 342,225 nodes
    actual_n = nx*nx
    I, J, V = Int[], Int[], PRECISION[]
    for i in 1:actual_n
        push!(I, i); push!(J, i); push!(V, 4.0)
    end
    sparse(I, J, V, actual_n, actual_n), ones(PRECISION, actual_n), ones(PRECISION, actual_n), actual_n
end

stats = solve_with_ilu(A, b, xt, PRECISION)
