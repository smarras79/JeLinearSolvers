using SparseArrays, LinearAlgebra, Random, Printf, Dates
using CUDA, CUDA.CUSPARSE
using Krylov 
using MatrixMarket

# ===== MODERN NATIVE GPU ILU PRECONDITIONER =====
struct GPUSparseILU{T}
    LU::CuSparseMatrixCSR{T, Int32}
    temp_z::CuVector{T} 
end

function GPUSparseILU(A_cpu::SparseMatrixCSC{T}) where T
    n = size(A_cpu, 1)
    nnz_A = nnz(A_cpu)
    
    # GPU Memory Check
    estimated_bytes = (nnz_A * (sizeof(T) + sizeof(Int32)) + n * sizeof(Int32)) * 2.5
    available = CUDA.available_memory()
    if estimated_bytes > available
        @warn "Memory tight! Needs ~$(round(estimated_bytes/1024^2, digits=2))MB, Available: $(round(available/1024^2, digits=2))MB"
    end

    # 1. Transfer and Convert to CSR (Required for CUSPARSE ILU)
    A_gpu = CuSparseMatrixCSR(A_cpu)
    
    # 2. Factorization (In-place on a copy)
    LU = copy(A_gpu)
    CUSPARSE.ilu0!(LU) 
    
    temp_z = CuVector{T}(undef, n)
    return GPUSparseILU{T}(LU, temp_z)
end

function LinearAlgebra.ldiv!(y, P::GPUSparseILU, x)
    # Step 1: Solve L * z = x (Unit lower triangular)
    ldiv!(P.temp_z, UnitLowerTriangular(P.LU), x)
    # Step 2: Solve U * y = z (Upper triangular)
    ldiv!(y, UpperTriangular(P.LU), P.temp_z)
    return y
end

# ===== DATA LOADING & GENERATION =====
function load_system_from_files(path_A, path_b, path_x, PREC)
    if !isfile(path_A) || !isfile(path_b) || !isfile(path_x)
        error("Matrix files not found at specified paths.")
    end

    println("Loading Matrix Market files...")
    A_raw = MatrixMarket.mmread(path_A)
    actual_n = size(A_raw, 1)
    
    # findnz is the safe way to get coordinates and values from any sparse format
    I, J, V = findnz(A_raw)
    
    # Convert to SparseMatrixCSC with specific precision
    # This ensures the matrix is properly structured for CUDA conversion
    A = sparse(I, J, Vector{PREC}(V), actual_n, actual_n)
    
    # Load and flatten vectors
    b_raw = MatrixMarket.mmread(path_b)
    b = Vector{PREC}(vec(b_raw))
    
    x_raw = MatrixMarket.mmread(path_x)
    x_true = Vector{PREC}(vec(x_raw))
    
    return A, b, x_true, actual_n
end


function create_2d_laplacian(n, PREC)
    nx = ny = Int(sqrt(n))
    actual_n = nx * ny
    println("Creating 2D Laplacian ($(nx)×$(ny) grid, $actual_n unknowns)...")
    
    rows, cols, vals = Int[], Int[], PREC[]
    for j in 1:ny, i in 1:nx
        k = i + (j-1)*nx
        push!(rows, k); push!(cols, k); push!(vals, PREC(4.0))
        if i > 1  push!(rows, k); push!(cols, k-1); push!(vals, PREC(-1.0)) end
        if i < nx push!(rows, k); push!(cols, k+1); push!(vals, PREC(-1.0)) end
        if j > 1  push!(rows, k); push!(cols, k-nx); push!(vals, PREC(-1.0)) end
        if j < ny push!(rows, k); push!(cols, k+nx); push!(vals, PREC(-1.0)) end
    end
    A = sparse(rows, cols, vals, actual_n, actual_n)
    x_true = ones(PREC, actual_n)
    b = A * x_true
    return A, b, x_true, actual_n
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
    
    println("\nStarting GMRES Solve (Memory=30)...")
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
    @printf("Final Relative Error: %.2e\n", err)
    @printf("Converged: %s\n", stats.solved)

    return stats
end

# ===== RUN SCRIPT =====

mode = "readdata"  # Options: "readdata" or "laplacian"
PRECISION = Float64

A, b, xt, n_actual = if mode == "readdata"
    load_system_from_files("sparse_Abx_data_A.mtx", "sparse_Abx_data_b.mtx", "sparse_Abx_data_x.mtx", PRECISION)
else
    create_2d_laplacian(400000, PRECISION)
end

stats = solve_with_ilu(A, b, xt, PRECISION)
