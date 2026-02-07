using SparseArrays, LinearAlgebra, Random, Printf, Dates
using CUDA, CUDA.CUSPARSE
using Krylov # For GMRES
using MatrixMarket

# ===== NATIVE GPU ILU PRECONDITIONER =====
struct GPUSparseILU{T}
    LU::CuSparseMatrixCSR{T, Int32}
    info::CUSPARSE.Csrilu02Info
    lower_info::CUSPARSE.Csrsv2Info
    upper_info::CUSPARSE.Csrsv2Info
    buffer::CuVector{UInt8}
    temp_z::CuVector{T} # Pre-allocated intermediate vector to avoid GPU GC
end

function GPUSparseILU(A_cpu::SparseMatrixCSC{T}) where T
    n = size(A_cpu, 1)
    nnz_A = nnz(A_cpu)
    
    # --- GPU Memory Check ---
    # Estimate: Matrix + LU Copy + Int32 RowPtrs/ColIndices + 2 Vectors
    estimated_bytes = (nnz_A * (sizeof(T) + sizeof(Int32)) + n * sizeof(Int32)) * 2.5
    available = CUDA.available_memory()
    
    if estimated_bytes > available
        @warn "Memory tight! Estimated: $(round(estimated_bytes/1024^2, digits=2))MB, Available: $(round(available/1024^2, digits=2))MB"
    end

    # 1. Transfer and Convert to CSR (Required for CUSPARSE ILU)
    A_gpu = CuSparseMatrixCSR(A_cpu)
    
    # 2. Setup CUSPARSE objects
    info = CUSPARSE.Csrilu02Info()
    lower_info = CUSPARSE.Csrsv2Info()
    upper_info = CUSPARSE.Csrsv2Info()
    
    # 3. Workspace allocation
    buffer_size = CUSPARSE.csrilu02_bufferSize(A_gpu, info)
    buffer = CuVector{UInt8}(undef, buffer_size)
    
    # 4. Factorization (In-place on a copy of A)
    LU = copy(A_gpu)
    CUSPARSE.csrilu02_analysis(LU, info, buffer)
    CUSPARSE.csrilu02!(LU, info, buffer) # This is the actual ILU(0)
    
    # 5. Analysis for Solves (Pre-plans the triangular solve dependency graph)
    CUSPARSE.csrsv2_analysis(LU, 'N', 'L', 'U', lower_info, buffer)
    CUSPARSE.csrsv2_analysis(LU, 'N', 'U', 'N', upper_info, buffer)
    
    temp_z = CuVector{T}(undef, n)
    
    return GPUSparseILU{T}(LU, info, lower_info, upper_info, buffer, temp_z)
end

function LinearAlgebra.ldiv!(y, P::GPUSparseILU, x)
    # L * z = x
    CUSPARSE.csrsv2_solve(P.LU, 'N', 'L', 'U', x, P.temp_z, P.lower_info, P.buffer)
    # U * y = z
    CUSPARSE.csrsv2_solve(P.LU, 'N', 'U', 'N', P.temp_z, y, P.upper_info, P.buffer)
    return y
end

# ===== PROBLEM GENERATORS =====
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
    return A, b, x_true
end

function load_system_from_files(path_A, path_b, path_x, PREC)
    A_raw = MatrixMarket.mmread(path_A)
    actual_n = size(A_raw, 1)
    A = sparse(A_raw.row, A_raw.col, Vector{PREC}(A_raw.nzval), actual_n, actual_n)
    b = Vector{PREC}(vec(MatrixMarket.mmread(path_b)))
    x_true = Vector{PREC}(vec(MatrixMarket.mmread(path_x)))
    return A, b, x_true
end

# ===== MAIN SOLVER LOOP =====
function solve_with_ilu(A_cpu, b_cpu, x_true, PREC)
    n = length(b_cpu)
    nnz_A = nnz(A_cpu)
    
    println("\n" * "="^70)
    @printf("SYSTEM: %d unknowns | %d non-zeros | %s\n", n, nnz_A, PREC)
    println("="^70)
    
    # Preconditioner Setup on GPU
    println("Building GPU-Native ILU(0)...")
    t_pre = @elapsed P = GPUSparseILU(A_cpu)
    @printf("  ✓ Done in %.3f seconds\n", t_pre)

    # Data to GPU
    A_gpu = CuSparseMatrixCSR(A_cpu)
    b_gpu = CuArray(b_cpu)
    x_gpu_true = CuArray(x_true)

    # Solver Parameters
    atol = PREC == Float64 ? 1e-8 : 1f-6
    rtol = PREC == Float64 ? 1e-8 : 1f-6
    
    println("\nStarting GMRES (Real-time logging enabled)...")
    println("-"^70)
    
    t_solve = @elapsed begin
        x_ilu, stats = gmres(A_gpu, b_gpu; 
                             M=P, 
                             ldiv=true, 
                             restart=true, 
                             memory=30, # GMRES(30)
                             itmax=500, 
                             atol=atol, 
                             rtol=rtol, 
                             verbose=1) # LOGGING ACTIVATED
    end
    println("-"^70)

    # Final Check
    err = norm(x_ilu - x_gpu_true) / norm(x_gpu_true)
    @printf("Solve completed in: %.2f ms\n", t_solve * 1000)
    @printf("Iterations: %d\n", stats.niter)
    @printf("Final Relative Error: %.2e\n", err)
    @printf("Converged: %s\n", stats.solved)

    return stats
end

# ===== EXECUTION =====
# problem_type = "laplacian"
# n_target = 1000000 

# Adjust this to "readdata" if you have your .mtx files ready
problem_type = "readdata" 
n_target = 400000 # 342732 is ~ 585x585 grid

PREC = Float32
A, b, x_true = if problem_type == "laplacian"
    create_2d_laplacian(n_target, PREC)
else
    
    # Replace with your actual file names
    load_system_from_files("sparse_Abx_data_A.mtx",
                           "sparse_Abx_data_b.mtx",
                           "sparse_Abx_data_x.mtx",
                           PREC)
end

stats = solve_with_ilu(A, b, x_true, PREC)
