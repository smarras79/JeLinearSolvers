using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using IncompleteLU, Krylov

# ===== SET PRECISION HERE =====
const PREC = Float32
# ==============================

Random.seed!(42)
n = 100
nnz = 200

# Create SPD matrix
rows = rand(1:n, nnz)
cols = rand(1:n, nnz)
vals = rand(PREC, nnz)
A_temp = sparse(rows, cols, vals, n, n)
A_cpu = A_temp + A_temp' + PREC(20.0) * spdiagm(0 => ones(PREC, n))

x_true = randn(PREC, n)
b_cpu = A_cpu * x_true

println("A size: ", size(A_cpu))
println("b size: ", size(b_cpu))
println("Precision: ", PREC)

# ===== INCOMPLETE LU FACTORIZATION =====
println("\nComputing ILU(τ=$(PREC(0.01)))...")
ilu_fact = ilu(A_cpu, τ=PREC(0.01))

nnz_L = nnz(ilu_fact.L)
nnz_U = nnz(ilu_fact.U)
println("ILU sparsity: L has $nnz_L nnz, U has $nnz_U nnz")
# =======================================

# Transfer to GPU - KEEP SPARSE (use CSR for CUSOLVER)
A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuArray(b_cpu)
L_gpu = CuSparseMatrixCSR(ilu_fact.L)
U_gpu = CuSparseMatrixCSR(ilu_fact.U)

# Sparse ILU Preconditioner using CUSOLVER
struct SparseiLUPreconditioner{T,TM}
    L::TM
    U::TM
    L_info::CUSOLVER.csrsv2Info
    U_info::CUSOLVER.csrsv2Info
    temp::CuVector{T}
    policy::CUSOLVER.cusparseSolvePolicy_t
end

function SparseiLUPreconditioner(L::TM, U::TM) where {T,TM<:CuSparseMatrixCSR{T}}
    n = size(L, 1)
    
    # Create analysis info for L and U
    L_info = CUSOLVER.csrsv2Info()
    U_info = CUSOLVER.csrsv2Info()
    
    policy = CUSOLVER.CUSPARSE_SOLVE_POLICY_USE_LEVEL
    
    # Analyze L (lower triangular)
    temp_buffer_L = CuVector{UInt8}(undef, 1)  # Will be resized
    CUSOLVER.cusparseXcsrsv2_bufferSize(
        CUSOLVER.handle(), 
        CUSOLVER.CUSPARSE_OPERATION_NON_TRANSPOSE,
        n, nnz(L), L, 
        L_info, 
        temp_buffer_L
    )
    
    CUSOLVER.cusparseXcsrsv2_analysis(
        CUSOLVER.handle(),
        CUSOLVER.CUSPARSE_OPERATION_NON_TRANSPOSE,
        n, nnz(L), L,
        L_info,
        policy,
        temp_buffer_L
    )
    
    # Analyze U (upper triangular)
    temp_buffer_U = CuVector{UInt8}(undef, 1)
    CUSOLVER.cusparseXcsrsv2_bufferSize(
        CUSOLVER.handle(),
        CUSOLVER.CUSPARSE_OPERATION_NON_TRANSPOSE,
        n, nnz(U), U,
        U_info,
        temp_buffer_U
    )
    
    CUSOLVER.cusparseXcsrsv2_analysis(
        CUSOLVER.handle(),
        CUSOLVER.CUSPARSE_OPERATION_NON_TRANSPOSE,
        n, nnz(U), U,
        U_info,
        policy,
        temp_buffer_U
    )
    
    SparseiLUPreconditioner{T,TM}(L, U, L_info, U_info, CUDA.zeros(T, n), policy)
end

function LinearAlgebra.ldiv!(y, P::SparseiLUPreconditioner{T}, x) where T
    n = length(x)
    alpha = one(T)
    
    # Forward solve: L * temp = x
    CUSOLVER.cusparseXcsrsv2_solve(
        CUSOLVER.handle(),
        CUSOLVER.CUSPARSE_OPERATION_NON_TRANSPOSE,
        n, nnz(P.L), Ref(alpha),
        P.L, x, P.temp,
        P.L_info, P.policy
    )
    
    # Backward solve: U * y = temp
    CUSOLVER.cusparseXcsrsv2_solve(
        CUSOLVER.handle(),
        CUSOLVER.CUSPARSE_OPERATION_NON_TRANSPOSE,
        n, nnz(P.U), Ref(alpha),
        P.U, P.temp, y,
        P.U_info, P.policy
    )
    
    return y
end

println("\nBuilding sparse ILU preconditioner on GPU...")
P = SparseiLUPreconditioner(L_gpu, U_gpu)

# Solve
atol = PREC == Float64 ? 1e-6 : PREC == Float32 ? 1f-6 : Float16(1e-4)
rtol = PREC == Float64 ? 1e-6 : PREC == Float32 ? 1f-6 : Float16(1e-4)

println("\nSolving with sparse ILU-preconditioned GMRES...")
x_gpu, stats = gmres(A_gpu, b_gpu; 
                     M=P, 
                     ldiv=true,
                     atol=atol,
                     rtol=rtol,
                     restart=true,
                     itmax=100,
                     verbose=1,
                     history=true)

println("\n✓ Converged: ", stats.solved)
println("✓ Iterations: ", stats.niter)
println("✓ Final residual: ", stats.residuals[end])

x_cpu = Array(x_gpu)
error = norm(A_cpu * x_cpu - b_cpu) / norm(b_cpu)
println("✓ Relative error: ", error)
