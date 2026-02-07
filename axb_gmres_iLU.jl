using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
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

# Check sparsity
nnz_L = nnz(ilu_fact.L)
nnz_U = nnz(ilu_fact.U)
println("ILU sparsity: L has $nnz_L nnz, U has $nnz_U nnz (full would have $(n*n))")
# =======================================

# Transfer to GPU - KEEP SPARSE
A_gpu = CuSparseMatrixCSC(A_cpu)
b_gpu = CuArray(b_cpu)
L_gpu_sparse = CuSparseMatrixCSC(ilu_fact.L)
U_gpu_sparse = CuSparseMatrixCSC(ilu_fact.U)

# Preconditioner with sparse GPU storage
struct ILUPreconditioner{T,TL,TU}
    L_sparse::TL
    U_sparse::TU
    L_dense::CuArray{T,2}  # For triangular solve
    U_dense::CuArray{T,2}
    temp::CuVector{T}
end

function ILUPreconditioner(L_sparse::TL, U_sparse::TU) where {T,TL<:CuSparseMatrixCSC{T},TU<:CuSparseMatrixCSC{T}}
    n = size(L_sparse, 1)
    # Convert to dense only for solve (GPU triangular solve limitation)
    L_dense = CuArray(Matrix(L_sparse))
    U_dense = CuArray(Matrix(U_sparse))
    ILUPreconditioner{T,TL,TU}(L_sparse, U_sparse, L_dense, U_dense, CUDA.zeros(T, n))
end

function LinearAlgebra.ldiv!(y, P::ILUPreconditioner, x)
    # Apply ILU preconditioner: solve (L*U) \ x
    ldiv!(P.temp, LowerTriangular(P.L_dense), x)
    ldiv!(y, UpperTriangular(P.U_dense), P.temp)
    return y
end

P = ILUPreconditioner(L_gpu_sparse, U_gpu_sparse)

# Solve
atol = PREC == Float64 ? 1e-6 : PREC == Float32 ? 1f-6 : Float16(1e-4)
rtol = PREC == Float64 ? 1e-6 : PREC == Float32 ? 1f-6 : Float16(1e-4)

println("\nSolving with ILU-preconditioned GMRES...")
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
