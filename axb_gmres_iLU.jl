using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
using IncompleteLU, Krylov

# ===== SET PRECISION HERE =====
const PREC = Float32  # Change to Float64, Float32, or Float16
# ==============================

# Set random seed for reproducibility
Random.seed!(42)
n = 100          # Matrix dimension
nnz = 200        # Number of non-zeros

# Create symmetric positive definite matrix
rows = rand(1:n, nnz)
cols = rand(1:n, nnz)
vals = rand(PREC, nnz)  # Use specified precision

A_temp = sparse(rows, cols, vals, n, n)

# Make symmetric and positive definite
A_cpu = A_temp + A_temp' + PREC(20.0) * spdiagm(0 => ones(PREC, n))

# Create right-hand side
x_true = randn(PREC, n)
b_cpu = A_cpu * x_true

# Verify dimensions
println("A size: ", size(A_cpu))
println("b size: ", size(b_cpu))
println("x_true size: ", size(x_true))
println("Precision: ", PREC)

# Compute ILU on CPU
ilu_fact = ilu(A_cpu, τ=PREC(0.01))

# Transfer to GPU
A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuArray(b_cpu)
L_gpu = CuSparseMatrixCSR(ilu_fact.L)
U_gpu = CuSparseMatrixCSR(ilu_fact.U)

# Generic preconditioner operator
struct ILUPreconditioner{T,TL,TU}
    L::TL
    U::TU
    temp::CuVector{T}
end

function ILUPreconditioner(L::TL, U::TU) where {T,TL<:CuSparseMatrixCSR{T},TU<:CuSparseMatrixCSR{T}}
    n = size(L, 1)
    ILUPreconditioner{T,TL,TU}(L, U, CUDA.zeros(T, n))
end

function LinearAlgebra.ldiv!(y, P::ILUPreconditioner, x)
    CUSPARSE.sv2!('N', 'L', 'U', one(eltype(P.L)), P.L, x, P.temp, 'O')
    CUSPARSE.sv2!('N', 'U', 'N', one(eltype(P.U)), P.U, P.temp, y, 'O')
    return y
end

# Create preconditioner
P = ILUPreconditioner(L_gpu, U_gpu)

# Solve
x_gpu, stats = gmres(A_gpu, b_gpu, M=P, verbose=1, restart=30, atol=PREC(1e-6))

println("Converged in $(stats.niter) iterations")
println("Residual: $(stats.residuals[end])")

# Verify solution
x_cpu = Array(x_gpu)
error = norm(A_cpu * x_cpu - b_cpu) / norm(b_cpu)
println("Relative error: ", error)
