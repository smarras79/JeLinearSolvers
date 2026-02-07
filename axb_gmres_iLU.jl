using SparseArrays
using LinearAlgebra
using Random
using CUDA, CUDA.CUSPARSE
using IncompleteLU
using Krylov

# Set random seed for reproducibility
Random.seed!(42)
n = 100          # Matrix dimension
nnz = 200        # Number of non-zeros (not dimension!)

# Create symmetric positive definite matrix
rows = rand(1:n, nnz)
cols = rand(1:n, nnz)
vals = rand(nnz)

A_temp = sparse(rows, cols, vals, n, n)

# Make symmetric and positive definite
A_cpu = A_temp + A_temp' + 20.0 * spdiagm(0 => ones(n))

# Create right-hand side - make sure it uses n, not nnz!
x_true = randn(n)      # Length n = 100
b_cpu = A_cpu * x_true # Length n = 100

# Verify dimensions
println("A size: ", size(A_cpu))
println("b size: ", size(b_cpu))
println("x_true size: ", size(x_true))

# Compute ILU on CPU
ilu_fact = ilu(A_cpu, τ=0.01)

# Transfer to GPU
A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuArray(b_cpu)
L_gpu = CuSparseMatrixCSR(ilu_fact.L)
U_gpu = CuSparseMatrixCSR(ilu_fact.U)

# Preconditioner operator
struct ILUPreconditioner{TL,TU}
    L::TL
    U::TU
    temp::CuVector{Float64}
end

function LinearAlgebra.ldiv!(y, P::ILUPreconditioner, x)
    CUDA.CUSPARSE.sv2!('N', 'L', 'U', 1.0, P.L, x, P.temp, 'O')
    CUDA.CUSPARSE.sv2!('N', 'U', 'N', 1.0, P.U, P.temp, y, 'O')
    return y
end

# Create preconditioner
P = ILUPreconditioner(L_gpu, U_gpu, CUDA.zeros(n))

# Solve
x_gpu, stats = gmres(A_gpu, b_gpu, M=P, verbose=1, restart=30, atol=1e-6)

println("Converged in $(stats.niter) iterations")
println("Residual: $(stats.residuals[end])")
