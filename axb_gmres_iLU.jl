using CUDA, IncompleteLU, Krylov, SparseArrays, LinearAlgebra, Random

Random.seed!(42)
n = 100
nnz = 200

# Create symmetric positive definite matrix
rows = rand(1:n, nnz)
cols = rand(1:n, nnz)
vals = rand(nnz)

A_temp = sparse(rows, cols, vals, n, n)

# Make symmetric and positive definite
A_cpu = A_temp + A_temp' + 20.0I  # I from LinearAlgebra works on sparse

# Create right-hand side
x_true = randn(n)
b_cpu = A_cpu * x_true

# Compute ILU on CPU (one-time cost)
ilu_fact = ilu(A_cpu, τ=0.01)  # Adjust τ for sparsity/accuracy tradeoff

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

function ILUPreconditioner(L, U, n)
    ILUPreconditioner(L, U, CUDA.zeros(n))
end

function LinearAlgebra.ldiv!(y, P::ILUPreconditioner, x)
    # Forward solve: L * temp = x
    CUDA.CUSPARSE.sv2!('N', 'L', 'U', 1.0, P.L, x, P.temp, 'O')
    # Backward solve: U * y = temp  
    CUDA.CUSPARSE.sv2!('N', 'U', 'N', 1.0, P.U, P.temp, y, 'O')
    return y
end

# Create preconditioner
P = ILUPreconditioner(L_gpu, U_gpu, length(b_gpu))

# Solve
x, stats = gmres(A_gpu, b_gpu, M=P, verbose=1, restart=30, atol=1e-6)

println("Converged in $(stats.niter) iterations")
