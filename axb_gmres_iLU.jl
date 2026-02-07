using CUDA
using CUDAPreconditioners
using Krylov
using SparseArrays
using Random

function axb_gmres_iLU()
    
    # Set random seed for reproducibility
    Random.seed!(42)

    n = 100
    nnz = 200

    # Create symmetric positive definite matrix
    I = rand(1:n, nnz)
    J = rand(1:n, nnz)
    V = rand(nnz)  # Positive values
    A_temp = sparse(I, J, V, n, n)

    # Make symmetric and positive definite
    A_cpu = A_temp + A_temp' + 20.0 * I  # Diagonal dominance

    # Create right-hand side
    x_true = randn(n)
    b_cpu = A_cpu * x_true

    # Load sparse matrix A and vector b
    A_gpu = CuSparseMatrixCSR(A)
    b_gpu = CuVector(b)

    # Compute ILU(0) on GPU
    # CUDAPreconditioners.jl uses CUSPARSE ilu0
    P = ilu0(A_gpu)

    # Solve Ax = b using GMRES and the GPU preconditioner
    x_gpu, stats = gmres(A_gpu, b_gpu, M=P)
end

axb_gmres_iLU()
