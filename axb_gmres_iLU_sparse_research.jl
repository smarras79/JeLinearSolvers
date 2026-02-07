using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
using IncompleteLU, Krylov

# ===== SET PRECISION HERE =====
const PREC = Float32  # Float64, Float32, or Float16
# ==============================

Random.seed!(42)
n = 100
num_nonzeros = 200

# Create SPD matrix
rows = rand(1:n, num_nonzeros)
cols = rand(1:n, num_nonzeros)
vals = rand(PREC, num_nonzeros)
A_temp = sparse(rows, cols, vals, n, n)
A_cpu = A_temp + A_temp' + PREC(20.0) * spdiagm(0 => ones(PREC, n))

x_true = randn(PREC, n)
b_cpu = A_cpu * x_true

println("="^60)
println("Mixed Precision ILU-Preconditioned GPU Solver")
println("="^60)
println("Matrix size: ", size(A_cpu))
println("RHS size: ", size(b_cpu))
println("Precision: ", PREC)

# ===== INCOMPLETE LU FACTORIZATION =====
println("\nComputing ILU(τ=$(PREC(0.01)))...")
ilu_fact = ilu(A_cpu, τ=PREC(0.01))

nnz_L = SparseArrays.nnz(ilu_fact.L)
nnz_U = SparseArrays.nnz(ilu_fact.U)
nnz_total = SparseArrays.nnz(A_cpu)
fill_ratio = (nnz_L + nnz_U) / (2 * nnz_total)

println("ILU sparsity statistics:")
println("  L: $nnz_L nnz")
println("  U: $nnz_U nnz") 
println("  Original A: $nnz_total nnz")
println("  Fill ratio: $(round(fill_ratio, digits=2))x")
println("  Memory saving vs dense: $(round(100*(1 - (nnz_L+nnz_U)/(n*n)), digits=1))%")
# =======================================

# Transfer to GPU - KEEP SPARSE
A_gpu = CuSparseMatrixCSC(A_cpu)
b_gpu = CuArray(b_cpu)
L_gpu = CuSparseMatrixCSC(ilu_fact.L)
U_gpu = CuSparseMatrixCSC(ilu_fact.U)

println("\nGPU memory allocated:")
println("  A: sparse CSC ($(SparseArrays.nnz(A_cpu)) nnz)")
println("  L: sparse CSC ($nnz_L nnz)")
println("  U: sparse CSC ($nnz_U nnz)")

# Fully Sparse GPU ILU Preconditioner using iterative refinement
struct FullySparseGPUILU{T,TM}
    L::TM  # Sparse lower triangular on GPU
    U::TM  # Sparse upper triangular on GPU
    temp::CuVector{T}
    temp2::CuVector{T}
    max_iter::Int  # Iterations for iterative triangular solve
end

function FullySparseGPUILU(L::TM, U::TM; max_iter=5) where {T,TM<:CuSparseMatrixCSC{T}}
    n = size(L, 1)
    FullySparseGPUILU{T,TM}(
        L, U,
        CUDA.zeros(T, n),
        CUDA.zeros(T, n),
        max_iter
    )
end

# Sparse triangular solve using Richardson iteration on GPU
function solve_lower_triangular!(y, L_sparse, x, max_iter, temp)
    # Solve L*y = x using Richardson iteration
    # L = D + L_strict where D is diagonal, L_strict is strictly lower
    n = length(x)
    
    # Extract diagonal of L for preconditioning
    D_inv = CuArray([1 / L_sparse[i,i] for i in 1:n])
    
    # Initialize y = D^{-1} * x
    y .= D_inv .* x
    
    # Richardson iteration: y^{k+1} = y^k + D^{-1}(x - L*y^k)
    for iter in 1:max_iter
        mul!(temp, L_sparse, y)  # temp = L*y
        temp .= x .- temp         # temp = x - L*y (residual)
        y .+= D_inv .* temp       # y = y + D^{-1}*residual
    end
    
    return y
end

function solve_upper_triangular!(y, U_sparse, x, max_iter, temp)
    # Solve U*y = x using Richardson iteration
    n = length(x)
    
    # Extract diagonal of U for preconditioning
    D_inv = CuArray([1 / U_sparse[i,i] for i in 1:n])
    
    # Initialize y = D^{-1} * x
    y .= D_inv .* x
    
    # Richardson iteration: y^{k+1} = y^k + D^{-1}(x - U*y^k)
    for iter in 1:max_iter
        mul!(temp, U_sparse, y)  # temp = U*y
        temp .= x .- temp         # temp = x - U*y (residual)
        y .+= D_inv .* temp       # y = y + D^{-1}*residual
    end
    
    return y
end

function LinearAlgebra.ldiv!(y, P::FullySparseGPUILU, x)
    # Forward solve: L * temp = x
    solve_lower_triangular!(P.temp, P.L, x, P.max_iter, P.temp2)
    
    # Backward solve: U * y = temp
    solve_upper_triangular!(y, P.U, P.temp, P.max_iter, P.temp2)
    
    return y
end

println("\nBuilding fully sparse GPU ILU preconditioner...")
P = FullySparseGPUILU(L_gpu, U_gpu, max_iter=5)

# Solve
atol = PREC == Float64 ? 1e-6 : PREC == Float32 ? 1f-6 : Float16(1e-4)
rtol = PREC == Float64 ? 1e-6 : PREC == Float32 ? 1f-6 : Float16(1e-4)

println("\nSolving with fully sparse GPU ILU-preconditioned GMRES...")
println("All operations on GPU, L and U kept sparse")

x_gpu, stats = gmres(A_gpu, b_gpu; 
                     M=P, 
                     ldiv=true,
                     atol=atol,
                     rtol=rtol,
                     restart=true,
                     itmax=100,
                     verbose=1,
                     history=true)

println("\n" * "="^60)
println("RESULTS")
println("="^60)
println("✓ Converged: ", stats.solved)
println("✓ GMRES iterations: ", stats.niter)
println("✓ Final residual: ", stats.residuals[end])

x_cpu = Array(x_gpu)
error = norm(A_cpu * x_cpu - b_cpu) / norm(b_cpu)
println("✓ Relative error: ", error)
println("✓ Precision used: ", PREC)
println("="^60)
