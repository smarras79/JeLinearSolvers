using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
using IncompleteLU, Krylov

include("cli_args.jl")

# ===== PARSE CLI ARGUMENTS =====
opts = parse_commandline_args(; default_maxiter=100, default_rtol=1e-6, default_precision=Float64)
PREC = opts.precision
# ================================

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

println("A size: ", size(A_cpu))
println("b size: ", size(b_cpu))
println("Precision: ", PREC)

# ===== INCOMPLETE LU FACTORIZATION =====
println("\nComputing ILU(τ=$(PREC(0.01)))...")
ilu_fact = ilu(A_cpu, τ=PREC(0.01))

nnz_L = SparseArrays.nnz(ilu_fact.L)
nnz_U = SparseArrays.nnz(ilu_fact.U)
println("ILU sparsity: L has $nnz_L nnz, U has $nnz_U nnz (vs $(n*n) for dense)")
# =======================================

# Transfer to GPU
A_gpu = CuSparseMatrixCSR(A_cpu)
b_gpu = CuArray(b_cpu)

# Sparse ILU Preconditioner - Hybrid CPU/GPU approach
# ILU factors stay sparse on CPU, only vectors transferred
struct SparseILUPreconditioner{T,TL,TU}
    L_cpu::TL  # Sparse L on CPU
    U_cpu::TU  # Sparse U on CPU  
    temp_cpu::Vector{T}
end

function SparseILUPreconditioner(L_cpu::TL, U_cpu::TU) where {T,TL<:SparseMatrixCSC{T},TU<:SparseMatrixCSC{T}}
    n = size(L_cpu, 1)
    SparseILUPreconditioner{T,TL,TU}(L_cpu, U_cpu, Vector{T}(undef, n))
end

function LinearAlgebra.ldiv!(y, P::SparseILUPreconditioner, x)
    # Transfer GPU -> CPU (only vectors, not matrices)
    x_cpu = Array(x)
    
    # Sparse triangular solves on CPU (fast for sparse L, U)
    # Forward solve: L \ x
    ldiv!(P.temp_cpu, LowerTriangular(P.L_cpu), x_cpu)
    
    # Backward solve: U \ temp
    y_cpu = P.U_cpu \ P.temp_cpu
    
    # Transfer CPU -> GPU
    copyto!(y, y_cpu)
    return y
end

println("\nBuilding sparse ILU preconditioner (hybrid CPU/GPU)...")
P = SparseILUPreconditioner(ilu_fact.L, ilu_fact.U)

# Solve
solve_rtol = PREC(opts.rtol)
solve_atol = PREC(opts.rtol)

println("\nSolving with sparse ILU-preconditioned GMRES...")
println("(ILU stored sparse on CPU, matrix-vector on GPU)")
println("  maxiter=$(opts.maxiter), rtol=$solve_rtol")

x_gpu, stats = gmres(A_gpu, b_gpu;
                     M=P,
                     ldiv=true,
                     atol=solve_atol,
                     rtol=solve_rtol,
                     restart=true,
                     itmax=opts.maxiter,
                     verbose=1,
                     history=true)

println("\n" * "="^50)
println("✓ Converged: ", stats.solved)
println("✓ Iterations: ", stats.niter)
println("✓ Final residual: ", stats.residuals[end])

x_cpu = Array(x_gpu)
error = norm(A_cpu * x_cpu - b_cpu) / norm(b_cpu)
println("✓ Relative error: ", error)
println("="^50)
