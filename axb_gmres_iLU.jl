using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
using IncompleteLU, Krylov
using MatrixMarket

include("cli_args.jl")

# ===== PARSE CLI ARGUMENTS =====
opts = parse_commandline_args(; default_maxiter=100, default_rtol=1e-6, default_precision=Float64)
PREC = opts.precision
# ================================

# ===== LOAD OR GENERATE PROBLEM =====
if length(opts.positional) >= 2
    # Load from Matrix Market files: A.mtx b.mtx [x.mtx]
    path_A = opts.positional[1]
    path_b = opts.positional[2]

    isfile(path_A) || error("File not found: $path_A")
    isfile(path_b) || error("File not found: $path_b")

    println("Loading Matrix Market files...")
    A_raw = MatrixMarket.mmread(path_A)
    I_idx, J_idx, V = findnz(A_raw)
    A_cpu = sparse(I_idx, J_idx, Vector{PREC}(V), size(A_raw)...)
    b_cpu = Vector{PREC}(vec(MatrixMarket.mmread(path_b)))
    n = size(A_cpu, 1)

    if length(opts.positional) >= 3
        path_x = opts.positional[3]
        isfile(path_x) || error("File not found: $path_x")
        x_true = Vector{PREC}(vec(MatrixMarket.mmread(path_x)))
    else
        x_true = nothing
    end
else
    # Generate random SPD test problem
    Random.seed!(42)
    n = 100
    num_nonzeros = 200

    rows = rand(1:n, num_nonzeros)
    cols = rand(1:n, num_nonzeros)
    vals = rand(PREC, num_nonzeros)
    A_temp = sparse(rows, cols, vals, n, n)
    A_cpu = A_temp + A_temp' + PREC(20.0) * spdiagm(0 => ones(PREC, n))

    x_true = randn(PREC, n)
    b_cpu = A_cpu * x_true
end
# =====================================

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
struct SparseILUPreconditioner{T,TF}
    ilu_fact::TF
    temp_cpu::Vector{T}
end

function SparseILUPreconditioner(ilu_fact, ::Type{T}, n::Int) where T
    SparseILUPreconditioner{T, typeof(ilu_fact)}(ilu_fact, Vector{T}(undef, n))
end

function LinearAlgebra.ldiv!(y, P::SparseILUPreconditioner, x)
    # Transfer GPU -> CPU (only vectors, not matrices)
    x_cpu = Array(x)

    # Use IncompleteLU's own ldiv! which handles L (unit lower) and U correctly
    ldiv!(P.temp_cpu, P.ilu_fact, x_cpu)

    # Transfer CPU -> GPU
    copyto!(y, P.temp_cpu)
    return y
end

println("\nBuilding sparse ILU preconditioner (hybrid CPU/GPU)...")
P = SparseILUPreconditioner(ilu_fact, PREC, n)

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
println("Converged: ", stats.solved)
println("Iterations: ", stats.niter)
println("Final residual: ", stats.residuals[end])

x_sol = Array(x_gpu)
rel_residual = norm(A_cpu * x_sol - b_cpu) / norm(b_cpu)
println("Relative residual ||Ax-b||/||b||: ", rel_residual)

if x_true !== nothing
    rel_error = norm(x_sol - x_true) / norm(x_true)
    println("Relative error ||x-x_true||/||x_true||: ", rel_error)
end
println("="^50)
