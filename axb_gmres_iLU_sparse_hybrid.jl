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

println("ILU sparsity statistics:")
println("  L: $nnz_L nnz")
println("  U: $nnz_U nnz") 
println("  Original A: $nnz_total nnz")
println("  Memory saving vs dense: $(round(100*(1 - (nnz_L+nnz_U)/(n*n)), digits=1))%")
# =======================================

# Transfer matrix to GPU, keep ILU on CPU (hybrid approach)
A_gpu = CuSparseMatrixCSC(A_cpu)
b_gpu = CuArray(b_cpu)

println("\nMemory strategy:")
println("  A: sparse on GPU ($(nnz_total) nnz)")
println("  L, U: sparse on CPU ($nnz_L + $nnz_U nnz)")
println("  Strategy: Hybrid - matvec on GPU, ILU solve on CPU")

# Hybrid ILU Preconditioner - SPARSE on CPU
struct HybridSparseILU{T,TL,TU}
    L_cpu::TL  # Sparse L on CPU
    U_cpu::TU  # Sparse U on CPU
    temp_cpu::Vector{T}
end

function HybridSparseILU(L_cpu::TL, U_cpu::TU) where {T,TL<:SparseMatrixCSC{T},TU<:SparseMatrixCSC{T}}
    n = size(L_cpu, 1)
    HybridSparseILU{T,TL,TU}(L_cpu, U_cpu, Vector{T}(undef, n))
end

function LinearAlgebra.ldiv!(y, P::HybridSparseILU, x)
    # Transfer GPU -> CPU (only the vector x, not the matrices!)
    x_cpu = Array(x)
    
    # EXACT sparse triangular solves on CPU (fast and numerically stable)
    # Forward solve: L \ x
    ldiv!(P.temp_cpu, LowerTriangular(P.L_cpu), x_cpu)
    
    # Backward solve: U \ temp
    y_cpu = P.U_cpu \ P.temp_cpu
    
    # Transfer result CPU -> GPU
    copyto!(y, y_cpu)
    return y
end

println("\nBuilding hybrid sparse ILU preconditioner...")
P = HybridSparseILU(ilu_fact.L, ilu_fact.U)

# Solve
atol = PREC == Float64 ? 1e-6 : PREC == Float32 ? 1f-6 : Float16(1e-4)
rtol = PREC == Float64 ? 1e-6 : PREC == Float32 ? 1f-6 : Float16(1e-4)

println("\nSolving with hybrid sparse ILU-preconditioned GMRES...")

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

# Comparison
println("\n" * "="^60)
println("COMPARISON: No Preconditioner")
println("="^60)
x_noprecond, stats_noprecond = gmres(A_gpu, b_gpu; 
                                      atol=atol,
                                      rtol=rtol,
                                      restart=true,
                                      itmax=100,
                                      verbose=0)
println("Without ILU: $(stats_noprecond.niter) iterations")
println("With ILU: $(stats.niter) iterations")
if stats.niter < stats_noprecond.niter
    println("✓ Speedup: $(round(stats_noprecond.niter / stats.niter, digits=2))x reduction in iterations")
else
    println("⚠ Preconditioner did not reduce iterations")
end
println("="^60)
