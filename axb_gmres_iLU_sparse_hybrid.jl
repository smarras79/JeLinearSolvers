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
println("\nComputing ILU...")

# Try different ILU strategies
ilu_fact = nothing
strategy = ""

# Strategy 1: ILU(0) - no dropping, most robust
try
    ilu_fact = ilu(A_cpu, τ=PREC(0.0))  # No dropping
    strategy = "ILU(0) - no fill-in"
    println("✓ Using ILU(0) (no dropping)")
catch e
    println("⚠ ILU(0) failed: $e")
end

# Strategy 2: Small drop tolerance if ILU(0) didn't work
if isnothing(ilu_fact)
    try
        ilu_fact = ilu(A_cpu, τ=PREC(1e-4))  # Very small drop tolerance
        strategy = "ILU with τ=1e-4"
        println("✓ Using ILU with τ=1e-4")
    catch e
        println("⚠ ILU with τ=1e-4 failed: $e")
    end
end

# Strategy 3: Modified ILU with diagonal shift for stability
if isnothing(ilu_fact)
    println("⚠ Trying diagonal-shifted ILU...")
    A_shifted = A_cpu + PREC(1e-6) * sparse(I, n, n)
    ilu_fact = ilu(A_shifted, τ=PREC(0.0))
    strategy = "Diagonal-shifted ILU(0)"
end

# Verify diagonal is non-zero
L_diag = [ilu_fact.L[i,i] for i in 1:n]
U_diag = [ilu_fact.U[i,i] for i in 1:n]

if any(abs.(L_diag) .< eps(PREC) * 100) || any(abs.(U_diag) .< eps(PREC) * 100)
    error("ILU produced near-zero diagonal entries. Matrix may be ill-conditioned.")
end

nnz_L = SparseArrays.nnz(ilu_fact.L)
nnz_U = SparseArrays.nnz(ilu_fact.U)
nnz_total = SparseArrays.nnz(A_cpu)

println("\nILU sparsity statistics ($strategy):")
println("  L: $nnz_L nnz, diag range: [$(minimum(abs.(L_diag))), $(maximum(abs.(L_diag)))]")
println("  U: $nnz_U nnz, diag range: [$(minimum(abs.(U_diag))), $(maximum(abs.(U_diag)))]")
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
    # Transfer GPU -> CPU
    x_cpu = Array(x)
    
    # EXACT sparse triangular solves on CPU
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

# Test preconditioner before using in GMRES
println("\nTesting preconditioner...")
test_x = CuArray(randn(PREC, n))
test_y = similar(test_x)
ldiv!(test_y, P, test_x)
if any(isnan.(Array(test_y))) || any(isinf.(Array(test_y)))
    error("Preconditioner produces NaN/Inf values!")
end
println("✓ Preconditioner test passed")

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
println("COMPARISON: No Preconditioner vs ILU")
println("="^60)
x_noprecond, stats_noprecond = gmres(A_gpu, b_gpu; 
                                      atol=atol,
                                      rtol=rtol,
                                      restart=true,
                                      itmax=100,
                                      verbose=0)
println("Without preconditioner: $(stats_noprecond.niter) iterations")
println("With ILU preconditioner: $(stats.niter) iterations")
if stats.niter < stats_noprecond.niter
    println("✓ ILU speedup: $(round(stats_noprecond.niter / stats.niter, digits=2))x fewer iterations")
else
    println("⚠ ILU did not reduce iterations (may need tuning)")
end
println("="^60)
