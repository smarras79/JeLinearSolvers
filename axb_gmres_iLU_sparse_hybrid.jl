using SparseArrays, LinearAlgebra, Random
using CUDA, CUDA.CUSPARSE
using IncompleteLU, Krylov

function create_test_problem(n, PREC)
    # Create 2D Laplacian-like matrix (realistic CFD/PDE problem)
    nx = ny = Int(sqrt(n))
    actual_n = nx * ny
    
    rows = Int[]
    cols = Int[]
    vals = PREC[]
    
    for j in 1:ny
        for i in 1:nx
            k = i + (j-1)*nx
            
            # Diagonal (4-connectivity)
            push!(rows, k); push!(cols, k); push!(vals, PREC(4.0))
            
            # Left neighbor
            if i > 1
                push!(rows, k); push!(cols, k-1); push!(vals, PREC(-1.0))
            end
            # Right neighbor
            if i < nx
                push!(rows, k); push!(cols, k+1); push!(vals, PREC(-1.0))
            end
            # Bottom neighbor
            if j > 1
                push!(rows, k); push!(cols, k-nx); push!(vals, PREC(-1.0))
            end
            # Top neighbor
            if j < ny
                push!(rows, k); push!(cols, k+nx); push!(vals, PREC(-1.0))
            end
        end
    end
    
    A = sparse(rows, cols, vals, actual_n, actual_n)
    x_true = randn(PREC, actual_n)
    b = A * x_true
    
    return A, b, x_true
end

#
# TEST MIXED PRECISION
#
for PREC in [Float64, Float32, Float16]
    
    println("\n" * "="^60)
    println("Testing with precision: $PREC")
    println("="^60)
    
    A_cpu, b_cpu, x_true = create_test_problem(10000, PREC)  # 100×100 grid

    min_diag_A = minimum(abs.([A_cpu[i,i] for i in 1:n]))
    println("Original matrix A min diagonal: $min_diag_A")

    x_true = randn(PREC, n)
    b_cpu = A_cpu * x_true

    println("="^60)
    println("Mixed Precision ILU-Preconditioned GPU Solver")
    println("="^60)
    println("Matrix size: ", size(A_cpu))
    println("RHS size: ", size(b_cpu))
    println("Precision: ", PREC)
    println("Condition number estimate: ", cond(Matrix(A_cpu)))

    # ===== INCOMPLETE LU FACTORIZATION =====
    println("\nComputing ILU...")

    ilu_fact = try
        ilu(A_cpu, τ=PREC(0.0))
    catch e
        println("Standard ILU failed: $e")
        nothing
    end

    if isnothing(ilu_fact)
        println("⚠ Using diagonal preconditioner instead")
        D = [A_cpu[i,i] for i in 1:n]
        L_fact = sparse(Diagonal(sqrt.(D)))
        U_fact = sparse(Diagonal(sqrt.(D)))
        ilu_fact = (L = L_fact, U = U_fact)
        strategy = "Diagonal preconditioner"
    else
        strategy = "ILU(0)"
        println("✓ Using ILU(0)")
    end

    # Verify diagonal is non-zero
    L_diag = [ilu_fact.L[i,i] for i in 1:n]
    U_diag = [ilu_fact.U[i,i] for i in 1:n]

    min_L_diag = minimum(abs.(L_diag))
    min_U_diag = minimum(abs.(U_diag))
    max_L_diag = maximum(abs.(L_diag))
    max_U_diag = maximum(abs.(U_diag))

    println("\nDiagonal check:")
    println("  L diagonal range: [$min_L_diag, $max_L_diag]")
    println("  U diagonal range: [$min_U_diag, $max_U_diag]")

    # Replace any zero diagonals with small values
    if min_L_diag < eps(PREC) * 1000
        println("⚠ Warning: Found near-zero L diagonals, fixing...")
        L_fact = copy(ilu_fact.L)
        for i in 1:n
            if abs(L_fact[i,i]) < eps(PREC) * 1000
                L_fact[i,i] = PREC(1.0)
            end
        end
        ilu_fact = (L = L_fact, U = ilu_fact.U)
    end

    if min_U_diag < eps(PREC) * 1000
        println("⚠ Warning: Found near-zero U diagonals, fixing...")
        U_fact = copy(ilu_fact.U)
        for i in 1:n
            if abs(U_fact[i,i]) < eps(PREC) * 1000
                U_fact[i,i] = PREC(1.0)
            end
        end
        ilu_fact = (L = ilu_fact.L, U = U_fact)
    end

    nnz_L = SparseArrays.nnz(ilu_fact.L)
    nnz_U = SparseArrays.nnz(ilu_fact.U)
    nnz_total = SparseArrays.nnz(A_cpu)

    println("\nILU sparsity statistics ($strategy):")
    println("  L: $nnz_L nnz")
    println("  U: $nnz_U nnz")
    println("  Original A: $nnz_total nnz")
    println("  Memory saving vs dense: $(round(100*(1 - (nnz_L+nnz_U)/(n*n)), digits=1))%")

    # Transfer to GPU
    A_gpu = CuSparseMatrixCSC(A_cpu)
    b_gpu = CuArray(b_cpu)

    println("\nMemory strategy:")
    println("  A: sparse on GPU ($(nnz_total) nnz)")
    println("  L, U: sparse on CPU ($nnz_L + $nnz_U nnz)")
    println("  Strategy: Hybrid - matvec on GPU, ILU solve on CPU")

    # Hybrid ILU Preconditioner
    struct HybridSparseILU{T,TL,TU}
        L_cpu::TL
        U_cpu::TU
        temp_cpu::Vector{T}
    end

    function HybridSparseILU(L_cpu::TL, U_cpu::TU) where {T,TL<:SparseMatrixCSC{T},TU<:SparseMatrixCSC{T}}
        n = size(L_cpu, 1)
        HybridSparseILU{T,TL,TU}(L_cpu, U_cpu, Vector{T}(undef, n))
    end

    function LinearAlgebra.ldiv!(y, P::HybridSparseILU, x)
        x_cpu = Array(x)
        ldiv!(P.temp_cpu, LowerTriangular(P.L_cpu), x_cpu)
        y_cpu = P.U_cpu \ P.temp_cpu
        copyto!(y, y_cpu)
        return y
    end

    println("\nBuilding hybrid sparse ILU preconditioner...")
    P = HybridSparseILU(ilu_fact.L, ilu_fact.U)

    # Test preconditioner
    println("\nTesting preconditioner...")
    test_x = CuArray(randn(PREC, n))
    test_y = similar(test_x)
    ldiv!(test_y, P, test_x)
    test_y_cpu = Array(test_y)

    if any(isnan.(test_y_cpu)) || any(isinf.(test_y_cpu))
        Base.error("Preconditioner produces NaN/Inf values!")  # Use Base.error explicitly
    end
    println("✓ Preconditioner test passed")
    println("  Input norm: $(norm(Array(test_x)))")
    println("  Output norm: $(norm(test_y_cpu))")

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
    rel_error = norm(A_cpu * x_cpu - b_cpu) / norm(b_cpu)  # ← RENAMED from 'error' to 'rel_error'
    println("✓ Relative error: ", rel_error)
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
        println("⚠ ILU did not reduce iterations")
    end
    println("="^60)

end
