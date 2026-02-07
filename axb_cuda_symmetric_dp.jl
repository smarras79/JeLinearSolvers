using SparseArrays, Krylov, LinearOperators
using CUDA, CUDA.CUSPARSE
using LinearAlgebra

# Define ldiv_ic0! at the top level
function ldiv_ic0!(P::CuSparseMatrixCSR, x, y, z)
    ldiv!(z, LowerTriangular(P), x)   # Forward substitution with L
    ldiv!(y, LowerTriangular(P)', z)  # Backward substitution with Lᴴ
    return y
end

function ldiv_ic0!(P::CuSparseMatrixCSC, x, y, z)
    ldiv!(z, UpperTriangular(P)', x)  # Forward substitution with L
    ldiv!(y, UpperTriangular(P), z)   # Backward substitution with Lᴴ
    return y
end

function setup_gpu_preconditioner(A_cpu, b_cpu; α=1e-8)
    n = size(A_cpu, 1)
    T = eltype(b_cpu)
    
    # Try IC(0) with regularization
    try
        A_reg = A_cpu + α * I
        A_gpu_reg = CuSparseMatrixCSR(A_reg)
        P = ic02(A_gpu_reg)
        
        z = CUDA.zeros(T, n)
        
        opM = LinearOperator(T, n, n, true, true, 
                            (y, x) -> ldiv_ic0!(P, x, y, z))
        
        A_gpu = CuSparseMatrixCSR(A_cpu)  # Return original unshifted matrix
        b_gpu = CuVector(b_cpu)
        
        return A_gpu, b_gpu, opM, :ic0
    catch e
        @warn "IC(0) failed, falling back to Jacobi" exception=e
        
        # Fallback to Jacobi
        A_gpu = CuSparseMatrixCSR(A_cpu)
        b_gpu = CuVector(b_cpu)
        d_inv = CuVector([1.0 / A_cpu[i,i] for i in 1:n])
        opM = LinearOperator(T, n, n, true, true, 
                            (y, x) -> (y .= d_inv .* x))
        
        return A_gpu, b_gpu, opM, :jacobi
    end
end

# Usage
if CUDA.functional()
    A_gpu, b_gpu, opM, prec_type = setup_gpu_preconditioner(A_cpu, b_cpu, α=1e-6)
    
    # Solve with history enabled and higher iteration limit
    x, stats = cg(A_gpu, b_gpu, M=opM, 
                  itmax=1000,      # Increase iteration limit
                  atol=1e-8,       # Absolute tolerance
                  rtol=1e-6,       # Relative tolerance
                  history=true,    # IMPORTANT: Enable residual history
                  verbose=1)       # Show iteration progress
    
    println("\n" * "="^60)
    println("Used preconditioner: ", prec_type)
    println("Converged: ", stats.solved)
    println("Iterations: ", stats.niter)
    
    if length(stats.residuals) > 0
        println("Final residual: ", stats.residuals[end])
        println("Initial residual: ", stats.residuals[1])
        println("Reduction factor: ", stats.residuals[end] / stats.residuals[1])
    else
        println("No residual history available")
    end
    println("="^60)
end