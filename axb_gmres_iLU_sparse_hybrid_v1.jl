using SparseArrays, LinearAlgebra, Random, Printf, Dates
using CUDA, CUDA.CUSPARSE
using Krylov
using MatrixMarket
using Plots

include("cli_args.jl")

# ===== 1. PRECONDITIONER SETUP =====
struct GPUSparseILU{T}
    LU::CuSparseMatrixCSR{T, Int32}
    temp_z::CuVector{T} 
end

function GPUSparseILU(A_gpu::CuSparseMatrixCSR{T}) where T
    LU = copy(A_gpu)
    CUSPARSE.ilu02!(LU) 
    return GPUSparseILU{T}(LU, CuVector{T}(undef, size(A_gpu, 1)))
end

function LinearAlgebra.ldiv!(y, P::GPUSparseILU, x)
    ldiv!(P.temp_z, UnitLowerTriangular(P.LU), x)
    ldiv!(y, UpperTriangular(P.LU), P.temp_z)
    return y
end

# ===== 2. DATA LOADING =====
function load_system_from_files(path_A, path_b, path_x, PREC)
    A_raw = MatrixMarket.mmread(path_A)
    I, J, V = findnz(A_raw)
    A = sparse(I, J, Vector{PREC}(V), size(A_raw)...)
    b = Vector{PREC}(vec(MatrixMarket.mmread(path_b)))
    xt = Vector{PREC}(vec(MatrixMarket.mmread(path_x)))
    return A, b, xt
end

# ===== 3. FLEXIBLE SOLVER FUNCTION =====
function solve_linear_system(A_cpu, b_cpu, x_true, solver_type="dqgmres";
                             PREC=Float64, myepsilon=1e-6, maxiter::Int=3000, rtol::Float64=1e-8)
    n = length(b_cpu)
    
    # Pre-processing: Slightly more aggressive shift for stability
    epsilon = PREC == Float64 ? myepsilon : 1f-4 
    println("\nPreprocessing: Scaling & Shift (ϵ = $epsilon)...")
    
    diag_A = diag(A_cpu)
    d = [1.0 / sqrt(abs(val) + epsilon) for val in diag_A]
    D = sparse(1:n, 1:n, PREC.(d))
    
    A_scaled_cpu = D * (A_cpu + epsilon * I) * D
    b_scaled_cpu = D * b_cpu
    
    A_gpu = CuSparseMatrixCSR(A_scaled_cpu)
    b_gpu = CuArray(b_scaled_cpu)
    P = GPUSparseILU(A_gpu)

    p = plot([0], [1.0], title="Convergence: $solver_type", xlabel="Iteration", 
             ylabel="Relative Residual", yaxis=:log10, lw=2, grid=true)
    display(p)

    println("Selected Solver: $solver_type")
    
    t_solve = @elapsed begin
        cb = (solver) -> begin
            it = solver.stats.niter
            if it > 0 && it % 10 == 0 && !isempty(solver.stats.residuals)
                res = solver.stats.residuals[end] / solver.stats.residuals[1]
                push!(p, it, res)
                gui(p)
            end
            return false
        end

        # DISPATCH LOGIC
        solve_rtol = PREC(rtol)
        y_gpu, stats = if solver_type == "gmres"
            Krylov.gmres(A_gpu, b_gpu; M=P, ldiv=true, restart=true, memory=100,
                         itmax=maxiter, rtol=solve_rtol, callback=cb, verbose=1)
        elseif solver_type == "bicgstab"
            Krylov.bicgstab(A_gpu, b_gpu; M=P, ldiv=true,
                            itmax=maxiter, rtol=solve_rtol, callback=cb, verbose=1)
        elseif solver_type == "dqgmres"
            # DQGMRES is very stable for large systems
            Krylov.dqgmres(A_gpu, b_gpu; M=P, ldiv=true, memory=100,
                           itmax=maxiter, rtol=solve_rtol, callback=cb, verbose=1)
        else
            error("Solver $solver_type not available or not recognized.")
        end
    end

    x_cpu = D * Array(y_gpu)
    err = norm(x_cpu - x_true) / norm(x_true)
    
    @printf("\nFinal Result: %s | Time: %.2f s | Error: %.2e\n", 
            stats.solved ? "CONVERGED" : "FAILED", t_solve, err)
    
    return x_cpu, stats
end

# ===== 4. EXECUTION =====
opts = parse_commandline_args(; default_maxiter=3000, default_rtol=1e-8, default_precision=Float64)

# Determine file paths: from CLI positional args, or fall back to defaults
if length(opts.positional) >= 2
    fA = opts.positional[1]
    fb = opts.positional[2]
    fx = length(opts.positional) >= 3 ? opts.positional[3] : nothing
else
    fA = "sparse_Abx_data_A.mtx"
    fb = "sparse_Abx_data_b.mtx"
    fx = "sparse_Abx_data_x.mtx"
end

if isfile(fA)
    isfile(fb) || error("File not found: $fb")
    if fx !== nothing
        isfile(fx) || error("File not found: $fx")
        A_orig, b_orig, xt_orig = load_system_from_files(fA, fb, fx, opts.precision)
    else
        A_raw = MatrixMarket.mmread(fA)
        I_idx, J_idx, V = findnz(A_raw)
        A_orig = sparse(I_idx, J_idx, Vector{opts.precision}(V), size(A_raw)...)
        b_orig = Vector{opts.precision}(vec(MatrixMarket.mmread(fb)))
        xt_orig = zeros(opts.precision, size(A_orig, 1))
    end

    epsilon = 1e-8

    x_final, stats = solve_linear_system(A_orig, b_orig, xt_orig, "dqgmres";
                                         PREC=opts.precision, myepsilon=epsilon,
                                         maxiter=opts.maxiter, rtol=opts.rtol)

    MatrixMarket.mmwrite("x_out.mtx", sparse(reshape(x_final, :, 1)))
else
    println("Matrix file not found: $fA")
    println("Usage: julia --project=. $(PROGRAM_FILE) [options] A.mtx b.mtx [x.mtx]")
end
