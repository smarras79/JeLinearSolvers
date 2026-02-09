# ===== COMMAND LINE ARGUMENT PARSING =====
# Shared utility for all solver scripts.
# Usage: include("cli_args.jl")
#        opts = parse_commandline_args()
#        # opts.maxiter, opts.rtol, opts.precision

const PRECISION_MAP = Dict(
    "float64" => Float64,
    "float32" => Float32,
    "float16" => Float16,
)

"""
    parse_commandline_args(args=ARGS; default_maxiter, default_rtol, default_precision)

Parse command line arguments for the linear solver scripts.

Supported arguments:
  --maxiter, -m N          Maximum solver iterations
  --rtol, -r VAL           Relative convergence tolerance
  --precision, -p TYPE     Floating-point precision: Float64, Float32, or Float16
  --help, -h               Print usage and exit
"""
function parse_commandline_args(args=ARGS;
                                default_maxiter::Int = 2500,
                                default_rtol::Float64 = 1e-8,
                                default_precision::DataType = Float64)
    maxiter   = default_maxiter
    rtol      = default_rtol
    precision = default_precision

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("--maxiter", "-m")
            i + 1 > length(args) && error("--maxiter requires a value")
            maxiter = parse(Int, args[i+1])
            maxiter <= 0 && error("--maxiter must be positive, got $maxiter")
            i += 2
        elseif arg in ("--rtol", "-r")
            i + 1 > length(args) && error("--rtol requires a value")
            rtol = parse(Float64, args[i+1])
            rtol <= 0 && error("--rtol must be positive, got $rtol")
            i += 2
        elseif arg in ("--precision", "-p")
            i + 1 > length(args) && error("--precision requires a value")
            key = lowercase(args[i+1])
            haskey(PRECISION_MAP, key) || error("Unknown precision '$(args[i+1])'. Use Float64, Float32, or Float16.")
            precision = PRECISION_MAP[key]
            i += 2
        elseif arg in ("--help", "-h")
            println("Usage: julia [--project=.] script.jl [options]")
            println()
            println("Options:")
            println("  --maxiter, -m N       Maximum solver iterations (default: $default_maxiter)")
            println("  --rtol, -r VAL        Relative convergence tolerance (default: $default_rtol)")
            println("  --precision, -p TYPE  Precision: Float64, Float32, Float16 (default: $default_precision)")
            println("  --help, -h            Show this help message")
            exit(0)
        else
            error("Unknown argument: '$arg'. Use --help for usage information.")
        end
    end

    println("CLI arguments: maxiter=$maxiter, rtol=$rtol, precision=$precision")
    return (maxiter=maxiter, rtol=rtol, precision=precision)
end
