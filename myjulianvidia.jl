using CUDA

# 1. Check if a CUDA-capable GPU is actually functional
if CUDA.functional()
    println("Status: GPU is available and functional!")
    
    # 2. Identify the specific hardware
    dev = CUDA.device()
    println("Device Name: ", CUDA.name(dev))
    
    # 3. The "Hello World" of GPU Computing: Vector Addition
    # Create arrays on the CPU (Host)
    N = 10^6
    a_host = ones(Float32, N)
    b_host = ones(Float32, N)

    # Move them to the GPU (Device)
    # The 'CuArray' type tells Julia to allocate this memory on the VRAM
    a_device = CuArray(a_host)
    b_device = CuArray(b_host)

    # Perform addition on the GPU
    # The '.' broadcast operator in CUDA.jl automatically triggers a GPU kernel
    c_device = a_device + b_device

    # Bring the result back to the CPU to verify
    c_host = Array(c_device)
    
    if all(c_host .== 2.0)
        println("Success: Calculation performed on the GPU and verified!")
    end
else
    println("Error: CUDA is not functional. Check your drivers or toolkit installation.")
end