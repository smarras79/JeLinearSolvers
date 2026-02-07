using CUDA, Krylov
using CUDA.CUSPARSE, SparseArrays

if CUDA.functional()
  # CPU Arrays
  A_cpu = sprand(200, 100, 0.3)
  b_cpu = rand(200)

  # GPU Arrays
  A_csc_gpu = CuSparseMatrixCSC(A_cpu)
  A_csr_gpu = CuSparseMatrixCSR(A_cpu)
  A_coo_gpu = CuSparseMatrixCOO(A_cpu)
  b_gpu = CuVector(b_cpu)

  # Solve a rectangular and sparse system on an Nvidia GPU
  x_csc, stats_csc = lslq(A_csc_gpu, b_gpu)
  x_csr, stats_csr = lsqr(A_csr_gpu, b_gpu)
  x_coo, stats_coo = lsmr(A_coo_gpu, b_gpu)
end