using Krylov, KrylovPreconditioners

# A_gpu can be a sparse COO, CSC or CSR matrix
opA_gpu = KrylovOperator(A_gpu)
x_gpu, stats = gmres(opA_gpu, b_gpu)