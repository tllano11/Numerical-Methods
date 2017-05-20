from numba import cuda
import numpy as np
import sys
import csv

@cuda.jit
def gaussian_lu_decomposition(A, L, size, i):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    index = idx*size+idy

    if idx < size and idy < size:
        if idx > i:
            mul = A[idx*size+i]/A[i*size+i]
            if idy >= i:
                A[index] -= A[i*size+idy] * mul
                if idy == i:
                    L[index] = mul
        elif idx == idy:
            L[index] = 1
        cuda.syncthreads()


def start(A_matrix):
    A = A_matrix.flatten()
    L = np.zeros_like(A)

    rows = len(A_matrix)
    columns = len(A_matrix)
    tpb = 32
    matrix_size = rows * columns

    with cuda.pinned(A, L):
      stream = cuda.stream()
      gpu_A = cuda.to_device(A, stream=stream)
      gpu_L = cuda.to_device(L, stream=stream)
      bpg = matrix_size + (tpb - 1) // tpb

      for i in range(0, rows):
        gaussian_lu_decomposition[(bpg, bpg), (tpb, tpb)](gpu_A, gpu_L, rows, i)

    gpu_A.copy_to_host(A, stream)
    gpu_L.copy_to_host(L, stream)

    U = A.reshape(rows, columns)
    L = L.reshape(rows, columns)
    print(L)
    print(U)
    print(np.matmul(L,U))
    del stream
    # return L,U

if __name__ == "__main__":
    with open(sys.argv[1]) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        matrix = list(reader)
        A_matrix = np.array(matrix).astype("float64")

    start(A_matrix)
