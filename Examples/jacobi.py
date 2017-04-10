from numba import cuda
import numpy as np
import time

@cuda.jit
def jacobi(A, b, x_current, x_next, rows, columns):
  idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
  if idx < rows:
    sigma = 0.0
    index = idx * columns
    for j in range(0, columns):
      if idx != j:
        sigma += A[index + j] * x_current[j]

    x_next[idx] = (b[idx] - sigma) / A[index + idx]

def main():
  rows      = 3
  columns   = 3
  tpb       = 32
  matrix_size = rows * columns


  A 	    = np.array([[8,-1,7],[-2,3,1],[0,1,9]])
# A         = np.random.random((rows, columns))
  A         = A.flatten()
  b         = np.array([4,0,1])
# b         = np.random.random((rows, 1))
#  b         = b.flatten()
  x_current = np.zeros(columns, dtype=np.float64)
  x_next    = np.zeros(columns, dtype=np.float64)

  bpg = matrix_size + (tpb - 1) // tpb

  start = time.time()
  for i in range(0, 25):
    if i % 2:
      jacobi[bpg, tpb](A, b, x_current, x_next, rows, columns )
    else:
      jacobi[bpg, tpb](A, b, x_next, x_current, rows, columns )

  end = time.time()

  print (end - start)

  print (x_next)

if __name__ == "__main__":
  main()
