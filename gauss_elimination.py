from numba import cuda
import substitution
import numpy as np
import time, csv, sys

"""
def gauss_jordan3(A, b, size):
  idx = cuda.threadIdx.x
  idy = cuda.threadIdx.y
  if idx < size and idy < size:
    for i in range(1, size):
      if (idy+1) < size:
        var = (-1) * (A[(i-1)*size+(i-1)]/A[(i+idy)*size+(i-1)])
        A[(i+idy)*size+idx] = A[(i-1)*size+idx] + ((var) * A[(i+idy)*size+idx])
      cuda.syncthreads()
    b[idy * (size+1) + idx] = A[idy*size + idx]
"""

@cuda.jit
def gauss_jordan(A, size, i):
  idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
  idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
  size += 1

  if idx < size and idy < size:
    if idx > i:
      pivot = A[idx*size+i]/A[i*size+i]
      if idy >= i:
        A[idx*size+idy]-= A[i*size+idy] * pivot
    cuda.syncthreads()

@cuda.jit
def normalize(A, size):
  idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
  idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
  size += 1

  if idx < size and idy < size:
    index = idx * size
    pivot = A[index + idx]
    if pivot != 0:
      A[index + idy] /= pivot
  cuda.syncthreads()

"""
@cuda.jit
def gauss_jordan1(A, size):
  idx = cuda.threadIdx.x
  idy = cuda.threadIdx.y
  size += 1
  cuda.syncthreads()
  if idx < size and idy < size:
    for j in range(0, size):
      pivot = A[j*size + j]
      A[idx*size + j] /= pivot
      cuda.syncthreads()
      if idy != j:
        A[idx*size+idy] -= A[j*size+idy] * A[idx*size+j]
      cuda.syncthreads()
"""

def main(argv):
  if len(argv) != 3:
    print("Usage: python3.6 gauss_jordan_elimination.py <A_matrix> <b_matrix>")
    sys.exit()

  A_name = argv[1]
  b_name = argv[2]

  with open(A_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    matrix = list(reader)
    A_matrix = np.array(matrix).astype("float64")
    #A = A_matrix.flatten()
  with open(b_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    matrix = list(reader)
    b_matrix = np.array(matrix).astype("float64")
    #b = b_matrix.flatten()

  b = b_matrix.reshape(len(b_matrix), 1)
  A = np.hstack((A_matrix, b))
  A = A.flatten()

  rows      = len(b)
  columns   = len(b)
  tpb       = 32
  matrix_size = rows * columns

  gpu_A = cuda.to_device(A)
  #gpu_b = cuda.to_device(b)
  bpg = matrix_size + (tpb - 1) // tpb

  for i in range(0, rows):
    gauss_jordan[(bpg, bpg), (tpb, tpb)](gpu_A, rows, i)
    normalize[(bpg, bpg), (tpb, tpb)](gpu_A, rows)

  A = gpu_A.copy_to_host()
  b = A.reshape(rows, (columns+1))[:, columns]
  A = A.reshape(rows, (columns+1))[..., :-1]

  x = substitution.back_substitution(A, b, rows)

  print(A)
  print(b)
  print(x)

  """
  b = gpu_b.copy_to_host()
  A = gpu_A.copy_to_host()
  print(b)
  print(A)
  """

if __name__ == "__main__":
  main(sys.argv)
