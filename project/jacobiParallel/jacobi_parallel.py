#!/usr/bin/env python
#-*- coding: utf-8 -*-

from numba import cuda
import numpy as np
import time, csv, sys

class JacobiParallel():

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

  def start(A, b, niter):
    tpb = 32
    bpg = len(A) + (tpb - 1) // tpb
    length = len(b)
    x_current = np.zeros(length, dtype=np.float64)
    x_next    = np.zeros(length, dtype=np.float64)
    gpu_A = cuda.to_device(A)
    gpu_b = cuda.to_device(b)
    gpu_x_current = cuda.to_device(x_current)
    gpu_x_next = cuda.to_device(x_next)
    for i in range(niter):
      if i % 2:
        jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_current, gpu_x_next, length, length)
      else:
        jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_next, gpu_x_current, length, length)

    x_next = gpu_x_next.copy_to_host()

    return x_next

def main(argv):
  if len(argv) != 3:
    print("Usage: python3.6 jacobi.py <A_matrix> <b_matrix>")
    sys.exit()

  A_name = argv[1]
  b_name = argv[2]

  with open(A_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    matrix = list(reader)
    A_matrix = np.array(matrix).astype("float64")
    A = A_matrix.flatten()

  with open(b_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    matrix = list(reader)
    b_matrix = np.array(matrix).astype("float64")
    b = b_matrix.flatten()

  rows      = len(b)
  columns   = len(b)
  tpb       = 32
  matrix_size = rows * columns

  x_current = np.zeros(columns, dtype=np.float64)
  x_next    = np.zeros(columns, dtype=np.float64)
  gpu_A = cuda.to_device(A)
  gpu_b = cuda.to_device(b)
  gpu_x_current = cuda.to_device(x_current)
  gpu_x_next = cuda.to_device(x_next)

  bpg = matrix_size + (tpb - 1) // tpb

  start = time.time()
  for i in range(0, 100):
    if i % 2:
      jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_current, gpu_x_next, rows, columns)
    else:
      jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_next, gpu_x_current, rows, columns)

  end = time.time()

  x_next = gpu_x_next.copy_to_host()

  print ("Jacobis's algorithm computation time was: {} sec".format(end - start))
  print (x_next)

if __name__ == "__main__":
  main(sys.argv)
