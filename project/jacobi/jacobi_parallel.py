#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numba import cuda
import numpy as np
import time, csv, sys

class JacobiParallel():

  @cuda.jit
  def jacobi(A, b, x_current, x_next, n, rel=1):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < n:
      sigma = 0.0
      index = idx * n
      for j in range(0, n):
        if idx != j:
          sigma += A[index + j] * x_current[j]

      x_next[idx] = (b[idx] - sigma) / A[index + idx]
      x_next[idx] = rel * x_next[idx] + (1-rel) * x_current[idx]

  @cuda.jit
  def get_error(x_current, x_next, x_error, rows):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < rows:
      x_error[idx] = abs(x_next[idx] - x_current[idx])

  def start(self, A, b, niter, tol, rel):
    A = A.flatten()
    b = b.flatten()
    tpb = 32
    bpg = len(A) + (tpb - 1) // tpb
    length = len(b)
    x_current = np.zeros(length, dtype=np.float64)
    x_next    = np.zeros(length, dtype=np.float64)
    x_error   = np.zeros(length, dtype=np.float64)
    gpu_A = cuda.to_device(A)
    gpu_b = cuda.to_device(b)
    gpu_x_current = cuda.to_device(x_current)
    gpu_x_next = cuda.to_device(x_next)
    gpu_x_error = cuda.to_device(x_error)
    count = 0
    error = tol + 1

    start = time.time()
    while error > tol and count < niter:
      if count % 2:
        self.jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_current, gpu_x_next, length, rel)
        self.get_error[bpg, tpb](gpu_x_current, gpu_x_next, gpu_x_error, length)
      else:
        self.jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_next, gpu_x_current, length, rel)
        self.get_error[bpg, tpb](gpu_x_next, gpu_x_current, gpu_x_error, length)

      x_error = gpu_x_error.copy_to_host()
      error = max(x_error)
      count += 1
    end = time.time()

    if error < tol:
      if count % 2:
        x_next = gpu_x_next.copy_to_host()
      else:
        x_next = gpu_x_current.copy_to_host()

      print("Jacobis's algorithm computation time was: {} sec".format(end - start))
      print("Jacobi done with an error of {} and iter {}".format(error, count))
      print(x_next)
      return x_next
    else:
      print("Jacobi failed")
      return None

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

  n = len(b)
  tpb = 32
  matrix_size = n * n

  x_current = np.zeros(n, dtype=np.float64)
  x_next    = np.zeros(n, dtype=np.float64)
  gpu_A = cuda.to_device(A)
  gpu_b = cuda.to_device(b)
  gpu_x_current = cuda.to_device(x_current)
  gpu_x_next = cuda.to_device(x_next)

  bpg = matrix_size + (tpb - 1) // tpb
  jacobiParallel = JacobiParallel()

  start = time.time()
  for i in range(0, 100):
    if i % 2:
      jacobiParallel.jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_current, gpu_x_next, n)
    else:
      jacobiParallel.jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_next, gpu_x_current, n)

  end = time.time()

  x_next = gpu_x_next.copy_to_host()

  print ("Jacobis's algorithm computation time was: {} sec".format(end - start))
  print (x_next)

if __name__ == "__main__":
  main(sys.argv)
