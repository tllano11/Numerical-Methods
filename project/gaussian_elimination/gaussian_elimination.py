#!/usr/bin/env python3.6
#-*- coding: utf-8 -*-

'''
    File name: gauss_elimination.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 13-April-2017
    Date last modified: 03-May-2017
    Python Version: 3.6.0
'''

from numba import cuda
import substitution
import numpy as np
import time, csv, sys

class GaussianElimination():

  @cuda.jit
  def gaussian_elimination(A, size, i):
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

  def start(self, A_matrix, b_matrix):
    b = b_matrix.reshape(len(b_matrix), 1)
    A = np.hstack((A_matrix, b))
    A = A.flatten()

    rows = len(b)
    columns = len(b)
    tpb = 32
    matrix_size = rows * columns

    with cuda.pinned(A):
      stream = cuda.stream()
      gpu_A = cuda.to_device(A, stream=stream)
      bpg = matrix_size + (tpb - 1) // tpb

      for i in range(0, rows):
        self.gaussian_elimination[(bpg, bpg), (tpb, tpb)](gpu_A, rows, i)
        self.normalize[(bpg, bpg), (tpb, tpb)](gpu_A, rows)

    gpu_A.copy_to_host(A, stream)
    b = A.reshape(rows, (columns+1))[:, columns]
    A = A.reshape(rows, (columns+1))[..., :-1]

    x = substitution.back_substitution(A, b, rows)
    print(x)
    return x

def main(argv):
  if len(argv) != 3:
    print("Usage: python3.6 gauss_jordan.py <A_matrix> <b_matrix>")
    sys.exit()

  A_name = argv[1]
  b_name = argv[2]

  with open(A_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    matrix = list(reader)
    A_matrix = np.array(matrix).astype("float64")

  with open(b_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    matrix = list(reader)
    b_matrix = np.array(matrix).astype("float64")

  b = b_matrix.reshape(len(b_matrix), 1)
  A = np.hstack((A_matrix, b))
  A = A.flatten()

  rows      = len(b)
  columns   = len(b)
  tpb       = 32
  matrix_size = rows * columns

  with cuda.pinned(A):
    stream = cuda.stream()
    gpu_A = cuda.to_device(A, stream=stream)
    bpg = matrix_size + (tpb - 1) // tpb

    for i in range(0, rows):
      gauss_jordan[(bpg, bpg), (tpb, tpb)](gpu_A, rows, i)
      normalize[(bpg, bpg), (tpb, tpb)](gpu_A, rows)

  gpu_A.copy_to_host(A, stream)
  b = A.reshape(rows, (columns+1))[:, columns]
  A = A.reshape(rows, (columns+1))[..., :-1]

  x = substitution.back_substitution(A, b, rows)



if __name__ == "__main__":
  main(sys.argv)
