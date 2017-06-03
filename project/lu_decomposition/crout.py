#!/usr/bin/env python3.6
#-*- coding: utf-8 -*-

'''
    File name: crout.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 20-May-2017
    Date last modified: 20-May-2017
    Python Version: 3.6.0
'''
from numba import cuda
import numpy as np
import sys
import csv

@cuda.jit
def crout(A, L, U, n):
  idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
  idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

  if idx < n and idy < n:
    if idx == idy:
      accum = 0
      for p in range(idx):
        accum += L[idx*n + p] * U[p*n + idx]
        cuda.syncthreads()
      L[idx*n+idx] = A[idx*n+idx] - accum
      U[idx*n+idx] = 1
    elif idx > idy:
      accum = 0
      for p in range(idy):
        accum += L[idx*n + p] * U[p*n + idy]
        cuda.syncthreads()
      L[idx*n + idy] = A[idx*n + idy] - accum
      U[idy*n + idx] = (A[idy*n + idx] - accum)/L[idy*n + idy]


def start(A_matrix):
  A = A_matrix.flatten()
  L = np.zeros_like(A)
  U = np.zeros_like(A)

  rows = len(A_matrix)
  columns = len(A_matrix)
  tpb = 32
  n = rows * columns

  with cuda.pinned(A, L, U):
    stream = cuda.stream()
    gpu_A = cuda.to_device(A, stream=stream)
    gpu_L = cuda.to_device(L, stream=stream)
    gpu_U = cuda.to_device(U, stream=stream)
    bpg = n + (tpb - 1) // tpb

    crout[(bpg, bpg), (tpb, tpb)](gpu_A, gpu_L, gpu_U, rows)

  gpu_L.copy_to_host(L, stream)
  gpu_U.copy_to_host(U, stream)

  L = L.reshape(rows, columns)
  U = U.reshape(rows, columns)

  print(L)
  print(U)
  print(np.matmul(L,U))

if __name__ == "__main__":
  with open(sys.argv[1]) as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    matrix = list(reader)
    A_matrix = np.array(matrix).astype("float64")
    print(A_matrix)

  start(A_matrix)
