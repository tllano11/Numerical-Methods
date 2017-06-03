#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
    File name: block_operations_tab.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 03-June-2017
    Python Version: 3.6.0
"""
from numba import cuda
import numpy as np
import time, csv, sys


class JacobiParallel:

    @cuda.jit('void(float64[:], float64[:], float64[:], float64[:],'\
              'int32, int32, int32, float32)', target='gpu', nopython=True)
    def jacobi(A, b, x_current, x_next, rows, cols, first_row_block, rel):
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        current_row = first_row_block + idx
        if idx < rows:
            sigma = 0.0
            index = idx * cols
            for j in range(0, cols):
                if current_row != j:
                    sigma += A[index + j] * x_current[j]

            x_next[idx] = (b[idx] - sigma) / A[index + current_row]
            x_next[idx] = rel * x_next[idx] + (1 - rel) * x_current[idx]


    @cuda.jit('void(float64[:], float64[:], float64[:], int32)',\
              target='gpu', nopython=True)
    def get_error(x_current, x_next, x_error, rows):
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx < rows:
            x_error[idx] = abs(x_next[idx] - x_current[idx])


    def start(self, A, b, x_current, first_row_block, rel=1):
        rows = len(b)
        col = first_row_block
        for i in range(0, rows):
            if A[i][col] == 0:
                return None
            col += 1

        A = A.flatten()
        tpb = 32
        bpg = len(A) + (tpb - 1) // tpb
        cols = len(A) // rows
        x_next = np.zeros(rows, dtype=np.float64)
        gpu_A = cuda.to_device(A)
        gpu_b = cuda.to_device(b)
        gpu_x_current = cuda.to_device(x_current)
        gpu_x_next = cuda.to_device(x_next)
        count = 0

        self.jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_current, \
                              gpu_x_next, rows, cols, first_row_block, rel)

        x_next = gpu_x_next.copy_to_host()
        return x_next
