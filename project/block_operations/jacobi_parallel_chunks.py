#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
@package BlockOperations
Provides tools to solve large systems of linear algebraic equations 
by loading submatrices from a coefficient matrices.
"""

"""
    File name: block_operations_tab.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 04-June-2017
    Python Version: 3.6.0
"""
from numba import cuda
import numpy as np
import time, csv, sys


class JacobiParallel:
    @cuda.jit('void(float64[:], float64[:], float64[:], float64[:],' \
              'int32, int32, int32, float32)', target='gpu', nopython=True)
    def jacobi(A, b, x_current, x_next, rows, cols, first_row_block, rel):
        """Performs jacobi for every thread in matrix A boundaries.

        @param A Matrix extracted from the coefficient matrix A.
        @param b Vector extracted from Linearly independent vector b.
        @param x_current Current answer's approximation.
        @param x_next vector in which to store new answer.
        @param rows Number of rows read (i.e. number of rows in the block).
        @param cols Number of columns from the original matrix.
        @param first_row_block Integer indicating the first row of the block by
        using an index from the coefficient matrix A (i.e. What is the
        correspondence between the first block's row and A).
        @param rel Relaxation coefficient.
        @return None
        """
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

    @cuda.jit('void(float64[:], float64[:], float64[:], int32)', \
              target='gpu', nopython=True)

    def get_error(x_current, x_next, x_error, rows):
        """Calculates jacobi's maximum error.

        @param x_current Pointer to list representing current
                         approximation for vector x in a system Ax = b.
        @param x_next    Pointer to list representing new
                         approximation for vector x in a system Ax = b.
        @param x_error   Pointer to list in which an error for each
                         approximation will be stored.
        @param rows      Coefficient matrix A number of rows.

        @return None
        """
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx < rows:
            x_error[idx] = abs(x_next[idx] - x_current[idx])

    def start(self, A, b, x_current, first_row_block, rel=1):
        """Launches parallel jacobi solver for a SLAE and returns its answer.

        @param A         Coefficient matrix of a SLAE.
        @param b         Linearly independent vector of a SLAE.
        @param x_current Pointer to list representing current
                         approximation for vector x in a system Ax = b.
        @param first_row_block Absolute position of the block's row in the complete matrix.
        @param rel       Relaxation coefficient.

        @return float64[:]
        """
        rows = len(b)
        col = first_row_block
        for i in range(0, rows):
            if A[i][col] == 0:
                return None
            col += 1

        A = A.flatten()
        A_size = len(A)
        tpb = 32
        bpg = A_size + (tpb - 1) // tpb
        cols = A_size // rows
        x_next = np.zeros(rows, dtype=np.float64)
        gpu_A = cuda.to_device(A)
        gpu_b = cuda.to_device(b)
        gpu_x_current = cuda.to_device(x_current)
        gpu_x_next = cuda.to_device(x_next)

        self.jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_current, \
                              gpu_x_next, rows, cols, first_row_block, rel)

        x_next = gpu_x_next.copy_to_host()
        return x_next
