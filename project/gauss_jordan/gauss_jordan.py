#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""@package Gauss Jordan
Solve a system of linear algebraic equations by using
the Gauss Jordan Elimination method
"""

"""
    File name: gauss_jordan.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 13-April-2017
    Date last modified: 04-June-2017
    Python Version: 3.6.0
"""

from numba import cuda
import numpy as np
import time, csv, sys, copy

tpb = 32


class GaussJordan:
    @cuda.jit
    def gauss_jordan(A, size, i):
        """Performs Gauss Jordan elimination for each row of a column.

        @param A        Augmented matrix representing a SLAE.
        @param size     Size of coefficiente matrix.
        @param i        Integer representing the current column in which all threads
        are performing row operations.

        @return None
        """

        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        size += 1
        # Indicates which row must be computed by the current thread.
        index_r = idx * size
        # Indicates the pivot row
        index_p = i * size
        # Thread does nothing when idx or idy are out of the matrix boundaries.
        if idx < size and idy < size:
            # Operates on rows below the diagonal.
            if idx > i:
                mul = A[index_r + i] / A[index_p + i]
                if idy >= i:
                    A[index_r + idy] -= A[index_p + idy] * mul
            # Operates on rows above the diagonal.
            elif idx < i:
                mul = A[index_r + i] / A[index_p + i]
                if idy >= i:
                    A[index_r + idy] -= A[index_p + idy] * mul
                    cuda.syncthreads()

    @cuda.jit
    def normalize(A, size):
        """Ensures every diagonal element of the augmented matrix A is
        set to one.

        @param A        Augmented matrix representing a SLAE.
        @param size     Size of coefficiente matrix.

        @return None
        """
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        size += 1

        if idx < size and idy < size:
            index = idx * size
            pivot = A[index + idx]
            if pivot != 0:
                A[index + idy] /= pivot
                cuda.syncthreads()

    def start(self, A_matrix, b_vector):
        """Launches parallel Gauss Jordan elimination for a SLAE and returns
        its answer.

        @param A_matrix     Coefficient matrix of a SLAE.
        @param b_vector     Linearly independent vector of a SLAE.

        @return float64[:]
        """
        if 0 in A_matrix.diagonal():
            return None

        b = b_vector.reshape(len(b_vector), 1)
        A = np.hstack((A_matrix, b))
        A = A.flatten()

        n = len(b)

        with cuda.pinned(A):
            stream = cuda.stream()
            gpu_A = cuda.to_device(A, stream=stream)
            bpg = 1

            for i in range(0, n):
                self.gauss_jordan[(bpg, bpg), (tpb, tpb)](gpu_A, n, i)
                self.normalize[(bpg, bpg), (tpb, tpb)](gpu_A, n)

        gpu_A.copy_to_host(A, stream)

        x = A.reshape(n, (n + 1))[:, n]
        return x


def main(argv):
    """For Test purposes"""
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

    gauss = GaussJordan()
    gauss.start(A_matrix, b_matrix)


if __name__ == "__main__":
    main(sys.argv)
