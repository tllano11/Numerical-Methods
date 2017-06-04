#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""@package GaussianElimination
Solve a system of linear algebraic equations by using
the Gaussian Elimination method
"""

"""
    File name: gaussian_elimination.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 13-April-2017
    Date last modified: 04-June-2017
    Python Version: 3.6.0
"""

from numba import cuda
from numba import float64
import substitution
import numpy as np
import csv, sys

# Threads Per Block
tpb = 32


class GaussianElimination:
    @cuda.jit('void(float64[:], int32, int32)', target='gpu', nopython=True)
    def gaussian_elimination(Ab, size, i):
        """ Performs Gaussian elimination for each row of a column.

        @param A      Augmented matrix representing a SLAE.
        @param size   Size of coefficiente matrix.
        @param i      Integer representing the current column in which all threads
        are performing row operations.

        @return None
        """
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        idx = cuda.blockIdx.x * cuda.blockDim.x + tx
        idy = cuda.blockIdx.y * cuda.blockDim.y + ty
        size += 1

        # Thread does nothing when idx or idy are out of the matrix boundaries.
        if idx < size and idy < size:
            # Indicates which row must be computed by the current thread.
            index_r = idx * size
            # Copy Ab current position's value into shared memory.
            sAb = cuda.shared.array(shape=(tpb, tpb), dtype=float64)
            sAb[tx, ty] = Ab[index_r + idy]
            cuda.syncthreads()
            # Operates on rows below the diagonal.
            if idx > i:
                mul = sAb[tx, i] / sAb[i, i]
                # Computes elements to the right of column i.
                if idy >= i:
                    sAb[tx, ty] -= sAb[i, ty] * mul
            cuda.syncthreads()
            Ab[index_r + idy] = sAb[tx, ty]
            cuda.syncthreads()

    def start(self, A_matrix, b_matrix):
        """Launches parallel Gaussian elimination for a SLAE and returns its answer.

        @param A_matrix   Coefficient matrix of a SLAE.
        @param b_matrix   Linearly independent vector of a SLAE.

        @return None
        """
        if 0 in A_matrix.diagonal():
            return None

        b = b_matrix.reshape(len(b_matrix), 1)
        A = np.hstack((A_matrix, b))
        A = A.flatten()

        n = len(b)

        with cuda.pinned(A):
            stream = cuda.stream()
            gpu_A = cuda.to_device(A, stream=stream)
            bpg = 1

            for i in range(0, n):
                self.gaussian_elimination[(bpg, bpg), (tpb, tpb)](gpu_A, n, i)

        gpu_A.copy_to_host(A, stream)

        # Restore A and b from augmented matrix Ab
        b = A.reshape(n, (n + 1))[:, n]
        A = A.reshape(n, (n + 1))[..., :-1]

        x = substitution.back_substitution(A, b)

        if True in np.isnan(x) or True in np.isinf(x):
            return None
        else:
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

    gauss = GaussianElimination()
    print(gauss.start(A_matrix, b_matrix))


if __name__ == "__main__":
    main(sys.argv)
