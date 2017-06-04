#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
@package LuDecomposition
Decompuses a matrix A into two matrices L and U.
"""

"""
    File name: gaussian_lu_decomposition.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 20-May-2017
    Date last modified: 04-June-2017
    Python Version: 3.6.0
"""

from numba import cuda
import numpy as np
import sys
import csv
import substitution


class GuassianLUDecomposition:
    @cuda.jit
    def gaussian_lu_decomposition(A, L, size, i):
        """ Performs Gaussian LU elimination.

        @param A Coefficient matrix A.
        @param L Matrix in which to store the multipliers.
        @param size Size of coefficiente matrix.
        @param i Integer representing the current column in which all threads
        are performing row operations.
        @return None
        """
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        index = idx * size + idy

        if idx < size and idy < size:
            if idx > i:
                mul = A[idx * size + i] / A[i * size + i]
                if idy >= i:
                    A[index] -= A[i * size + idy] * mul
                    if idy == i:
                        L[index] = mul
            elif idx == idy:
                L[index] = 1
            cuda.syncthreads()

    def start(self, A_matrix):
        """Decomposes A_matrix into two matrices L and U.

        @param A_matrix Coefficient matrix.
        @return float64[:,:], float64[:,:]
        """
        A = A_matrix.flatten()
        L = np.zeros_like(A)

        rows = len(A_matrix)
        columns = len(A_matrix)
        tpb = 32
        matrix_size = rows * columns

        with cuda.pinned(A, L):
            stream = cuda.stream()
            gpu_A = cuda.to_device(A, stream=stream)
            gpu_L = cuda.to_device(L, stream=stream)
            bpg = 1

            for i in range(0, rows):
                self.gaussian_lu_decomposition[(bpg, bpg), (tpb, tpb)](gpu_A,\
                                                                       gpu_L,\
                                                                       rows, i)

        gpu_A.copy_to_host(A, stream)
        gpu_L.copy_to_host(L, stream)

        U = A.reshape(rows, columns)
        L = L.reshape(rows, columns)
        del stream
        return L, U

    def get_solution(self, L, U, b):
        """Solves a LU system.

        @param L The lower triangular matrix of the system.
        @param U The upper triangular matrix of the system.
        @param b Linearly independent vector.
        @return float64[:]
        """
        z = substitution.forward_substitution(L, b)
        x = substitution.back_substitution(U, z)
        return x

    def gen_identity_matrix(self, size):
        """Creates an identity matrix given a size.

        @param size Number of rows and columns that the matrix will have.
        @return float64[:,:]
        """
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            matrix[i][i] = 1
        return matrix

    def get_inverse(self, L, U):
        """Returns the inverse of a given matrix by means of LU decomposition.

        @param L The lower triangular matrix of the system.
        @param U The upper triangular matrix of the system.
        @return float64[:,:]
        """
        deter = self.get_determinant(L, U)
        if deter == 0:
            return None

        size = len(U)
        identity_matrix = np.array(self.gen_identity_matrix(size))
        cont = 0
        for b in identity_matrix:
            z = substitution.forward_substitution(L, b);
            x = np.array(substitution.back_substitution(U, z))
            x_column = x.reshape(len(x), 1)
            if cont == 0:
              AI = x_column
            else:
              AI = np.insert(AI,cont, x, axis=1)
            cont += 1
        return(AI)

    def get_determinant(self, L, U):
        """Returns the determinant of a given matrix by means of
        LU decomposition.

        keyword arguments:
        @param L The lower triangular matrix of the system.
        @param U The upper triangular matrix of the system.
        @return float64
        """
        l_product = 1
        u_product = 1
        for i in range(0, len(L)):
          l_product *= L[i][i]
          u_product *= U[i][i]
        return l_product * u_product


if __name__ == "__main__":
    with open(sys.argv[1]) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        matrix = list(reader)
        A_matrix = np.array(matrix).astype("float64")

    decomposition = GuassianLUDecomposition()
    decomposition.start(A_matrix)
