#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""
    File name: gaussian_lu_decomposition.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 20-May-2017
    Date last modified: 20-May-2017
    Python Version: 3.6.0
"""

from numba import cuda
import numpy as np
import sys
import csv


class GuassianLUDecomposition:
    @cuda.jit
    def gaussian_lu_decomposition(A, L, size, i):
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
            bpg = matrix_size + (tpb - 1) // tpb

            for i in range(0, rows):
                self.gaussian_lu_decomposition[(bpg, bpg), (tpb, tpb)](gpu_A, gpu_L, rows, i)

        gpu_A.copy_to_host(A, stream)
        gpu_L.copy_to_host(L, stream)

        U = A.reshape(rows, columns)
        L = L.reshape(rows, columns)
        del stream
        return L, U

    def get_solution(self, L, U, b):
        z = self.forward_substitution(L, b)
        x = self.back_substitution(U, z)
        return x

    def forward_substitution(self, L, b):
        n = len(L[0])
        z = [0] * n
        for i in range(0, n):
            if L[i][i] != 0:
                accum = 0
                for j in range(0, i):
                    accum += L[i][j] * z[j]
                z[i] = (b[i] - accum) / L[i][i]
        return z

    def back_substitution(self, U, z):
        n = len(U[0])
        x = [0] * n
        for i in range(n - 1, -1, -1):
            if U[i][i] != 0:
                accum = 0
                for j in range(i, n):
                    accum += U[i][j] * x[j]
                x[i] = (z[i] - accum) / U[i][i]
        return x

    def get_inverse(self, L, U):
        pass

    def get_determinant(self, L, U):
        pass


if __name__ == "__main__":
    with open(sys.argv[1]) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        matrix = list(reader)
        A_matrix = np.array(matrix).astype("float64")

    decomposition = GuassianLUDecomposition()
    decomposition.start(A_matrix)
