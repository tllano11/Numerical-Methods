#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  File name: serial_decomposition_LU.py
  Authors: Tomás Felipe Llano Ríos,
           Juan Diego Ocampo García,
           Johan Sebastián Yepes Ríos
  Date last modified: 29-May-2017
  Python Version: 3.6.0
"""

import sys
import numpy as np
import substitution


class SerialLUDecomposition():
    def decomposition_LU(self, A):
        """Splits a given matrix into two matrices (lower and upper triangular matrices).
    It is based on multiplication of matrices.

    keyword arguments:
    A -- The coefficient matrix to be splited.
    """
        n = len(A)
        L = np.zeros(shape=(n, n))
        P = np.identity(n)
        for k in range(0, n - 1):
            L[k][k] = 1
            # A, P = self.partial_pivot(A, P, k)
            for i in range(k + 1, n):
                multiplier = A[i][k] / A[k][k]
                L[i][k] = multiplier
                for j in range(k, n):
                    A[i][j] = A[i][j] - multiplier * A[k][j]
        L[n - 1][n - 1] = 1
        return L, A

    def solve_system(self, L, U, b):
        """Solves a LU system.

    keyword arguments:
    L -- The lower triangular matrix of the system.
    U -- The upper triangular matrix of the system.
    b -- Linearly independent vector.
    """
        size = len(b)
        z = substitution.forward_substitution(L, b)
        x = substitution.back_substitution(U, z)
        print(x)
        return x


if __name__ == '__main__':
    A = np.array([[-7, 2, -3, 4], [5, -1, 14, -1], [1, 9, -7, 5], [-12, 13, -8, -4]], dtype="float")
    b = np.array([-12, 13, 31, -32], dtype="float")
    LUdecomposition = SerialLUDecomposition()
    L, U = LUdecomposition.decomposition_LU(A)
    x = LUdecomposition.solve_system(L, U, b)
