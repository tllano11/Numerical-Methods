#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@package LuDecomposition
Decompuses a matrix A into two matrices L and U.
"""

"""
  File name: serial_decomposition_LU.py
  Authors: Tomás Felipe Llano Ríos,
           Juan Diego Ocampo García,
           Johan Sebastián Yepes Ríos
  Date last modified: 04-June-2017
  Python Version: 3.6.0
"""

import numpy as np
import substitution


class SerialLUDecomposition():
    def decomposition_LU(self, A):
        """Splits a given matrix into two matrices (lower and upper triangular matrices).
        It is based on multiplication of matrices.

        @param A The coefficient matrix to be splited.
        @return float128[:,:], float128[:,:]
        """
        n = len(A)
        L = np.zeros(shape=(n, n))
        P = np.identity(n)
        for k in range(0, n - 1):
            L[k][k] = 1
            # A, P = self.partial_pivot(A, P, k)
            for i in range(k + 1, n):
                if A[k][k] == 0:
                    return None, None
                multiplier = A[i][k] / A[k][k]
                L[i][k] = multiplier
                for j in range(k, n):
                    A[i][j] = A[i][j] - multiplier * A[k][j]
        L[n - 1][n - 1] = 1
        return L, A

    def solve_system(self, L, U, b):
        """Solves a LU system.

        @param L The lower triangular matrix of the system.
        @param U The upper triangular matrix of the system.
        @param b Linearly independent vector.
        @return float128[:]
        """
        size = len(b)
        z = substitution.forward_substitution(L, b)
        x = substitution.back_substitution(U, z)
        return x
