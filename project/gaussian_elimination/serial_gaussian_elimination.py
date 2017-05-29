#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File name: serial_gauss_elimination.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 29-May-2017
    Python Version: 3.6.0
"""

import sys
import numpy as np
import substitution


class SerialGaussianElimination:
    def elimination(self, A, b):
        """Takes a system of linear equations represented by a matrix and a vector
    and returns the answer applying Gaussian elimination method.

    keyword arguments:
    A -- The coefficient matrix of the system.
    b -- The linearly independent vector.
    """
        n = len(A)
        for k in range(0, n - 1):
            A, b = self.partial_pivot(A, b, k)
            for i in range(k + 1, n):
                if A[k][k] == 0:
                    return None
                multiplier = A[i][k] / A[k][k]
                for j in range(k, n):
                    A[i][j] = A[i][j] - multiplier * A[k][j]
                b[i] = b[i] - multiplier * b[k]
        x = substitution.back_substitution(A, b)
        print(A)
        return x

    def partial_pivot(self, A, b, k):
        """Applies the partial pivot strategy to a system of linear equations.

    keyword arguments:
    A -- The coefficient matrix of the system.
    b -- The linearly independent vector.
    k -- The current elimination stage.
    """
        maximum = abs(A[k][k])
        max_row = k
        n = len(A)
        for s in range(k + 1, n):
            if abs(A[s][k]) > maximum:
                maximum = abs(A[s][k])
                max_row = s
        if (maximum != 0):
            if (max_row != k):
                aux_A = np.copy(A[k])
                A[k] = np.copy(A[max_row])
                A[max_row] = np.copy(aux_A)
                aux_B = b[k]
                b[k] = b[max_row]
                b[max_row] = aux_B
        return A, b


if __name__ == '__main__':
    A = np.array([[14, 6, -2, 3], [3, 15, 2, -5], [-7, 4, -23, 2], [1, -3, -2, 16]], dtype="float")
    b = np.array([12, 32, -24, 14], dtype="float")
    gauss = GaussianElimination()
    x = gauss.elimination(A, b)
