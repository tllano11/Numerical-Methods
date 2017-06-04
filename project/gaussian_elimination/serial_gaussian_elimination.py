#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@package Gaussian Elimination
Solve a system of linear algebraic equations by using
the Gaussian Elimination method
"""

"""
    File name: serial_gauss_elimination.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 04-June-2017
    Python Version: 3.6.0
"""

import sys
import numpy as np
import substitution
from time import time


class SerialGaussianElimination:
    def elimination(self, A, b):
        """Takes a system of linear equations represented by a matrix and a vector
        and returns the answer applying Gaussian elimination method.

        @param A    The coefficient matrix of the system.
        @param b    The linearly independent vector.

        @return     float128[:]
        """
        b = b.flatten()
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
        return x

    def partial_pivot(self, A, b, k):
        """Applies the partial pivot strategy to a system of linear equations.

        @param A    The coefficient matrix of the system.
        @param b    The linearly independent vector.
        @param k    The current elimination stage.

        @return     float128[:,:], float128[:]
        """
        maximum = abs(A[k][k])
        max_row = k
        n = len(A)
        for s in range(k + 1, n):
            if abs(A[s][k]) > maximum:
                maximum = abs(A[s][k])
                max_row = s
        if maximum != 0:
            if max_row != k:
                aux_A = np.copy(A[k])
                A[k] = np.copy(A[max_row])
                A[max_row] = np.copy(aux_A)
                aux_B = np.copy(b[k])
                b[k] = np.copy(b[max_row])
                b[max_row] = np.copy(aux_B)
        return A, b
