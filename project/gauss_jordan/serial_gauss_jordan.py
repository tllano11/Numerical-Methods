#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File name: serial_gauss_jordan.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 29-May-2017
    Python Version: 3.6.0
"""

import sys
import numpy as np


class GaussJordanSerial:
    def elimination(self, A, b):
        """Takes a system of linear equations represented by a matrix and a vector
        and returns the answer applying Gauss-Jordan method

        keyword arguments:
        A -- The coefficient matrix of the system.
        b -- The linearly independent vector.
        """
        n = len(A)
        for k in range(0, n):
            for i in range(0, n):
                if i != k:
                    if A[k][k] == 0:
                        return None
                    multiplier = A[i][k] / A[k][k]
                    for j in range(k, n):
                        A[i][j] = A[i][j] - multiplier * A[k][j]
                    b[i] = b[i] - multiplier * b[k]
        for i in range(0, n):
            b[i] = b[i] / A[i][i]
            A[i][i] = A[i][i] / A[i][i]
        return b


if __name__ == '__main__':
    A = np.array([[14, 6, -2, 3], [3, 15, 2, -5], [-7, 4, -23, 2], [1, -3, -2, 16]], dtype="float")
    b = np.array([12, 32, -24, 14], dtype="float")
    gauss = GaussJordanSerial()
    x = gauss.elimination(A, b)
    print(x)
