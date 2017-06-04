#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@package GaussJordan
Solve a system of linear algebraic equations by using
the Gauss Jordan Elimination method
"""

"""
    File name: serial_gauss_jordan.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 04-June-2017
    Python Version: 3.6.0
"""

import sys
import csv
import numpy as np


class GaussJordanSerial:
    def elimination(self, A, b):
        """Takes a system of linear equations represented by a matrix and a vector
        and returns the answer applying Gauss-Jordan method

        @return A   The coefficient matrix of the system.
        @return b   The linearly independent vector.

        @return float128[:]
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


def main(argv):
    """For Test purposes"""
    A_name = argv[1]
    b_name = argv[2]

    with open(A_name) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        matrix = list(reader)
        A = np.array(matrix).astype("float64")

    with open(b_name) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        matrix = list(reader)
        b = np.array(matrix).astype("float64")

    gauss = GaussJordanSerial()
    x = gauss.elimination(A, b)
    print(x)


if __name__ == '__main__':
    main(sys.argv)
