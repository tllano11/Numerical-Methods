#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

"""@package GaussianElimination
Solve a system of linear algebraic equations by using
the Gaussian Elimination method
"""

"""
    File name: substitution.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 03-May-2017
    Date last modified: 29-May-2017
    Python Version: 3.6.0
"""


def forward_substitution(L, b):
    """Returns the solution for a SLAE represented
    by a lower triangular coefficient matrix.

    @param A    Lower triangular coefficient matrix.
    @param b    Linearly independent vector.
    @param n    Size of matrix A.

    @return     float128[:]
    """
    n = len(L[0])
    z = [0] * n
    for i in range(0, n):
        if L[i][i] != 0:
            accum = 0
            for j in range(0, i):
                accum += L[i][j] * z[j]
            z[i] = (b[i] - accum) / L[i][i]
    return z


def back_substitution(U, z):
    """Returns the solution for a SLAE represented
    by an upper triangular coefficient matrix.

    @param A    Upper triangular coefficient matrix.
    @param b    Linearly independent vector.
    @param n    Size of matrix A.

    @return     float128[:]
    """
    n = len(U[0])
    x = [0] * n
    for i in range(n - 1, -1, -1):
        if U[i][i] != 0:
            accum = 0
            for j in range(i, n):
                accum += U[i][j] * x[j]
            x[i] = (z[i] - accum) / U[i][i]
    return x
