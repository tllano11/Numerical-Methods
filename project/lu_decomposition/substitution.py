#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

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

      Keyword arguments:
      A -- Lower triangular coefficient matrix.
      b -- Linearly independent vector.
      n -- Size of matrix A.
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

      Keyword arguments:
      A -- Upper triangular coefficient matrix.
      b -- Linearly independent vector.
      n -- Size of matrix A.
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


def main():
    A = [[3, 0, 0, 0], [-1, 1, 0, 0], [3, -2, -1, 0], [1, -2, 6, 2]]
    b = [5, 6, 4, 2]
    print("forward subs: ", forward_substitution(A, b))

    A = [[4, -1, 2, 3], [0, -2, 7, -4], [0, 0, 6, 5], [0, 0, 0, 3]]
    b = [20, -7, 4, 6]
    print("back subs: ", back_substitution(A, b))


if __name__ == "__main__":
    main()
