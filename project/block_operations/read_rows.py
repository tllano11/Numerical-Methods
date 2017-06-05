#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@package BlockOperations
Provides tools to solve large systems of linear algebraic equations 
by loading submatrices from a coefficient matrices.
"""

"""
File name: read_rows.py
Authors: Tomás Felipe Llano Ríos,
         Juan Diego Ocampo García,
         Johan Sebastián Yepes Ríos
Date created: 28-May-2017
Date last modified: 04-June-2017
Python Version: 3.6.0
"""

from numba import cuda
import numpy as np
import time, csv, sys
from jacobi_parallel_chunks import JacobiParallel


def get_error(x_vector, xant_vector):
    """Returns the norm of two given vectors,
    which represents the error of the current method.

    @param x_vector     The vector of the current stage of the method.
    @param xant_vector  The vector of the previous stage of the method.

    @return float128
    """
    maximum = 0
    size = len(x_vector)
    for i in range(0, size):
        tmp = float(abs(x_vector[i] - xant_vector[i]))
        if tmp > maximum:
            maximum = tmp
    return maximum


def start(a_name, b_name, rows_to_read, n, niter, tol):
    """Launches Jacobi for each set of rows read and returns a
    solution to the system.

    Keyword arguments:
    @param a_name String indicating A's path in the filesystem.
    @param b_name String indicating b's path in the filesystem.
    @param rows_to_read Integer indicating block size (rows to read from A).
    @param matrix_size Integer indicating N for a NxN matrix A.
    @param niter Maximum number of iterations to reach before stopping
    jacobi's execution.
    @return float64[:], float64, int32
    """
    matrix_size = n*n
    b_file = open(b_name).read().split('\n')[:-1]
    A_file = open(a_name).read().split('\n')[:-1]
    jp = JacobiParallel()
    error = tol + 1

    x_current = np.zeros(matrix_size, dtype=np.float64)
    count = 0
    while error > tol and count < niter:
        x_next = []
        A = []
        b = []
        count2 = 0
        for line in A_file:
            values = line.split()
            A_line = np.array(values).astype("float64")
            b_value = np.float64(b_file[count2])

            if (count2 % rows_to_read) != 0 or count2 == 0:
                A.append(A_line)
                b.append(b_value)
            else:
                first_row_block = count2 - rows_to_read
                x = jp.start(np.array(A), np.array(b), x_current, \
                             first_row_block)
                if x is None:
                    return None, None, None

                x_next = np.append(x_next, x)
                A.clear()
                A.append(A_line)
                b.clear()
                b.append(b_value)
            count2 += 1
        first_row_block = count2 - len(A)
        x = jp.start(np.array(A), np.array(b), x_current, first_row_block)
        if x is None:
            return None, None, None

        x_next = np.append(x_next, x)
        error = get_error(x_next, x_current)
        x_current = np.array(x_next).flatten()
        count += 1

    if error > tol:
        return None, error, count
    else:
        return x_current, error, count
