#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File name: jacobi_parallel.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date created: 13-April-2017
    Date last modified: 03-June-2017
    Python Version: 3.6.0
"""

from numba import cuda
import numpy as np
import time, csv, sys


class JacobiParallel:

    @cuda.jit('void(float64[:], float64[:], float64[:], float64[:],'\
              'int32, float32)', target='gpu', nopython=True)
    def jacobi(A, b, x_current, x_next, n, rel):
        """Performs jacobi for every thread in matrix A boundaries.

        Key arguments:
        A -- Coefficient matrix.
        b -- Linearly independent vector.
        x_current -- Current answer's approximation.
        x_next -- vector in which to store new answer.
        n -- Coefficient matrix' size.
        rel -- Relaxation coefficient.
        """
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx < n:
            sigma = 0.0
            index = idx * n
            for j in range(0, n):
                if idx != j:
                    sigma += A[index + j] * x_current[j]

            x_next[idx] = (b[idx] - sigma) / A[index + idx]
            x_next[idx] = rel * x_next[idx] + (1 - rel) * x_current[idx]


    @cuda.jit('void(float64[:], float64[:], float64[:], int32)',\
              target='gpu', nopython=True)
    def get_error(x_current, x_next, x_error, rows):
        """Calculates jacobi's maximum error"""
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx < rows:
            x_error[idx] = abs(x_next[idx] - x_current[idx])


    def start(self, A, b, niter, tol, rel=1):
        """Launches parallel jacobi solver for a SLAE and returns its answer.

        Keyword arguments:
        A -- Coefficient matrix of a SLAE.
        b -- Linearly independent vector of a SLAE.
        niter -- Maximum number of iterations before jacobi stops.
        tol -- Maximum error reached by jacobi when solving the system.
        rel -- relaxation coefficient.
        """
        if 0 in A.diagonal():
            return None, None, None

        A = A.flatten()
        b = b.flatten()
        tpb = 32
        bpg = len(A) + (tpb - 1) // tpb
        length = len(b)
        x_current = np.zeros(length, dtype=np.float64)
        x_next = np.zeros(length, dtype=np.float64)
        x_error = np.zeros(length, dtype=np.float64)
        gpu_A = cuda.to_device(A)
        gpu_b = cuda.to_device(b)
        gpu_x_current = cuda.to_device(x_current)
        gpu_x_next = cuda.to_device(x_next)
        gpu_x_error = cuda.to_device(x_error)
        count = 0
        error = tol + 1

        while error > tol and count < niter:
            if count % 2:
                self.jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_current,\
                                      gpu_x_next, length, rel)
                self.get_error[bpg, tpb](gpu_x_current, gpu_x_next,\
                                         gpu_x_error, length)
            else:
                self.jacobi[bpg, tpb](gpu_A, gpu_b, gpu_x_next,\
                                      gpu_x_current, length, rel)
                self.get_error[bpg, tpb](gpu_x_next, gpu_x_current,\
                                         gpu_x_error, length)

            x_error = gpu_x_error.copy_to_host()
            error = max(x_error)
            count += 1

        if error < tol:
            if count % 2:
                x_next = gpu_x_next.copy_to_host()
            else:
                x_next = gpu_x_current.copy_to_host()

            if True in np.isnan(x_next) or True in np.isinf(x_next):
                return None, None, None

            print("Jacobi done with an error of {} and iter {}".format(error,\
                                                                       count))
            return x_next, count, error
        else:
            print("Jacobi failed")
            return None, count, error
