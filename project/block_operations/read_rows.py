#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numba import cuda
import numpy as np
import time, csv, sys
from jacobi_parallel_chunks import JacobiParallel

def start(a_name, b_name, rows_to_read, matrix_size, niter):
  b_file = open(b_name).read().split('\n')[:-1]
  A_file = open(a_name).read().split('\n')[:-1]
  jp = JacobiParallel()

  x_current = np.zeros(matrix_size, dtype=np.float64)
  count = 0
  while count < niter:
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
        x = jp.start(np.array(A).flatten(),\
                      np.array(b),\
                      x_current, (count2-rows_to_read))

        x_next = np.append(x_next, x)
        A = [A_line]
        b = [b_value]
      count2 += 1
    x = jp.start(np.array(A).flatten(), np.array(b),\
                 x_current, (count2 - len(A)))
    x_next = np.append(x_next, x)
    x_current = np.array(x_next).flatten()
    count += 1

    return x_current

