#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numba import cuda
import numpy as np
import time, csv, sys
from jacobi_parallel_chunks import JacobiParallel

def main(argv):
  a_name = argv[1]
  b_name = argv[2]
  rows_to_read = int(argv[3])
  matrix_size = int(argv[4])
  niter = int(argv[5])

  b_file = open(b_name).read().split('\n')[:-1]
  A_file = open(a_name).read().split('\n')[:-1]
  jp = JacobiParallel()

  x_current = np.zeros(matrix_size, dtype=np.float64)
  count = 0
  print("Comenzando jacobi ...")
  while count < niter:
    print("Iteracion: ", count)
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
        #print(x_current)
        x_next = np.append(x_next, x)
        A = [A_line]
        b = [b_value]
      count2 += 1
    x = jp.start(np.array(A).flatten(), np.array(b),\
                 x_current, (count2 - len(A)))
    x_next = np.append(x_next, x)
    x_current = np.array(x_next).flatten()
    print("x_current: \n", x_current[:10])
    count += 1

  print(x_next[:10])

if __name__ == "__main__":
  main(sys.argv)
