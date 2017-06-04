#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File name: sparse_matrix.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 03-June-2017
    Python Version: 3.6.0
"""

import numpy as np
import random
import json
import sys
from pprint import pprint
import csv
from random import uniform


class SparseMatrix():
    def create_sparse_matrix(self, filename, matrix_length, density):
        matrix = []
        for i in range(0, matrix_length):
            row = []
            for j in range(0, matrix_length):
                probability = random.random()
                if probability < density:
                    val = random.randint(1, 10)
                else:
                    val = 0
                row.append(val)
            matrix.append(row)
        matrix_A = np.matrix(matrix)
        return matrix_A

def main(argv):
    if len(argv) != 4:
        print("Unsage: ./program filename matrix_length density")
        sys.exit()
    sparseMatrix = SparseMatrix()
    matrix_A= sparseMatrix.create_sparse_matrix(argv[1], int(argv[2]), float(argv[3]))
    np.savetxt(argv[1]+"_A", matrix_A, fmt="%1.9f", delimiter=" ")


if __name__ == '__main__':
    main(sys.argv)
