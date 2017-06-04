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
        pos = 0
        aux_pos = 0
        pointerB = []
        pointerE = []
        columns = []
        values = []

        for i in range(0, matrix_length):
            row = []
            pointerB.append(pos)
            aux_pos = pos
            for j in range(0, matrix_length):
                probability = random.random()
                if probability < density:
                    pos += 1
                    val = random.randint(1, 10)
                    values.append(val)
                    columns.append(j)
            pointerE.append(pos)
        data = {"values": values, "columns": columns, "pointerB": pointerB, "pointerE": pointerE}
        CSR_A = json.dumps(data)
        return CSR_A

def main(argv):
    if len(argv) != 4:
        print("Unsage: ./program filename matrix_length density")
        sys.exit()
    sparseMatrix = SparseMatrix()
    CSR_A = sparseMatrix.create_sparse_matrix(argv[1], int(argv[2]), float(argv[3]))
    file = open(argv[1]+"_CSR", 'w')
    file.write(CSR_A)
    file.close()

if __name__ == '__main__':
    main(sys.argv)
