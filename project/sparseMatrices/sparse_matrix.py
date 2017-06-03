#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File name: sparse_matrix.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 29-May-2017
    Python Version: 3.6.0
"""

import numpy as np
import random
import json
import sys
from pprint import pprint


class SparseMatrix():
    def create_sparse_matrix(self, filename, matrix_length, density):
        """Creates a sparse matrix with CSR format (four arrays)

        keyword arguments:
        filename -- The file name where will be stored the final result.
        matrix_length -- The length of the matrix.
        density -- percentage of non-zeros elements
        """
        pos = 0
        aux_pos = 0
        matrix = []
        pointerB = []
        pointerE = []
        columns = []
        values = []

        for i in range(0, matrix_length):
            row = []
            #if (pos != aux_pos):
            pointerB.append(pos)
            aux_pos = pos
            for j in range(0, matrix_length):
                probability = random.random()
                if probability < density:
                    pos += 1
                    val = random.randint(1, 10)
                    values.append(val)
                    columns.append(j)
                else:
                    val = 0
                row.append(val)
            matrix.append(row)
            #if (pos != aux_pos):
            pointerE.append(pos)
        print(matrix)
        data = {"values": values, "columns": columns, "pointerB": pointerB, "pointerE": pointerE}
        data_json = json.dumps(data)
        file = open(filename, 'w')
        file.write(data_json)
        file.close()

    def load_sparse_matrix(self, filename):
        """Takes a file and get the values array of it.

        keyword arguments:
        filename -- The file name where arrayes are stored.
        """
        with open(filename) as data_file:
            data = json.load(data_file)
        values = data["values"]
        print("JSON")
        print(values)

    def multiply(self, filename_matrix, vector):
        with open(filename_matrix) as f_matrix:
            matrix1 = json.load(f_matrix)

        values_matrix1 = matrix1["values"]
        columns_matrix1 = matrix1["columns"]
        pointerB_matrix1 = matrix1["pointerB"]
        pointerE_matrix1 = matrix1["pointerE"]

        res = []
        for i in range(0, len(vector)):
            val = 0
            try:
                for j in range(pointerB_matrix1[i], pointerE_matrix1[i]):
                    val += values_matrix1[j] * vector[columns_matrix1[j]]
            except:
                val = 0
            res.append(val)
        return res
                

def main(argv):
    if len(argv) != 4:
        print("Unsage: ./program filename matrix_length density")
        sys.exit()
    sparseMatrix = SparseMatrix()
    sparseMatrix.create_sparse_matrix(argv[1], int(argv[2]), float(argv[3]))
    sparseMatrix.multiply(argv[1], [9,8,7,6,5,4,3,2,1,0])
  
if __name__ == '__main__':
    main(sys.argv)