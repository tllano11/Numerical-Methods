#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@package SparseMatrices
Represents a matrix with CSR format.
"""

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
    @staticmethod
    def gen_vector(size):
        """Creates a random vector given a size.

        @param size Length of the vector that will be created.
        @return float128[:]
        """
        solution = []
        for i in range(size):
            rand_num = uniform(-size, size)
            solution.append(rand_num)
        return np.array(solution)

    def create_sparse_matrix(self, filename, matrix_length, density):
        """Creates a sparse matrix with CSR format (four arrays)

        @param filename The file name where will be stored the final result.
        @param matrix_length The length of the matrix.
        @param density percentage of non-zeros elements
        @return float128[:,:], str, float128[:], float128[:]
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
            pointerE.append(pos)
        vector_x = SparseMatrix.gen_vector(matrix_length)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x).reshape(matrix_length, 1)
        data = {"values": values, "columns": columns, "pointerB": pointerB, "pointerE": pointerE}
        CSR_A = json.dumps(data)
        '''
        print("x: ", vector_x)
        print("A: ", matrix_A)
        print("b: ", vector_b)
        data = {"values": values, "columns": columns, "pointerB": pointerB, "pointerE": pointerE}
        data_json = json.dumps(data)
        file = open(filename, 'w')
        file.write(data_json)
        file.close()
        np.savetxt("vector.txt", vector_x, fmt="%1.9f", delimiter=" ")
        '''
        return matrix_A, CSR_A, vector_x, vector_b

    def load_sparse_matrix(self, filename):
        """Takes a file and get the values array of it.

        @param filename The file name where arrayes are stored.
        @return None
        """
        with open(filename) as data_file:
            data = json.load(data_file)
        values = data["values"]
        print("JSON")
        print(values)

    def multiply(self, filename_matrix, vector):
        """Takes a file with a sparse matrix in CSR format and multiply it with a vector.

        @param filename_matrix The filename where the CSR matrix is located.
        @param vector The vector to multiply with the matrix
        @return 128[:]
        """
        vector = vector.flatten()
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
        print(res)
        return res


def main(argv):
    if len(argv) != 4:
        print("Unsage: ./program filename matrix_length density")
        sys.exit()
    sparseMatrix = SparseMatrix()
    matrix_A, CSR_A, vector_x, vector_b = sparseMatrix.create_sparse_matrix(argv[1], int(argv[2]), float(argv[3]))
    '''
    with open("vector.txt") as vector_file:
        reader = csv.reader(vector_file, delimiter=' ')
        vector = list(reader)
        vector = np.array(vector).astype("float64")
        print(type(vector))
    '''
    np.savetxt(argv[1]+"_A", matrix_A, fmt="%1.9f", delimiter=" ")
    np.savetxt(argv[1]+"_x", vector_x, fmt="%1.9f", delimiter=" ")
    np.savetxt(argv[1]+"_b", vector_b, fmt="%1.9f", delimiter=" ")
    file = open(argv[1]+"_CSR", 'w')
    file.write(CSR_A)
    file.close()

if __name__ == '__main__':
    main(sys.argv)
