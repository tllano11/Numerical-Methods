#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File name: matrix_generator.py
    Authors: Tomás Felipe Llano Ríos,
             Juan Diego Ocampo García,
             Johan Sebastián Yepes Ríos
    Date last modified: 29-May-2017
    Python Version: 3.6.0
"""

import sys
import numpy as np
from random import randrange, random, uniform


class MatrixGenerator:
    @staticmethod
    def gen_vector(size):
        """Creates a random vector given a size.

        keyword arguments:
        size -- Length of the vector that will be created.
        """
        solution = []
        for i in range(size):
            rand_num = uniform(-size, size)
            solution.append(rand_num)
        return np.array(solution)

    @staticmethod
    def gen_dominant(size):
        """Creates a diagonally dominant matrix given a size.

        keyword arguments:
        size -- Number of rows and columns that the matrix will have.
        """
        matrix = []
        for i in range(size):
            row = []
            for j in range(size):
                rand_num = uniform(-size, size + 1)
                row.append(rand_num)
            # ensure diagonal dominance here:
            for value in row:
                row[i] += abs(value)
            row[i] += 1
            matrix.append(row)
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return matrix_A, vector_x, vector_b

    @staticmethod
    def gen_symmetric_matrix(size):
        """Creates a symmetric matrix given a size.

        keyword arguments:
        size -- Number of rows and columns that the matrix will have.
        """
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            for j in range(0, i + 1):
                value = uniform(-size, size)
                matrix[i][j] = value
                matrix[j][i] = value
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_band_matrix(size, k1, k2):
        """Creates a band matrix given a size.

        keyword arguments:
        size -- Number of rows and columns that the matrix will have.
        k1 -- Number of diagonals with non-zero elements below the main diagonal (Inclusive).
        k2 -- Number of diagonals with non-zero elements above the main diagonal (Inclusive).
        """
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            for j in range(0, size):
                if j <= i - k1 or j >= i + k2:
                    matrix[i][j] = 0
                else:
                    matrix[i][j] = 1
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_identity_matrix(size):
        """Creates an identity matrix given a size.

        keyword arguments:
        size -- Number of rows and columns that the matrix will have.
        """
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            matrix[i][i] = 1
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_diagonal_matrix(size):
        """Creates a diagonal matrix given a size.

        keyword arguments:
        size -- Number of rows and columns that the matrix will have.
        """
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            for j in range(0, size):
                if i == j:
                    matrix[i][j] = uniform(-size, size)
                vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_scalar_matrix(size):
        """Creates a scalar matrix given a size.

        keyword arguments:
        size -- Number of rows and columns that the matrix will have.
        """
        matrix = np.zeros(shape=(size, size))
        value = uniform(-size, size)
        for i in range(0, size):
            matrix[i][I] = value
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_antisymmetric_matrix(size):
        """Creates an anti-symmetric matrix given a size.

        keyword arguments:
        size -- Number of rows and columns that the matrix will have.
        """
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            for j in range(0, i + 1):
                if (i == j):
                    matrix[i][j] = 0
                else:
                    value = uniform(-size, size)
                    matrix[i][j] = value
                    matrix[j][i] = -value
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_lower_matrix(size):
        """Creates a lower triangular matrix given a size.

        keyword arguments:
        size -- Number of rows and columns that the matrix will have.
        """
        matrix = np.zeros(shape=(size, size))
        for i in range(size):
            for j in range(size):
                if i > j:
                    rand_num = uniform(-size, size)
                    matrix[i][j] = rand_num
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_upper_matrix(size):
        """Creates an upper triangular matrix given a size.

        keyword arguments:
        size -- Number of rows and columns that the matrix will have.
        """
        matrix = np.zeros(shape=(size, size))
        for i in range(size):
            for j in range(size):
                if i < j:
                    rand_num = uniform(-size, size)
                    matrix[i][j] = rand_num
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)