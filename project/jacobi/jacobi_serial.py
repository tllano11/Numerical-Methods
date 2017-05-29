#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
  File name: jacobi_serial.py
  Authors: Tomás Felipe Llano Ríos,
           Juan Diego Ocampo García,
           Johan Sebastián Yepes Ríos
  Date last modified: 29-May-2017
  Python Version: 3.6.0
"""

import sys
import numpy as np
import csv


class SerialJacobi:
    def multiply_matrix_vector(self, A_matrix, b_vector):
        """Returns the dot product between a matrix and a vector

        keyword arguments:
        A_matrix -- The matrix to be multiplied.
        b_vector -- The vector to be multiplied.
        """
        size = len(A_matrix)
        result = []
        for i in range(0, size):
            tmp = 0
            for j in range(0, size):
                tmp += A_matrix[i][j] * b_vector[j]
            result.append(tmp)
        return result

    def multiply_matrix_matrix(self, matrix1, matrix2):
        """Returns the dot product between two matrices.

        keyword arguments:
        A_matrix -- The first matrix to be multiplied.
        b_vector -- The second matrix to be multiplied.
        """
        size = len(matrix1)
        matrix_result = []
        for i in range(0, size):
            row = []
            for j in range(0, size):
                row.append(0)
                for k in range(0, size):
                    row[j] += matrix1[i][k] * matrix2[k][j]
            matrix_result.append(row)
        return matrix_result

    def get_D_and_U(self, matrix):
        """Split a given matrix into two matrices D and U (uower and upper triangular matrices)

        keyword arguments:
        matrix -- The matrix to be splited
        """
        matrixD = []
        matrixU = []
        size = len(matrix)
        for i in range(0, size):
            rowD = []
            rowU = []
            for j in range(0, size):
                if i == j:
                    rowD.append(matrix[i][j])
                    rowU.append(0)
                else:
                    rowD.append(0)
                    rowU.append(-matrix[i][j])
            matrixD.append(rowD)
            matrixU.append(rowU)
        return matrixD, matrixU

    def get_inverse(self, matrixD):
        """Returns the inverse of a LOWER TRIANGULAR MATRIX.

        keyword arguments:
        A_matrix -- The matrix base to calculate the inverse.
        """
        size = len(matrixD)
        for i in range(0, size):
            matrixD[i][i] = pow(matrixD[i][i], -1)
        return matrixD

    def sum_vectors(self, vector1, vector2):
        """Takes two vector and sum them.

        keyword arguments:
        vector1 -- The first vector to be added.
        vector2 -- The second vector to be added.
        """
        size = len(vector1)
        result = []
        for i in range(0, size):
            result.append(vector1[i] + vector2[i])
        return result

    def get_error(self, x_vector, xant_vector):
        """Returns the norm of two given vectors,
        the norm represents the error of the current method.

        keyword arguments:
        x_vector -- The vector of the current stage of the method.
        xant_vector -- The vector of the previous stage of the method.
        """
        maximum = 0
        size = len(x_vector)
        for i in range(0, size):
            tmp = float(abs(x_vector[i] - xant_vector[i]))
            if tmp > maximum:
                maximum = tmp
        return maximum

    def relaxation(self, x_vector, xant_vector, relaxation):
        """Applies the relaxation method to Jacobi.

        keyword arguments:
        x_vector -- The vector of the current stage of the method.
        xant_vector -- The vector of the previous stage of the method.
        relaxation -- The number that will be used in the relaxation of the method.
        """
        size = len(x_vector)
        xrelax_vector = []
        for i in range(0, size):
            xrelax_vector.append(relaxation * x_vector[i] + (1 - relaxation) * xant_vector[i])
        return xrelax_vector

    def jacobi(self, A_matrix, b_vector, max_iterations, tolerance, relaxation=1):
        """Applies Jacobi method to a system of linear equations.

        keyword arguments:
        A_matrix -- The coefficient matrix of the system.
        b_vector -- The linearly independent vector.
        max_iterations -- Maximum number of iterations of the method.
        tolerance -- The tolerance of the method
        relaxation -- The number that will be used in the relaxation of the method.
        """
        size = len(A_matrix)
        x_vector = [0] * size
        error = tolerance + 1
        matrixD, matrixU = self.get_D_and_U(A_matrix)
        matrixD = self.get_inverse(matrixD)
        vector1 = self.multiply_matrix_vector(matrixD, b_vector)
        matrix_aux = self.multiply_matrix_matrix(matrixD, matrixU)
        count = 0
        while error > tolerance and count < max_iterations:
            xant_vector = x_vector
            vector2 = self.multiply_matrix_vector(matrix_aux, xant_vector)
            x_vector = self.sum_vectors(vector1, vector2)
            x_vector = self.relaxation(x_vector, xant_vector, relaxation)
            error = self.get_error(x_vector, xant_vector)
            count += 1
        if count > max_iterations:
            return None, count, error
        else:
            return x_vector, count, error
