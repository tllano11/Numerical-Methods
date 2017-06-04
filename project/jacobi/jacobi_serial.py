#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@package Jacobi
Solve a system of linear algebraic equations by using the Jacobi
Iterative method.
"""

"""
  File name: jacobiSerial.py
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
        """Returns the dot product between a matrix and a vector.

        @param A_matrix  The matrix to be multiplied.
        @param b_vector  The vector to be multiplied.

        @return float128[:]
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

        @param matrix1  The first matrix to be multiplied.
        @param matrix2  The second matrix to be multiplied.

        @return float128[:]
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
        """Split a given matrix into two matrices D and U (lower and upper triangular matrices)

        @param matrix  The matrix to be splitted.
        @return float128[:,:],float128[:,:]
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

        @param matrixD The matrix base to calculate the inverse.

        @return float128[:,:]
        """
        size = len(matrixD)
        for i in range(0, size):
            if matrixD[i][i] == 0:
                return None
            matrixD[i][i] = pow(matrixD[i][i], -1)
        return matrixD

    def sum_vectors(self, vector1, vector2):
        """Takes two vector and sum them.

        @param  vector1  The first vector to be added.
        @param  vector2  The second vector to be added.

        @return float128[:]
        """
        size = len(vector1)
        result = []
        for i in range(0, size):
            result.append(vector1[i] + vector2[i])
        return result

    def get_error(self, x_vector, xant_vector):
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

    def relaxation(self, x_vector, xant_vector, relaxation):
        """Applies the relaxation method to Jacobi.

        @param  x_vector      The vector of the current stage of the method.
        @param  xant_vector   The vector of the previous stage of the method.
        @param  relaxation    The number that will be used in the
                              relaxation of the method.

        @return  float128[:]
        """
        size = len(x_vector)
        xrelax_vector = []
        for i in range(0, size):
            xrelax_vector.append(relaxation * x_vector[i] + (1 - relaxation) * xant_vector[i])
        return xrelax_vector

    def jacobi(self, A_matrix, b_vector, max_iterations, tolerance, relaxation=1):
        """Applies Jacobi method to a system of linear equations and returns
        its answer (except if it was not found), number of iterations executed
        and the maximum error.

        @param  A_matrix        The coefficient matrix of the system.
        @param  b_vector        The linearly independent vector.
        @param  max_iterations  Maximum number of iterations of the method.
        @param  tolerance       The tolerance of the method
        @param  relaxation      The number that will be used in the
                                relaxation of the method.

        @return float128[:] or None, int32, float128
        """
        size = len(A_matrix)
        x_vector = [0] * size
        error = tolerance + 1
        matrixD, matrixU = self.get_D_and_U(A_matrix)
        matrixD = self.get_inverse(matrixD)
        if matrixD is None:
            return None, None, None

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

        if error < tolerance:
            return x_vector, count, error
        else:
            return None, count, error
