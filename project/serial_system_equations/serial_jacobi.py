import sys
import numpy as np
import csv

class SerialJacobi():

	def multiply_matrix_vector(self, A_matrix, b_vector):
		size = len(A_matrix)
		result = []
		for i in range(0, size):
			tmp = 0
			for j in range(0, size):
				tmp += A_matrix[i][j] * b_vector[j]
			result.append(tmp)
		return result

	def multiply_matrix_matrix(self, matrix1, matrix2):
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

	def getDandU(self, matrix):
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

	def getInverse(self, matrixD):
		size = len(matrixD)
		for i in range(0, size):
			matrixD[i][i] = pow(matrixD[i][i], -1)
		return matrixD

	def sum_vectors(self, vector1, vector2):
		size = len(vector1)
		result = []
		for i in range(0, size):
			result.append(vector1[i] + vector2[i])
		return result

	def get_error(self, x_vector, xant_vector):
		max = 0
		tmp = 0
		size = len(x_vector)
		for i in range(0, size):
			tmp = abs(x_vector[i] - xant_vector[i])
			if(tmp > max):
				max = tmp
		return max

	def jacobi(self, A_matrix, b_vector, max_iterations, tolerance):
		size = len(A_matrix)
		x_vector = [0] * size
		error = tolerance + 1
		matrixD, matrixU = self.getDandU(A_matrix)
		matrixD = self.getInverse(matrixD)
		vector1 = self.multiply_matrix_vector(matrixD, b_vector)
		matrix_aux = self.multiply_matrix_matrix(matrixD, matrixU)
		count = 0
		while error > tolerance and count < max_iterations:
			xant_vector = x_vector
			vector2 = self.multiply_matrix_vector(matrix_aux,x_vector)
			x_vector = self.sum_vectors(vector1, vector2)
			error = self.get_error(x_vector, xant_vector)
			count += 1
		if error < tolerance:
			print("The solution is: ", x_vector)
		else:
			print("Sorry, it failed in ", count, "iterations")
		return x_vector


if __name__ == '__main__':
	with open('../../a.txt') as A_file:
		reader = csv.reader(A_file, delimiter=' ')
		matrix = list(reader)
		A_matrix = np.array(matrix).astype("float")
	with open('../../b.txt') as b_file:
		reader = csv.reader(b_file, delimiter=' ')
		vector = list(reader)
		b_vector = np.array(vector).astype("float")
	#A_matrix = [[4, 3, -2, -7],[3, 12, 8, -3], [2, 3, -9, 2], [1, -2, -5, 6]]
	#b_vector = [20, 18, 21, 12]
	max = 1000
	tol = 0.001
	serial_jacobi =	SerialJacobi()
	serial_jacobi.jacobi(A_matrix, b_vector, max, tol)
