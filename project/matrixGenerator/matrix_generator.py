import sys
import numpy as np
from random import randrange, random, uniform

class MatrixGenerator():

    # gens vector b
    @staticmethod
    def gen_vector(size):
        solution = []
        for i in range(size):
            rand_num = uniform(-size, size)
            solution.append(rand_num)
        return np.array(solution)

    # gens matrix A
    @staticmethod
    def gen_dominant(size):
        matrix = []
        for i in range(size):
            row = []
            for j in range(size):
                rand_num = uniform(-size, size+1)
                # rand_num = randrange(-size,size+1)
                row.append(rand_num)
            # ensure diagonal dominance here:
            for value in row:
                row[i] += abs(value)

            row[i] += 1
            matrix.append(row)
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_symmetric_matrix(size):
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            for j in range(0, i+1):
                value = uniform(-size, size)
                matrix[i][j] = value
                matrix[j][i] = value
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_band_matrix(size, band):
        matrix = np.zeros(shape=(size, size))
        width = int((band - 1)/2)
        for i in range(0, size):
            for j in range(i - width, i + width + 1):
                if 0 <= j < size:
                    matrix[i][j] = 1  # uniform(-size, size)
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_identity_matrix(size):
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            for j in range(0, size):
                if i == j:
                    matrix[i][j] = 1
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_diagonal_matrix(size):
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            for j in range(0, size):
                if i == j:
                    matrix[i][j] = uniform(-size, size)
        return matrix

    @staticmethod
    def gen_scalar_matrix(size):
        matrix = np.zeros(shape=(size, size))
        value = uniform(-size, size)
        for i in range(0, size):
            for j in range(0, size):
                if i == j:
                    matrix[i][j] = value
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_scalar_matrix(size):
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
    def gen_antisymmetric_matrix(size):
        matrix = np.zeros(shape=(size, size))
        for i in range(0, size):
            for j in range(0, i+1):
                value = uniform(-size, size)
                matrix[i][j] = value
                matrix[j][i] = -value
        vector_x = MatrixGenerator.gen_vector(size)
        matrix_A = np.matrix(matrix)
        vector_b = np.dot(matrix_A, vector_x)
        return (matrix_A, vector_x, vector_b)

    @staticmethod
    def gen_lower_matrix(size):
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


def main(argv):
    if len(argv) != 4:
        print("Usage: python matrix_generator.py <size> <output_filename for A> <output_filename for b>")
        sys.exit()

    size = int(argv[1])
    fname = argv[2]
    fname2 = argv[3]

    matrix_generator = MatrixGenerator()
    matrix, x, vector = matrix_generator.gen_antisymmetric_matrix(size)
    print(matrix)
    print(x)
    print(vector)
    # vector = matrix_generator.gen_vector(fname2, size)

if __name__ == "__main__":
    main(sys.argv)
