import sys
import numpy
from random import randrange

class CoefficientGenerator():
    def gen_matrix(self, filename, size):
        matrix = []
        for i in range(size):
            row = []
            for j in range(size):
                rand_num = randrange(1,size+1)
                row.append(rand_num)
            # ensure diagonal dominance here:
            row[i] = sum(row) + 1
            matrix.append(row)
        numpy.savetxt(filename, matrix, fmt="%1.9f", delimiter=",")
        return matrix

    def gen_vector(self, filename, size):
        solution = []
        for i in range(size):
            rand_num = randrange(1,size+1)
            solution.append(rand_num)
        numpy.savetxt(filename, solution, fmt="%1.9f", delimiter=",")
        return solution


def main(argv):
    if len(argv) != 4:
        print("Usage: python gen_diag_dominant_matrix.py <size> <output_filename for A> <output_filename for b>\n")
        sys.exit()

    size = int(argv[1])
    fname = argv[2]
    fname2 = argv[3]
    
    coefficient_generator = CoefficientGenerator()
    matrix = coefficient_generator.gen_matrix(fname, size)
    vector = coefficient_generator.gen_vector(fname2, size)

if __name__ == "__main__":
    main(sys.argv)
