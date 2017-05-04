import sys
import numpy
from random import randrange, random, uniform

class CoefficientGenerator():
    # Generates matrix A
    def gen_matrix(self, filename, size):
        matrix = []
        for i in range(size):
            row = []
            for j in range(size):
                rand_num = uniform(-size,size+1)
                #rand_num = randrange(-size,size+1)
                row.append(rand_num)
            # ensure diagonal dominance here:
            for value in row:
              row[i] += abs(value)
            row[i] += 1
            matrix.append(row)
        # Save file with numpy
        numpy.savetxt(filename, matrix, fmt="%1.9f", delimiter=" ")
        return matrix

    # Generates vector b
    def gen_vector(self, filename, size):
        solution = []
        for i in range(size):
            rand_num = randrange(1,size+1)
            solution.append(rand_num)
        # Save file with numpy
        numpy.savetxt(filename, solution, fmt="%1.9f", delimiter=" ")
        return solution


def main(argv):
    if len(argv) != 4:
        print("Usage: python coefficient_generator.py <size> <output_filename for A> <output_filename for b>")
        sys.exit()

    size = int(argv[1])
    fname = argv[2]
    fname2 = argv[3]

    coefficient_generator = CoefficientGenerator()
    matrix = coefficient_generator.gen_matrix(fname, size)
    vector = coefficient_generator.gen_vector(fname2, size)

if __name__ == "__main__":
    main(sys.argv)
