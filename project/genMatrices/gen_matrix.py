import sys
import numpy
from random import randrange

def gen_matrix(size):
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            rand_num = randrange(1,size+1)
            row.append(rand_num)
        # ensure diagonal dominance here:
        row[i] = sum(row) + 1
        matrix.append(row)
    return matrix

def get_random_solution(size):
    solution = []
    for i in range(size):
        rand_num = randrange(1,size+1)
        solution.append(rand_num)
    return solution


def main(argv):
    if len(argv) != 4:
        print("Usage: python gen_diag_dominant_matrix.py <size> <output_filename for A> <output_filename for b>\n")
        sys.exit()

    size = int(argv[1])
    fname = argv[2]
    fname2 = argv[3]

    matrix = gen_matrix(size)
    solution = get_random_solution(size)

    numpy.savetxt(fname, matrix, fmt="%1.9f", delimiter=",")
    numpy.savetxt(fname2, solution, fmt="%1.9f", delimiter=",")

if __name__ == "__main__":
    main(sys.argv)
