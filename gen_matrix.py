import sys
from random import randrange

def gen_matrix(size):
    matrix = []
    for i in xrange(size):
        row = []
        for j in xrange(size):
            rand_num = randrange(1,size+1)
            row.append(rand_num)
        # ensure diagonal dominance here:
        row[i] = sum(row) + 1
        matrix.append(row)
    return matrix

def get_random_solution(size):
    solution = []
    for i in xrange(size):
        rand_num = randrange(1,size+1)
        solution.append(rand_num)
    return solution


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print "Usage: python gen_diag_dominant_matrix.py <size> <output_filename for A> <output_filename for b>\n"
        sys.exit()

    size = int(sys.argv[1])
    fname = sys.argv[2]
    fname2 = sys.argv[3]

    matrix = gen_matrix(size)
    solution = get_random_solution(size)

    outfile = open(fname, 'w')
    outfile2 = open(fname2, 'w')

    for row in matrix:
        outfile.write('\n'.join(map(str,row)))
        outfile.write('\n')

    for item in solution:
      outfile2.write("%d\n" % item)

    #outfile.write('\n'.join(map(str,solution)))
    outfile.close
