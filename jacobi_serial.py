from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot, ones, random
import time, csv

def jacobi(A,b,N=25,x=None):
    """Solves the equati)n Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed
    if x is None:
        x = zeros(50)

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times
    for i in range(N):
        x = (b - dot(R,x)) / D
    return x

with open("../example.txt") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    matrix = list(reader)
    A_matrix = array(matrix).astype("float")
#A_matrix = A_matrix.flatten()

with open("../b.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    matrix = list(reader)
    b_matrix = array(matrix).astype("float")
#    b_matrix = b_matrix.flatten()

#A = array([[8,-1,7],[-2,3,1],[0,1,9]])
#A = random.random((r, c))
#b = array([4,0,1])
#b = random.random((1200, 1))
guess = [0]*50
#guess = ones(1200)
start = time.time()
sol = jacobi(A_matrix,b_matrix,N=100)
end = time.time()

print(end - start)

print("X:")
pprint(sol)
