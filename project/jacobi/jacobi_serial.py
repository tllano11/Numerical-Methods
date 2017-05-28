import csv
import sys
import time
from numpy import array, diag, diagflat, dot, zeros_like


class JacobiSerial:

  def jacobi(self, A_matrix,b_vector,niter, tol=0):
    #Solves the equation Ax=b via the Jacobi iterative method.
    # Create an initial guess if needed
    x_next = zeros_like(b_vector)
    x_vector = zeros_like(b_vector)
    count = 0
    error = tol + 1

    # Create a vector of the diagonal elements of A
    # and subtract them from A
    D_matrix = diag(A_matrix)
    R_matrix = A_matrix - diagflat(D_matrix)
    # Iterate for N times
    start = time.time()
    while count < niter:
      x_next = (b_vector - dot(R_matrix,x_next)) / D_matrix
#print(x_next)
#      print(x_vector)
#      error = self.get_max_error(x_vector, x_next)
#      x_vector = copy(x_next)
      count += 1
    end = time.time()

    print("Computation Time was: {}".format(end - start))
    #print("X:")
    # print(x_next)
    return x_next

  def get_max_error(self, x_vector, x_next):
    x_error = zeros_like(x_vector)
    for i in range(len(x_vector)):
      x_error[i] = abs(x_next[i] - x_vector[i])

    return max(x_error)

if __name__ == "__main__":
  jacobi = JacobiSerial()
  A_name = sys.argv[1]
  b_name = sys.argv[2]

  with open(A_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    matrix = list(reader)
    A_matrix = array(matrix).astype("float64")

  with open(b_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=' ')
    matrix = list(reader)
    b_vector = array(matrix).astype("float64")

  jacobi.jacobi(A_matrix, b_vector, 60)
