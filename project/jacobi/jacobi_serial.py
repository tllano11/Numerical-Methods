from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot, ones, random
import time, csv

class JacobiSerial():

    def jacobi(self, A_matrix,b_vector,niter):
        #Solves the equation Ax=b via the Jacobi iterative method.
        # Create an initial guess if needed
        x_vector = zeros(len(b_vector))

        # Create a vector of the diagonal elements of A
        # and subtract them from A
        D_matrix = diag(A_matrix)
        R_matrix = A_matrix - diagflat(D_matrix)

        # Iterate for N times
        for i in range(niter):
            x_vector = (b_vector - dot(R_matrix,x_vector)) / D_matrix
        print("X:")
        pprint(x_vector)
        return x_vector

