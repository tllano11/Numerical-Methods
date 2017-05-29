import sys
import numpy as np


class GaussJordanSerial:

  def elimination(self, A, b):
    n = len(A)
    for k in range(0, n):
      for i in range(0, n): 
        if i != k:
          multiplier = A[i][k]/A[k][k]
          for j in range(k,n):
            A[i][j] = A[i][j] - multiplier * A[k][j]
          b[i] = b[i] - multiplier * b[k]
    for i in range(0, n):
      b[i] = b[i]/A[i][i]
      A[i][i] = A[i][i]/A[i][i]
    return b

if __name__ == '__main__':
  A =  np.array([[14, 6, -2, 3], [3, 15, 2, -5], [-7, 4, -23, 2], [1, -3, -2, 16]], dtype="float")
  b = np.array([12, 32, -24, 14], dtype="float")
  #A =  np.array([[25, -3, 4, -7],[3, -17, 4, -5], [5, -6, 37, -8], [3, -8, 5, -73]], dtype="float")
  #b = np.array([208, -32, 29, 128], dtype="float")
  #A =  np.array([[-7, 2, -3, 4], [5, -1, 14, -1], [1, 9, -7, 5], [-12, 13, -8, -4]], dtype="float")
  #b = np.array([-12, 13, 31, -32], dtype="float")
  #A =  np.array([[4, 3, -2, -7], [3, 12, 8, -3], [2, 3, -9, 2], [1, -2, -5, 6]], dtype="float")
  #b = np.array([20, 18, 31, 12], dtype="float")
  gauss = GaussianElimination()
  x = gauss.elimination(A, b)
  print(x)
