import sys
import numpy as np

class LUDecomposition():

  def LU_decomposition(self, A):
    n = len(A)
    L = np.zeros(shape=(n, n))
    P = np.identity(n)
    for k in range(0, n-1):
      A, P = self.partial_pivot(A, P, k)
      for i in range(k + 1, n):
        multiplier = A[i][k]/A[k][k]
        L[i][k] = multiplier
        for j in range(k, n):
          if i == j:
            L[i][j] = 1
          A[i][j] = A[i][j] - multiplier * A[k][j]
    return L, A, P

  def partial_pivot(self, A, P, k):
    maximum = abs(A[k][k])
    max_row = k
    n = len(A)
    for s in range(k+1, n):
      if abs(A[s][k]) > maximum:
        maximum = abs(A[s][k])
        max_row = s
    if(maximum != 0):
      if(max_row != k):
        aux_A = np.copy(A[k])
        A[k] = np.copy(A[max_row])
        A[max_row] = np.copy(aux_A)
        aux_P = np.copy(P[k])
        P[k] = np.copy(P[max_row])
        P[max_row] = aux_P
    return A, P
if __name__ == '__main__':
  A =  np.array([[-7, 2, -3, 4], [5, -1, 14, -1], [1, 9, -7, 5], [-12, 13, -8, -4]], dtype="float")
  b = np.array([-12, 13, 31, -32], dtype="float")
  LUdecomposition = LUDecomposition()
  L, U, P = LUdecomposition.LU_decomposition(A)
  print(L)
  print(U)
  print(P)