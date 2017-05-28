import sys
import numpy as np

class LUDecomposition():

  def LU_decomposition(self, A):
    n = len(A)
    L = np.zeros(shape=(n, n))
    P = np.identity(n)
    for k in range(0, n-1):
      L[k][k] = 1
      #A, P = self.partial_pivot(A, P, k)
      for i in range(k + 1, n):
        multiplier = A[i][k]/A[k][k]
        L[i][k] = multiplier
        for j in range(k, n):
          A[i][j] = A[i][j] - multiplier * A[k][j]
    L[n-1][n-1] = 1
    return L, A, P

  # In this case, k is the number of the 
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

  def back_substitution(self, U, z):
    n = len(U)
    x = [0]*n
    for i in range(n-1, -1, -1):
      if U[i][i] != 0:
        accum = 0
        for j in range(i+1, n):
          accum += U[i][j]*x[j]
        x[i] = ((z[i] - accum)/U[i][i])
    return x

  def forward_substitution(self, L, b):
    n = len(L)
    z = [0]*n
    for i in range(0, n):
      if L[i][i] != 0:
        accum = 0
        for j in range(0, i):
          accum += L[i][j]*z[j]
        z[i] = (b[i] - accum)/ L[i][i]
    return z

if __name__ == '__main__':
  #A =  np.array([[-7, 2, -3, 4], [5, -1, 14, -1], [1, 9, -7, 5], [-12, 13, -8, -4]], dtype="float")
  #b = np.array([-12, 13, 31, -32], dtype="float")
  A =  np.array([[4,3,-2,-7], [3, 12, 8, -3], [2, 3, -9, 2], [1, -2, -5, 6]], dtype="float")
  b = np.array([20, 18, 31, 12], dtype="float")
  LUdecomposition = LUDecomposition()
  L, U, P = LUdecomposition.LU_decomposition(A)
  print("L")
  print(L)
  print(U)
  z = LUdecomposition.forward_substitution(L, b)
  print(z)
  x = LUdecomposition.back_substitution(U, z)
  print(x)