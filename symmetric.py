import numpy as np
from random import uniform

def generate_symmetric_matrix(size):
  matrix = np.zeros(shape=(size,size))
  for i in range(0,size):
    for j in range(0,i+1):
      value = uniform(-size, size)
      matrix[i][j] = value 
      matrix[j][i] = value
  return matrix

def generate_band_matrix(size, band):
  matrix = np.zeros(shape=(size, size))
  width = int((band - 1)/2)
  for i in range(0, size):
    for j in range(i - width, i + width + 1):
      if j >= 0 and j < size:
        matrix[i][j] = 1 #uniform(-size, size)
  return matrix

def generate_identity_matrix(size):
  matrix = np.zeros(shape=(size, size))
  for i in range(0, size):
    for j in range(0, size):
      if i == j:
        matrix[i][j] = 1
  return matrix
  
def generate_diagonal_matrix(size): 
  matrix = np.zeros(shape=(size, size))
  for i in range(0, size):
    for j in range(0, size):
      if i == j:
        matrix[i][j] = uniform(-size, size)
  return matrix

def generate_scalar_matrix(size): 
  matrix = np.zeros(shape=(size, size))
  value = uniform(-size, size)
  for i in range(0, size):
    for j in range(0, size):
      if i == j:
        matrix[i][j] = value
  return matrix

def generate_scalar_matrix(size): 
  matrix = np.zeros(shape=(size, size))
  for i in range(0, size):
    for j in range(0, size):
      if i == j:
        matrix[i][j] = uniform(-size, size)
  return matrix

def generate_antisymmetric_matrix(size):
  matrix = np.zeros(shape=(size,size))
  for i in range(0,size):
    for j in range(0,i+1):
      value = uniform(-size, size)
      matrix[i][j] = value 
      matrix[j][i] = -value
  return matrix

if __name__ == '__main__':
  #generate_symmetric_matrix(5)
  #generate_band_matrix(10,5)
  generate_identity_matrix(10)
