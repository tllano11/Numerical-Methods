from numba import cuda
import numpy as np

def to_gpu(flatten_matrix):
  gpu_pointer_to_matrix = cuda.to_device(flatten_matrix)
  return gpu_pointer_to_matrix

def to_cpu(gpu_matrix):
  cpu_matrix = gpu_matrix.copy_to_host()
  return cpu_matrix
