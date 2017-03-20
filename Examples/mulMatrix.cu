#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <inttypes.h>
#include <iostream>

using namespace std;

const int N = 5;

// CPU copies of a, b, c
double a_cpu[N][N], b_cpu[N][N], c_cpu[N][N];

__global__ void mul(double *a, double *b, double *c,\
	     unsigned int a_width, unsigned int a_height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  double cellValue = 0;

  for (unsigned int i = 0; i < a_width; ++i) {
    if (row < a_height) {
      cellValue += *(a + a_width*row + i) * *(b + i);
    }
  }

  __syncthreads();

  if (row < a_height) {
    *(c + row) = cellValue;
    __syncthreads();
  }
}

void random_elements (int N) {
  int i;
  int j;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      a_cpu[i][j] = 1;
      b_cpu[i][j] = 1;
    }
  }
}

void print_matrix(int N) {
  int i;
  int j;
  cout << "Matrix:" << endl;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      cout << c_cpu[i][j] << endl;
    }
  }
}

#define THREADS_PER_BLOCK 64
int main (void) {

  // GPU copies of a, b, c
  double *a_gpu, *b_gpu, *c_gpu;
  int size = N * N * sizeof(int);

  cudaMalloc((void **)&a_gpu, size);
  cudaMalloc((void **)&b_gpu, size);
  cudaMalloc((void **)&c_gpu, size);

  // Set up random input variables
  random_elements(N);

  // Copy inputs to device
  cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyHostToDevice);

  int blockDimension = (N*N)/THREADS_PER_BLOCK;
  dim3 dimBlock(blockDimension, blockDimension);
  dim3 dimThread(THREADS_PER_BLOCK);

  // Exec add function on GPU
  mul<<<dimBlock, dimThread>>>(a_gpu, b_gpu, c_gpu, N, N);

  // Copy results to CPU copy of c
  cudaMemcpy(c_cpu, c_gpu, size, cudaMemcpyDeviceToHost);

    // Print
  print_matrix(N);

  // Cleanup
  cudaFree(a_gpu); cudaFree(b_gpu); cudaFree(c_gpu);

  return 0;
}
