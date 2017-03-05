#include <cstdint>
#include <cuda.h>
#include <iostream>

using namespace std;

const int N = 512;
int a_cpu[N][N], b_cpu[N][N], c_cpu[N][N];

__global__ void add(int *a, int *b, int *c, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int index = row * N + col;

  if (row < N && col < N){
    c[index] = a[index] + b[index];
  }
}

void random_ints (int N) {
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
int main(void) {
  // CPU copies of a, b, c
  int *a_gpu, *b_gpu, *c_gpu;
  int size = N * N * sizeof(int);

  // GPU copies of a, b, c
  cudaMalloc((void **)&a_gpu, size);
  cudaMalloc((void **)&b_gpu, size);
  cudaMalloc((void **)&c_gpu, size);

  // Setup random input variables
  random_ints(N);

  // Copy inputs to device
  cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyHostToDevice);

  int blockDimension = (N*N)/THREADS_PER_BLOCK;
  dim3 dimBlock(blockDimension, blockDimension);
  dim3 dimThread(THREADS_PER_BLOCK);

  // Exec add function on GPU
  add<<<dimBlock, dimThread>>>(a_gpu, b_gpu, c_gpu, N);

  // Copy results to CPU copy of c
  cudaMemcpy(c_cpu, c_gpu, size, cudaMemcpyDeviceToHost);

  // Print
  print_matrix(N);

  // Cleanup
  cudaFree(a_gpu); cudaFree(b_gpu); cudaFree(c_gpu);
  //delete[] a_cpu, b_cpu, c_cpu;

  return 0;
}
