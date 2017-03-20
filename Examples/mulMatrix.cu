#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <inttypes.h>
#include <iostream>
#include <ctime>

using namespace std;

const long N = 12800;
const int THREADS_PER_BLOCK = 32;

// CPU copies of a, b, c
float *a_cpu, *b_cpu, *c_cpu;

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, long N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}

void matrixMultiplication(float *A, float *B, float *C, int N){

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
  int blockDim = (N*N)/THREADS_PER_BLOCK;


    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksPerGrid(blockDim, blockDim);
        // if (N*N > 512){
        //     threadsPerBlock.x = 512;
        //     threadsPerBlock.y = 512;
        //     blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        //     blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
        // }

    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}

void random_elements (long N) {
  int i;
  int j;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      a_cpu[i * N + j] = 1.0f;
      b_cpu[i * N + j] = 1.0f;
    }
  }
}

void print_matrix(float *m, long N) {
  int i;
  int j;
  cout << "Matrix:" << endl;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      cout << m[i * N + j] << "-";
    }
    cout << endl;
  }
}

int main (void) {
  clock_t begin = clock();

  // GPU copies of a, b, c
  float *a_gpu, *b_gpu, *c_gpu;
  long size = N * N * sizeof(float);

  cudaMalloc((void **)&a_gpu, size);
  cudaMalloc((void **)&b_gpu, size);
  cudaMalloc((void **)&c_gpu, size);

  // Allocate GPU space for CPU copies of a, b, c
    a_cpu = (float *)malloc(size);
    b_cpu = (float *)malloc(size);
    c_cpu = (float *)malloc(size);

  // Set up random input variables
  random_elements(N);

  //print_matrix(c_cpu, N);

  // Copy inputs to device
  cudaMemcpy(a_gpu, a_cpu, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_gpu, b_cpu, size, cudaMemcpyHostToDevice);

  matrixMultiplication(a_gpu, b_gpu, c_gpu, N);

  cudaDeviceSynchronize();

  // Copy results to CPU copy of c
  cudaMemcpy(c_cpu, c_gpu, size, cudaMemcpyDeviceToHost);

    // Print
  //print_matrix(c_cpu, N);
  cout << c_cpu[0] << endl;

  // Cleanup
  cudaFree(a_gpu); cudaFree(b_gpu); cudaFree(c_gpu);
  delete a_cpu, b_cpu, c_cpu;
  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
   cout << elapsed_secs;
  return 0;
}
