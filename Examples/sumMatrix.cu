#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <inttypes.h>
#include <iostream>

using namespace std;

__global__ void add(int *a, int *b, int *c, int N) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  c[index] = a[index] + b[index];

}

void random_ints(int *a, int N) {
  int i,j;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      a[i * N + j] = 1;
    }
  }
}

void print_vector(int *a, int N) {
  int i;
  int j;
  cout << "Matrix:" << endl;
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      cout << a[i * N + j] << " ";
    }
    cout << endl;
  }
}

int main(int argc, char* argv[]) {

    // CPU copies of a, b, c
    int N;
    const int THREADS_PER_BLOCK = 32;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    if (argc != 2) {
      fprintf(stderr, "usage: %s <size>\n", argv[0]);
      exit(0);
    }

    N = strtol(argv[1], NULL, 10);
    size_t size = sizeof(int) * N * N;

    // GPU copies of a, b, c
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Allocate GPU space for CPU copies of a, b, c
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // Setup random input variables
    random_ints(a, N);
    random_ints(b, N);

    //print_vector(a, N);
    //print_vector(b, N);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Exec add function on GPU
    add<<<(N*N)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    // Copy results to CPU copy of c
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    print_vector(c, N);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
