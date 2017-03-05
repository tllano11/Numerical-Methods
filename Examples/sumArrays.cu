#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <inttypes.h>

__global__ void add(int64_t *a, int64_t *b, int64_t *c) {

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];

}

void random_ints(int64_t *a, int N) {

  int i;
  for (i = 0; i < N; ++i) {
    a[i] = 1;
  }

}

void print_vector(int64_t *a, int N) {

  int i;
  printf("Vector:\n");
  for (i = 0; i < N; ++i) {
    printf("%" PRId64 "\n", a[i]);
  }

}

#define N 10000000
#define THREADS_PER_BLOCK 1000
int main(void) {

    // CPU copies of a, b, c
    int64_t *a, *b, *c;
    int64_t *d_a, *d_b, *d_c;
    int size = sizeof(int64_t) * N;

    // GPU copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate GPU space for CPU copies of a, b, c
    a = (int64_t *)malloc(size);
    b = (int64_t *)malloc(size);
    c = (int64_t *)malloc(size);

    // Setup random input variables
    random_ints(a, N);
    random_ints(b, N);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Exec add function on GPU
    add<<<N/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_a, d_b, d_c);

    // Copy results to CPU copy of c
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
