#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  if(i<n) C[i] = A[i] + B[i];
}

void vecAdd(float *h_A, float *h_B, float *h_C, int n) {
  for(int i = 0; i < n; i++) h_C[i] = h_A[i] + h_B[i];
}

double wctime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

int main() {
  float *A, *B, *C, *d_A, *d_B, *d_C;
  int i, N=10000000;
  double t1;
  float nops;

  A = (float*)malloc(N*sizeof(float));
  B = (float*)malloc(N*sizeof(float));
  C = (float*)malloc(N*sizeof(float));
  cudaMalloc((void**)&d_A, N*sizeof(float));
  cudaMalloc((void**)&d_B, N*sizeof(float));
  cudaMalloc((void**)&d_C, N*sizeof(float));

  for(i=0; i<N; i++) {
    A[i] = (float) rand() / (float) rand();
    B[i] = (float) rand() / (float) rand();
  }
  for(i=0; i<N; i++)
    C[i] = 0.0;

  t1 = wctime();
  vecAdd(A, B, C, N);
  t1 = wctime() - t1;

  printf("Vector A sample: ");
  for(i=0; i<10; i++)
    printf("%0.3f ", A[i]);
  printf("\n");
  printf("Vector B sample: ");
  for(i=0; i<10; i++)
    printf("%0.3f ", B[i]);
  printf("\n");
  printf("Vector C sample: ");
  for(i=0; i<10; i++)
    printf("%0.3f ", C[i]);
  printf("\n");

  printf("CPU sequential program:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N;
  printf("Performance = %f GFLOPs\n", nops/t1);

  for(i=0; i<N; i++)
    C[i] = 0.0;

  t1 = wctime();
  cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  vecAddKernel<<<ceil(N/1024.0), 1024>>>(d_A, d_B, d_C, N);

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
  t1 = wctime() - t1;

  printf("Cuda with data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N;
  printf("Performance = %f GFLOPs\n", nops/t1);

  for(i=0; i<N; i++)
    C[i] = 0.0;

  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  t1 = wctime();
  vecAddKernel<<<ceil(N/1024.0), 1024>>>(d_A, d_B, d_C, N);
  t1 = wctime() - t1;

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Cuda without data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N;
  printf("Performance = %f GFLOPs\n", nops/t1);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);
}
