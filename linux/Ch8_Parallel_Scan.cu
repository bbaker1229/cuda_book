// Copyright 2021 Bryan Baker

// Compile using:
// nvcc Ch8_Parallel_Scan.cu -o ParScan

// Need to add last section to add each SECTION together.

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define SECTION_SIZE 32
#define TILE_SIZE 32

__global__ void Kogge_Stone_scan_kernel(float *X, float *Y, int InputSize) {
  __shared__ float XY[SECTION_SIZE];
  __shared__ float S[InputSize / SECTION_SIZE];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < InputSize) {
    XY[threadIdx.x] = X[i];
  }

  // the code below performs iterative scan on XY
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (threadIdx.x >= stride) XY[threadIdx.x] += XY[threadIdx.x-stride];
  }

  Y[i] = XY[threadIdx.x];

}

__global__ void Brent_Kung_scan_kernel(float *X, float *Y, int InputSize) {
  __shared__ float XY[SECTION_SIZE];
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < InputSize) XY[threadIdx.x] = X[i];
  if (i+blockDim.x < InputSize) XY[threadIdx.x+blockDim.x] = X[i + blockDim.x];

  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < SECTION_SIZE) {
      XY[index] += XY[index - stride];
    }
  }

  for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < SECTION_SIZE) {
      XY[index + stride] += XY[index];
    }
  }

  __syncthreads();
  if (i < InputSize) Y[i] = XY[threadIdx.x];
  if (i + blockDim.x < InputSize) Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
}

void sequential_scan(float *x, float *y, int Max_i) {
  float accumulator = x[0];
  y[0] = accumulator;
  for(int i = 1; i < Max_i; i++) {
    accumulator += x[i];
    y[i] = accumulator;
  }
}

double wctime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

int main() {
  float *A, *C, *d_A, *d_C;
  int i, N=10000000;
  double t1;
  float nops;

  A = (float*)malloc(N*sizeof(float));  // Input vector
  C = (float*)malloc(N*sizeof(float));  // Output vector
  cudaMalloc((void**)&d_A, N*sizeof(float));  // Input vector
  cudaMalloc((void**)&d_C, N*sizeof(float));  // Output vector

  for(i=0; i<N; i++) {
    A[i] = (float) rand() / (float) rand();
  }
  for(i=0; i<N; i++)
    C[i] = 0.0;

// CPU Only
  t1 = wctime();
  sequential_scan(A, C, N);
  t1 = wctime() - t1;

  printf("Vector A sample: ");
  for(i=30; i<40; i++)
    printf("%0.3f ", A[i]);
  printf("\n");
  printf("Vector C sample: ");
  for(i=30; i<40; i++)
    printf("%0.3f ", C[i]);
  printf("\n");

  printf("CPU:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

// Kogge-Stone Kernel
  for(i=0; i<N; i++)
    C[i] = 0.0;

  t1 = wctime();
  cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  Kogge_Stone_scan_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, N);

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
  t1 = wctime() - t1;

  printf("Vector A sample: ");
  for(i=30; i<40; i++)
    printf("%0.3f ", A[i]);
  printf("\n");
  printf("Vector C sample: ");
  for(i=30; i<40; i++)
    printf("%0.3f ", C[i]);
  printf("\n");

  printf("Kogge-Stone with data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

  for(i=0; i<N; i++)
    C[i] = 0.0;

  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  t1 = wctime();
  Kogge_Stone_scan_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, N);
  t1 = wctime() - t1;

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Kogge-Stone without data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

  // Brent-Kung Kernel
  for(i=0; i<N; i++)
    C[i] = 0.0;

  t1 = wctime();
  cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  Brent_Kung_scan_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, N);

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
  t1 = wctime() - t1;

  printf("Vector A sample: ");
  for(i=30; i<40; i++)
    printf("%0.3f ", A[i]);
  printf("\n");
  printf("Vector C sample: ");
  for(i=30; i<40; i++)
    printf("%0.3f ", C[i]);
  printf("\n");

  printf("Brent-Kung with data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

  for(i=0; i<N; i++)
    C[i] = 0.0;

  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  t1 = wctime();
  Brent_Kung_scan_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, N);
  t1 = wctime() - t1;

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Brent-Kung without data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

  cudaFree(d_A);
  cudaFree(d_C);
  free(A);
  free(C);
}
