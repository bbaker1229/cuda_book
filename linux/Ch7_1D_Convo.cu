// Copyright 2021 Bryan Baker

// Compile using:
// nvcc Ch7_1D_Convo.cu -o OneDConvo

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_MASK_WIDTH 10
#define TILE_SIZE 64

__constant__ float d_B[MAX_MASK_WIDTH];

__global__ void convolution_1D_basic_kernel(float *N, float *P, int Mask_Width, int Width) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  float Pvalue = 0;
  int N_start_point = i - (Mask_Width/2);
  for (int j = 0; j < Mask_Width; j++) {
    if (N_start_point + j >= 0 && N_start_point + j < Width) {
      Pvalue += N[N_start_point + j]*d_B[j];
    }
  }
  P[i] = Pvalue;
}

__global__ void convolution_1D_tiled_kernel(float *N, float *P, int Mask_Width, int Width) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];

  int n = Mask_Width/2;

  int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
  if (threadIdx.x >= blockDim.x - n) {
    N_ds[threadIdx.x - (blockDim.x - n)] =
      (halo_index_left < 0) ? 0 : N[halo_index_left];
  }
  
  N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x];

  int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
  if (threadIdx.x < n) {
    N_ds[n + blockDim.x + threadIdx.x] =
      (halo_index_right >= Width) ? 0 : N[halo_index_right];
  }

  __syncthreads();

  float Pvalue = 0;
  for (int j = 0; j < Mask_Width; j++) {
    Pvalue += N_ds[threadIdx.x + j]*d_B[j];
  }
  P[i] = Pvalue;
}

__global__ void convolution_1D_tiled_caching_kernel(float *N, float *P, int Mask_Width, int Width) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ float N_ds[TILE_SIZE];

  N_ds[threadIdx.x] = N[i];

  __syncthreads();

  int This_tile_start_point = blockIdx.x * blockDim.x;
  int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
  int N_start_point = i - (Mask_Width/2);
  float Pvalue = 0;
  for (int j = 0; j < Mask_Width; j++) {
    int N_index = N_start_point + j;
    if (N_index >= 0 && N_index < Width) {
      if ((N_index >= This_tile_start_point)
        && (N_index < Next_tile_start_point)) {
          Pvalue += N_ds[threadIdx.x+j-(Mask_Width/2)]*d_B[j];
        } else {
          Pvalue += N[N_index] * d_B[j];
        }
    }
  }
  P[i] = Pvalue;
}

void convolution_1D(float *N, float *M, float *P, int Mask_Width, int Width) {
  float Pvalue = 0;
  int N_start_point = 0;
  for (int i = 0; i < Width; i++) {
    N_start_point = i - (Mask_Width/2);
    for (int j = 0; j < Mask_Width; j++) {
      if (N_start_point + j >= 0 && N_start_point + j < Width) {
        Pvalue += N[N_start_point + j]*M[j];
      }
    }
    P[i] = Pvalue;
  }
}

double wctime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

int main() {
  float *A, *C, *d_A, *d_C;
  int i, N=10000000, M=5;
  float B[5] = {3, 4, 5, 4, 3};
  double t1;
  float nops;

  A = (float*)malloc(N*sizeof(float));  // Input vector
  C = (float*)malloc(N*sizeof(float));  // Output vector
  cudaMalloc((void**)&d_A, N*sizeof(float));  // Input vector
  // cudaMalloc((void**)&d_B, M*sizeof(float));  // Mask vector
  cudaMemcpyToSymbol(d_B, B, M*sizeof(float));  // Mask vector
  cudaMalloc((void**)&d_C, N*sizeof(float));  // Output vector

  for(i=0; i<N; i++) {
    A[i] = (float) rand() / (float) rand();
  }
  for(i=0; i<N; i++)
    C[i] = 0.0;

// CPU Only
  t1 = wctime();
  convolution_1D(A, B, C, M, N);
  t1 = wctime() - t1;

  /*printf("Vector A sample: ");
  for(i=0; i<10; i++)
    printf("%0.3f ", A[i]);
  printf("\n");
  printf("Vector B sample: ");
  for(i=0; i<5; i++)
    printf("%0.3f ", B[i]);
  printf("\n");
  printf("Vector C sample: ");
  for(i=0; i<10; i++)
    printf("%0.3f ", C[i]);
  printf("\n");*/

  printf("CPU:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N * M;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

// Basic Kernel
  for(i=0; i<N; i++)
    C[i] = 0.0;

  t1 = wctime();
  cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_B, B, M*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  convolution_1D_basic_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, M, N);

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
  t1 = wctime() - t1;

  printf("Cuda basic with data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N * M;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

  for(i=0; i<N; i++)
    C[i] = 0.0;

  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  t1 = wctime();
  convolution_1D_basic_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, M, N);
  t1 = wctime() - t1;

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Cuda basic without data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N * M;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

// Basic tiled Kernel
  for(i=0; i<N; i++)
    C[i] = 0.0;

  t1 = wctime();
  cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_B, B, M*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  convolution_1D_tiled_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, M, N);

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
  t1 = wctime() - t1;

  printf("Cuda tiled with data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N * M;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

  for(i=0; i<N; i++)
    C[i] = 0.0;

  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  t1 = wctime();
  convolution_1D_tiled_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, M, N);
  t1 = wctime() - t1;

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Cuda tiled without data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N * M;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

// General tiled Kernel (use L2 cache)
  for(i=0; i<N; i++)
    C[i] = 0.0;

  t1 = wctime();
  cudaMemcpy(d_A, A, N*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_B, B, M*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  convolution_1D_tiled_caching_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, M, N);

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
  t1 = wctime() - t1;

  printf("Cuda caching-tiled with data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N * M;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

  for(i=0; i<N; i++)
    C[i] = 0.0;

  cudaMemcpy(d_C, C, N*sizeof(float), cudaMemcpyHostToDevice);

  t1 = wctime();
  convolution_1D_tiled_caching_kernel<<<ceil(N/((float) TILE_SIZE)), TILE_SIZE>>>(d_A, d_C, M, N);
  t1 = wctime() - t1;

  cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Cuda caching-tiled without data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) N * M;
  printf("Performance = %f GFLOPs\n", nops/t1);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(C);
}
