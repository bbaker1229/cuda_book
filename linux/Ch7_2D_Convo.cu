// Copyright 2021 Bryan Baker

// Compile using:
// nvcc Ch7_2D_Convo.cu -o TwoDConvo

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_MASK_WIDTH 10
#define MAX_MASK_HEIGHT 10
#define TILE_SIZE 64
#define O_TILE_WIDTH 10

__constant__ float d_B[MAX_MASK_WIDTH * MAX_MASK_HEIGHT];

__global__ void convolution_2D_tiled_kernel(float *P, float *data, int height, int width, int pitch, int channels, int Mask_Width, const float *M) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row_o = blockIdx.y*O_TILE_WIDTH + ty;
  int col_o = blockIdx.x*O_TILE_WIDTH + tx;

  int row_i = row_o - Mask_Width/2;
  int col_i = col_o - Mask_Width/2;

  __shared__ float N_ds[TILE_SIZE+MAX_MASK_WIDTH-1][TILE_SIZE+MAX_MASK_HEIGHT-1];
  if((row_i >= 0) && (row_i < height) && 
     (col_i >= 0) && (col_i < width)) {
      N_ds[ty][tx] = data[row_i * pitch + col_i];
    } else {
      N_ds[ty][tx] = 0.0f;
    }
  
  float output = 0.0f;
  if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
    for(int i = 0; i < Mask_Width; i++) {
      for(int j = 0; j < Mask_Width; j++) {
        output += M[i*Mask_Width + j] * N_ds[i+ty][j+tx];
      }
    }
    if(row_o < height && col_o < width) {
      P[row_o*width + col_o] = output;
    }
  }
}

void convolution_2D(float *N, float *M, float *P, int Mask_Width, int Width, int Height) {
  float Pvalue = 0;
  int N_x_start_point = 0;
  int N_y_start_point = 0;
  for (int i = 0; i < Width; i++) {
    N_x_start_point = i - (Mask_Width/2);
    for (int k = 0; k < Height; k++) {
      N_y_start_point = k - (Mask_Width/2);
      for (int j = 0; j < Mask_Width; j++) {
        for (int z = 0; z < Mask_Width; z++) {
          if ((N_x_start_point + j >= 0) && (N_x_start_point + j < Width) && (N_y_start_point + z >= 0) && (N_y_start_point + z < Height)) {
            Pvalue += N[(N_y_start_point + z)*Mask_Width + (N_x_start_point + j)]*M[z*Mask_Width + j];
          }
        }
      }
      P[k*Width + i] = Pvalue;
    }
  }
}

double wctime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

void rand_init(int rdim, int cdim, float *A) {
    for(int i = 0; i < rdim; i++) {
        for(int j = 0; j < cdim; j++) {
            A[i*cdim+j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

void zero_init(int rdim, int cdim, float *A) {
    for(int i = 0; i < rdim; i++) {
        for(int j = 0; j < cdim; j++) {
            A[i*cdim+j] = 0.0;
        }
    }
}

int main() {
  float *A, *C, *d_A, *d_C;
  int M=5;
  float B[M*M] = {3, 4, 5, 4, 3, 3, 4, 5, 4, 3, 3, 4, 5, 4, 3, 3, 4, 5, 4, 3, 3, 4, 5, 4, 3};
  int idim = 1000;
  int jdim = 1000;
  double t1;
  float nops;
  A = (float*) malloc(idim*jdim*sizeof(float));
  C = (float*) malloc(idim*jdim*sizeof(float));

  cudaMalloc((void**)&d_A, idim*jdim*sizeof(float));  // Input vector
  cudaMalloc((void**)&d_B, M*M*sizeof(float));  // Mask vector
  cudaMalloc((void**)&d_C, idim*jdim*sizeof(float));  // Output vector

  zero_init(idim, jdim, C);
  rand_init(idim, jdim, A);

// CPU Only
  t1 = wctime();
  convolution_2D(A, B, C, M, idim, jdim);
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
  nops = (float) idim * jdim * M * M;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

// Basic Kernel
  zero_init(idim, jdim, C);

  t1 = wctime();
  cudaMemcpy(d_A, A, idim*jdim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, M*M*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, idim*jdim*sizeof(float), cudaMemcpyHostToDevice);

  convolution_2D_tiled_kernel<<<ceil(idim*jdim/((float) TILE_SIZE)), TILE_SIZE>>>(d_C, d_A, idim, jdim, jdim, 1, M, d_B);

  cudaMemcpy(C, d_C, idim*jdim*sizeof(float), cudaMemcpyDeviceToHost);
  t1 = wctime() - t1;

  printf("Cuda 2D with data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) idim * jdim * M * M;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

  zero_init(idim, jdim, C);

  cudaMemcpy(d_C, C, idim*jdim*sizeof(float), cudaMemcpyHostToDevice);

  t1 = wctime();
  convolution_2D_tiled_kernel<<<ceil(idim*jdim/((float) TILE_SIZE)), TILE_SIZE>>>(d_C, d_A, idim, jdim, jdim, 1, M, d_B);
  t1 = wctime() - t1;

  cudaMemcpy(C, d_C, idim*jdim*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Cuda 2D without data transfer:\n");
  printf("Finished in %lf seconds.\n", t1);
  t1 *= (1.E+09);
  nops = (float) idim * jdim * M * M;
  printf("Performance = %f GFLOPs\n", nops/t1);
  printf("\n");

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(C);
}
