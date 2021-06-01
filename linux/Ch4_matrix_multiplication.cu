// Copyright 2021 Bryan Baker

// Compile using:
// nvcc Ch4_matrix_multiplication.cu -o matmult

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

#define TILE_WIDTH 60

__global__ void BasicMatrixMulKernel(float *A, float *B, float *C, int I, int J, int K) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    
    float tmp = 0.0;

    if(row < I && col < J) {
      for(int i=0; i<K; i++) {
        tmp += A[row*K+i] * B[i*J+col];
      }
      C[row*J+col] = tmp;
    }
}

__global__ void matrixMulKernel(float *d_M, float *d_N, float *d_P, int Width) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  //Identify the row and column of the d_P element to work on
  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;

  float Pvalue = 0.0;
  // Loop over the d_M and d_N tiles required to compute d_P element
  for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph) {
    // Collaborative loading of d_M and d_N tiles into shared memory
    if ((Row < Width) && (ph*TILE_WIDTH+tx) < Width)
      Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
    else
      Mds[ty][tx] = 0.0;
    if ((ph*TILE_WIDTH+ty) < Width && Col < Width)
      Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];
    else
      Nds[ty][tx] = 0.0;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  if ((Row < Width) && (Col < Width)) d_P[Row*Width + Col] = Pvalue;
}

double wctime();
void zero_init(int rdim, int cdim, float *A);
void rand_init(int rdim, int cdim, float *A);
void matrix_mult(int rdim, int cdim, int kdim, float *A, float *B, float *C);
float error_calc(int rdim, int cdim, float *A, float *B);
void print_sample(int rdim, int cdim, float *A, int rsize, int csize);

int main(int argc, char *argv[]) {
    int idim = 1000;
    int jdim = 1000;
    int kdim = 1000;
    double t1;
    float nops, err;
    float *A, *B, *C, *actualC, *Ag, *Bg, *Cg;
    A = (float*) malloc(idim*kdim*sizeof(float));
    B = (float*) malloc(kdim*jdim*sizeof(float));
    C = (float*) malloc(idim*jdim*sizeof(float));
    actualC = (float*) malloc(idim*jdim*sizeof(float));

    zero_init(idim, jdim, C);
    zero_init(idim, jdim, actualC);
    rand_init(idim, kdim, A);
    rand_init(kdim, jdim, B);

    //printf("A matrix sample: \n");
    //print_sample(idim, kdim, A, 2, 10);
    //printf("B matrix sample: \n");
    //print_sample(kdim, jdim, B, 2, 10);

    // This is the standard matrix multiplication - do not adjust
    t1 = wctime();
    matrix_mult(idim, jdim, kdim, A, B, actualC);
    t1 = wctime() - t1;

    printf("CPU:\n");
    printf("Finished in %lf seconds.\n", t1);
    t1 *= (1.E+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n", nops/t1);
    printf("\n");

    //printf("ActualC matrix sample: \n");
    //print_sample(idim, jdim, actualC, 2, 10);

    cudaMalloc(&Ag, idim*kdim*sizeof(float));
    cudaMalloc(&Bg, kdim*jdim*sizeof(float));
    cudaMalloc(&Cg, idim*jdim*sizeof(float));
    
    t1 = wctime();
    cudaMemcpy(Ag, A, idim*kdim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bg, B, kdim*jdim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Cg, C, idim*jdim*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(jdim/32.0), ceil(idim/32.0), 1);
    dim3 dimBlock(32, 32, 1);
    BasicMatrixMulKernel<<<dimGrid, dimBlock>>>(Ag, Bg, Cg, idim, jdim, kdim);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error) {
      printf("CUDA error: %s \n", cudaGetErrorString(error));
      exit(1);
    }
    cudaMemcpy(C, Cg, idim*jdim*sizeof(float), cudaMemcpyDeviceToHost);
    t1 = wctime() - t1;
    
    printf("Basic Cuda with data transfer:\n");
    printf("Finished in %lf seconds.\n", t1);
    t1 *= (1.E+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n", nops/t1);
    err = error_calc(idim, jdim, actualC, C);
    printf("Error: %f\n", err/((float)idim*jdim));
    printf("\n");

    zero_init(idim, jdim, C);
    cudaMemcpy(Cg, C, idim*jdim*sizeof(float), cudaMemcpyHostToDevice);
    t1 = wctime();
    BasicMatrixMulKernel<<<dimGrid, dimBlock>>>(Ag, Bg, Cg, idim, jdim, kdim);
    cudaDeviceSynchronize();
    t1 = wctime() - t1;
    if (error) {
      printf("CUDA error: %s \n", cudaGetErrorString(error));
      exit(1);
    }
    cudaMemcpy(C, Cg, idim*jdim*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Basic Cuda with out data transfer:\n");
    printf("Finished in %lf seconds.\n", t1);
    t1 *= (1.E+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n", nops/t1);
    err = error_calc(idim, jdim, actualC, C);
    printf("Error: %f\n", err/((float)idim*jdim));
    printf("\n");

    zero_init(idim, jdim, C);
    t1 = wctime();
    cudaMemcpy(Ag, A, idim*kdim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bg, B, kdim*jdim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Cg, C, idim*jdim*sizeof(float), cudaMemcpyHostToDevice);
    matrixMulKernel<<<dimGrid, dimBlock>>>(Ag, Bg, Cg, idim);
    cudaDeviceSynchronize();
    if (error) {
      printf("CUDA error: %s \n", cudaGetErrorString(error));
      exit(1);
    }
    cudaMemcpy(C, Cg, idim*jdim*sizeof(float), cudaMemcpyDeviceToHost);
    t1 = wctime() - t1;

    printf("Optimized Cuda with data transfer:\n");
    printf("Finished in %lf seconds.\n", t1);
    t1 *= (1.E+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n", nops/t1);
    err = error_calc(idim, jdim, actualC, C);
    printf("Error: %f\n", err/((float)idim*jdim));
    printf("\n");

    zero_init(idim, jdim, C);
    cudaMemcpy(Cg, C, idim*jdim*sizeof(float), cudaMemcpyHostToDevice);
    t1 = wctime();
    matrixMulKernel<<<dimGrid, dimBlock>>>(Ag, Bg, Cg, idim);
    cudaDeviceSynchronize();
    t1 = wctime() - t1;
    if (error) {
      printf("CUDA error: %s \n", cudaGetErrorString(error));
      exit(1);
    }
    cudaMemcpy(C, Cg, idim*jdim*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Optimized Cuda with out data transfer:\n");
    printf("Finished in %lf seconds.\n", t1);
    t1 *= (1.E+09);
    nops = (float) 2 * idim * kdim * jdim;
    printf("Performance = %f GFLOPs\n", nops/t1);
    err = error_calc(idim, jdim, actualC, C);
    printf("Error: %f\n", err/((float)idim*jdim));
    printf("\n");

    cudaFree(Ag);
    cudaFree(Bg);
    cudaFree(Cg);
    free(A);
    free(B);
    free(C);
    free(actualC);
    return(0);
}

double wctime() {
  // calculate wall time.
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec + 1E-6 * tv.tv_usec);
}

void zero_init(int rdim, int cdim, float *A) {
    for(int i = 0; i < rdim; i++) {
        for(int j = 0; j < cdim; j++) {
            A[i*cdim+j] = 0.0;
        }
    }
}

void rand_init(int rdim, int cdim, float *A) {
    for(int i = 0; i < rdim; i++) {
        for(int j = 0; j < cdim; j++) {
            A[i*cdim+j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

void matrix_mult(int rdim, int cdim, int kdim, float *A, float *B, float *C) {
    int i, j, k;
    for(i = 0; i < rdim; i++)
      for(k = 0; k < kdim; k++)
        for(j = 0; j < cdim; j++)
          C[i*cdim+j] += A[i*kdim+k] * B[k*cdim+j];
}

float error_calc(int rdim, int cdim, float *A, float *B) {
    int i, j;
    float err = 0.0, t = 0.0;
    for(i = 0; i < rdim; i++) {
      for(j = 0; j < cdim; j++) {
        err += ((A[i*cdim+j] - B[i*cdim+j]) * (A[i*cdim+j] - B[i*cdim+j]));
        t += (A[i*cdim+j] * A[i*cdim+j]);
      }
    }
    return(sqrt(err/t));
}

void print_sample(int rdim, int cdim, float *A, int rsize, int csize) {
    int i, j;
    for(i = 0; i < rsize; i++) {
      printf("[ ");
      for(j = 0; j < csize; j++) {
        printf("%f ", A[i*cdim+j]);
      }
      printf("]\n");
    }
}

