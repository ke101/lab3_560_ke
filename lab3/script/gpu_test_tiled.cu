#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>



#define TILE_WIDTH 16
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) { 
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH]; 
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH]; 

    int bx = blockIdx.x; int by = blockIdx.y; 
    int tx = threadIdx.x; int ty = threadIdx.y; 

    int Row = by * TILE_WIDTH + ty; 
    int Col = bx * TILE_WIDTH + tx; 
    
    float Pvalue = 0.0; 

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) { 
        

        if (Row < N && (m * TILE_WIDTH + tx) < N) 
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx]; 
        else 
            ds_A[ty][tx] = 0.0f; 

        if (Col < N && (m * TILE_WIDTH + ty) < N) 
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col]; 
        else 
            ds_B[ty][tx] = 0.0f; 
        
        __syncthreads(); 

        for (int k = 0; k < TILE_WIDTH; ++k) 
            Pvalue += ds_A[ty][k] * ds_B[k][tx]; 
        __syncthreads(); 
    } 
    
    if (Row < N && Col < N) 
        C[Row * N + Col] = Pvalue;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    printf("Matrix Size: %d x %d\n", N, N);

    size_t bytes = N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    
    // Grid size must cover the whole matrix N
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, 
                   (N + TILE_WIDTH - 1) / TILE_WIDTH);

    printf("Kernel Config: Blocks(%d, %d), Threads(%d, %d)\n", 
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    clock_t start = clock();

    matrixMultiplyTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    // Output formatted specifically for your Python script
    printf("GPU execution time: %f seconds\n", elapsed);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Quick Verification (Check index 0)
    // Note: CPU verification is skipped here for speed, but h_C[0] should be roughly valid
    printf("Result sample (C[0]): %f\n", h_C[0]);
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}