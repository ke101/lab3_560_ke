#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void m_m_muplty_GPU(float *A, float *B, float *C, int N) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < N && col < N) { 
        float sum = 0.0f; 
        for (int k = 0; k < N; k++) { 
            sum += A[row * N + k] * B[k * N + col]; 
        } 
        C[row * N + col] = sum; 
    } 
} 

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024; 
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes); 

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

    dim3 threadsPerBlock(16, 16);
    
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Launching Kernel...\n");
    clock_t start = clock();

    m_m_muplty<<<nB, tB>>>(d_A, d_B, d_C, N);


    cudaDeviceSynchronize();

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("GPU execution time: %f seconds\n", elapsed);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
