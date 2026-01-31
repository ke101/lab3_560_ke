#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h> // Required for cuBLAS

int main(int argc, char **argv) {
    // 1. Setup Matrix Size
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    printf("Matrix Size: %d x %d\n", N, N);

    size_t bytes = N * N * sizeof(float);

    // 2. Allocate Host Memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize matrices
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
        h_C[i] = 0.0f;
    }

    // 3. Allocate Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 4. Copy Data Host -> Device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 5. Setup cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Scala factors for: C = alpha * (A * B) + beta * C
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    printf("Running cuBLAS Sgemm...\n");
    clock_t start = clock();

    // 6. Perform Matrix Multiplication
    // Note: cuBLAS is Column-Major. To use Row-Major C-arrays, we calculate C = B * A
    // This effectively performs Transpose(A * B) which aligns with linear memory.
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                N, N, N, 
                &alpha, 
                d_B, N, // Leading dimension of B
                d_A, N, // Leading dimension of A
                &beta, 
                d_C, N); // Leading dimension of C

    // Wait for GPU to finish (cuBLAS is asynchronous)
    cudaDeviceSynchronize();

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("GPU execution time: %f seconds\n", elapsed);

    // 7. Cleanup
    cublasDestroy(handle);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Verify first element
    printf("Result sample (C[0]): %f\n", h_C[0]);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}