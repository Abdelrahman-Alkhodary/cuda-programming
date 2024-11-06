#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256  // Number of rows in A and C
#define K 512   // Number of columns in A and rows in B
#define N 256  // Number of columns in B and C
#define BLOCK_SIZE 32

// CPU matrix multiplication
void matmul_cpu(float *a, float *b, float *c, int m, int k, int n){
    for (int i=0; i< m; i++){
        for (int j=0; j< n; j++){
            float sum = 0;
            for (int l=0; l < k; l++){
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmul_gpu(float *a, float *b, float *c, int m, int k, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n){
        float sum = 0;
        for (int i=0; i<k; i++){
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int row, int col){
    for (int i=0; i < row * col; i++){
        mat[i] = (float)rand() / RAND_MAX;  
    }
}

// Function to measure execution time   
double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}


int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    // Allocate host memory
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c_cpu = (float *)malloc(size_c);
    h_c_gpu = (float *)malloc(size_c);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    // Allocate device memory
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Warm-up runs
    printf("Warm-up runs\n");
    for (int i=0; i<3; i++){
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
        cudaDeviceSynchronize();
        // cudaDeviceSynchronize: makes sure all the kernel for one problem are caught up so you can safely begin the next. 
        // Called from your int main() {} or another non__global__ function.
    }

    // Measure execution time of CPU
    printf("Benchmarking CPU implementation\n");
    double cpu_total_time = 0.0;
    for (int i=0; i<20; i++){
        double start_time = get_time();
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Measure execution time of GPU
    printf("Benchmarking GPU implementation\n");
    double gpu_total_time = 0.0;
    for (int i=0; i<20; i++){
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}