#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define BLOCK_SIZE 16

// CUDA kernel for matrix multiplication (assumes square matrices of size nxn)
__global__ void matrixMulKernel(float *A, float *B, float *C, int n){
    // Calculate the row and column indices of the element this thread will compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < n && col < n){
        for (int i=0; i<n; i++){
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Function to set up and execute matrix multiplication on the GPU
void matrixMul(float *A, float *B, float *C, int n){
    // Start an NVTX range for profiling the matrix multiplication function
    nvtxRangePush("matrix Multiplication");

    // Device pointers for matrices A, B, and C
    float *d_A, *d_B, *d_C; 
    // Total size of each matrix in bytes
    int size = n * n * sizeof(float);

    nvtxRangePush("Memory Allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    nvtxRangePop();

    nvtxRangePush("Memory Copy H2D");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    nvtxRangePop();

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel and synchronize the device after execution
    nvtxRangePush("Kernel Execution");
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    // End of kernel execution range
    nvtxRangePop();

    // Copy the result matrix C from the device (GPU) back to the host (CPU)
    nvtxRangePush("Memory Copy D2H");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    nvtxRangePop();

    nvtxRangePush("Memory Deallocation");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop();

    nvtxRangePop();
}


int main() {
    const int N = 1024;
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];

    // Initialize matrices A and B here...

    matrixMul(A, B, C, N);

    // Use result in C...

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}