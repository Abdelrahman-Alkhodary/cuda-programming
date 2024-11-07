#include <cuda_runtime.h> // Includes CUDA runtime functions
#include <stdio.h>        // Standard I/O library
#include <iostream>       // For C++ I/O

// Macro to simplify error-checking for CUDA calls
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// Function to check CUDA errors and print details if there is a failure
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) { // Check if error code is not cudaSuccess
        // Print detailed error information and exit program
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", 
                file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

// First kernel: Multiplies each element in `data` array by 2
__global__ void kernel1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index
    if (idx < n) { // Ensure within array bounds
        data[idx] *= 2.0f; // Double the value at index `idx`
    }
}

// Second kernel: Adds 1 to each element in `data` array
__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index
    if (idx < n) { // Ensure within array bounds
        data[idx] += 1.0f; // Increment the value at index `idx`
    }
}

// Callback function for CUDA stream
// This function will be called when all previous operations in the stream complete
void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Stream callback: Operation completed\n");
}

int main(void) {
    const int N = 1000000;              // Total number of elements in the array
    size_t size = N * sizeof(float);    // Total size in bytes for `N` float elements
    float *h_data, *d_data;             // Host and device pointers for the data array
    cudaStream_t stream1, stream2;      // CUDA streams for asynchronous operations
    cudaEvent_t event;                  // CUDA event for synchronization

    // Allocate pinned host memory (pinned memory is faster for GPU transfers)
    CHECK_CUDA_ERROR(cudaMallocHost(&h_data, size));  
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));      // Allocate device memory

    // Initialize host data with consecutive values
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Get priority range for CUDA streams (used to assign priorities to streams)
    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

    // Create two streams with different priorities for concurrent task management
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority)); 
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));

    // Create an event that can be used for synchronization between streams
    CHECK_CUDA_ERROR(cudaEventCreate(&event));

    // Asynchronously copy data from host to device using stream1
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1));

    // Launch `kernel1` in `stream1` to double each element in `d_data`
    kernel1<<<(N + 255) / 256, 256, 0, stream1>>>(d_data, N);

    // Record an event in `stream1` after `kernel1` completes
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream1));

    // Make `stream2` wait until the `event` is recorded in `stream1`
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2, event, 0));

    // Launch `kernel2` in `stream2` to add 1 to each element in `d_data`
    kernel2<<<(N + 255) / 256, 256, 0, stream2>>>(d_data, N);

    // Add a callback function to `stream2` to notify when all operations in `stream2` are complete
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0));

    // Asynchronously copy modified data from device back to host using `stream2`
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2));

    // Synchronize both streams to make sure all operations are complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // Verify results to ensure the computations are correct
    for (int i = 0; i < N; ++i) {
        float expected = (static_cast<float>(i) * 2.0f) + 1.0f;
        if (fabs(h_data[i] - expected) > 1e-5) { // Check for small differences due to floating-point precision
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free memory and destroy streams and events
    CHECK_CUDA_ERROR(cudaFreeHost(h_data));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    CHECK_CUDA_ERROR(cudaEventDestroy(event));

    return 0;
}
