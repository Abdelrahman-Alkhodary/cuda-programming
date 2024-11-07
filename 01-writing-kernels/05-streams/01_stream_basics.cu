#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

// Error checking function
// Takes in a CUDA function's return value, and if there's an error, prints an error message 
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

// kernel function to add two vectors element-wise
// Each thread calculates one element of the result vector
__global__ void vectorAdd(float *a, float *b, float *c, int numElements){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements){
        c[i] = a[i] + b[i];
    }
}

int main(){
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    cudaStream_t stream1, stream2; // create two streams for asynchronous execution

    // Allocate memory for host vectors
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Allocate memory for device vectors
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));

    // Initialize host vectors
    for (int i=0; i<numElements; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // create two CUDA streams for handling asynchronous execution
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    // Copy host vectors to device
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2));

    // configure the kernel launch parameters   
    int threadPerBlock = 256; 
    int blocksPerGrid = (numElements + threadPerBlock - 1) / threadPerBlock;

    // Launch the kernel on `stream1`, with each thread handling one element of the vectors
    vectorAdd<<<blocksPerGrid, threadPerBlock, 0, stream1>>>(d_a, d_b, d_c, numElements);

    // Copy result vector back to host
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_c, d_c, size, cudaMemcpyDeviceToHost, stream1));

    // Wait for the stream operations to complete
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

     // Verify that the result on the host matches expected values
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) { // Check if the difference is significant
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE); // Exit if verification fails
        }
    }

    printf("Test PASSED\n"); // If verification succeeds, print "Test PASSED"

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    // Destroy CUDA streams
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}