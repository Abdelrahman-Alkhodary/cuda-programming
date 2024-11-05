#include <stdio.h>

// Define a kernal function to identify and print thread and block information
__global__ void whoami(void){
    // Calculate a unique id for each block within the grid
    int block_id = 
        blockIdx.x +
        blockIdx.y * gridDim.x +
        blockIdx.z * gridDim.x * gridDim.y;

    // Calculate the starting offset for threads in the current block
    int block_offset = 
        block_id *
        (blockDim.x * blockDim.y * blockDim.z); // total number of threads in a block

    // Calculate the unique offset for threads within the block
    int thread_offset = 
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;

    // Calculate the global thread id by adding block and thread offset
    int id = block_offset + thread_offset; 

    // Print the unique thread id and block/thread details
    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n", 
        id, blockIdx.x, blockIdx.y, blockIdx.z, block_id, threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main(int argc, char **argv){
    // Define dimensions for the grid of blocks
    const int b_x = 2, b_y = 3, b_z = 4; 
    // Define dimensions for each block of threads
    const int t_x = 4, t_y = 4, t_z = 4;
    // we will get 2 warp of 32 threads per block

    // Calculate and print grid/block details
    int blocks_per_grid = b_x * b_y * b_z; // Total number of blocks in the grid
    int threads_per_block = t_x * t_y * t_z; // Total number of threads in each block

    printf("%d blocks/grid\n", blocks_per_grid);
    printf("%d threads/block\n", threads_per_block);
    printf("%d total threads\n", blocks_per_grid * threads_per_block);

    // Define grid and block dimensions using dim3 to set 3D configuration
    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    // Launch the kernel function on the GPU
    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    // Ensure all threads complete before exiting
    cudaDeviceSynchronize();
}