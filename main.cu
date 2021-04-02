#include <iostream>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

void normalCPU() {
    std::cout<<"Test from CPU."<<std::endl;
}

__global__ void kernelTest() {
    std::cout<<"Test from GPU."<<std::endl;
}

void loop(int N) {
    for (int i = 0; i < N; ++i)
        std::cout<<"Iteration from CPU: "<< i <<std::endl;
}

__global__ void AcceleratedGridStrideLoop(int sizeIterable) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x

    for(int i=idx; i<sizeIterable; i+=gridStride )
        std::cout<<"Iteration from GPU: "<< i <<std::endl;
}



int main(void){
    // length of an iterable
    int N = 100000;

    /* allocate memory
    size_t size = N * sizeof(int);
    int *a;
    cudaMallocManaged(&a, size);
    */


    
    size_t threads_per_block = 256;
    // Ensure there are at least `N` threads in the grid, but only 1 block's worth extra
    size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

    AcceleratedGridStrideLoop<<<number_of_blocks, threads_per_block>>>(N);
    checkCuda(cudaDeviceSynchronize());

    /* free memory if allocated
    cudaFree();
    */

    return 0;
}