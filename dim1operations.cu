#include <iostream>
#include <assert.h>


__global__
void doubleElement(int *a, int N) {
    // dataset NOT matching the context configuration (i.e Dataset larger than the total nb of thread)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x

    for(int i=idx; i<N; i+=stride)
        a[i] *= 2;
}

void initArray(int *a, int N){
    for(int i=0; i<N; i++)
        a[i] = 10;
}


int main(void){
    // length of an array
    int N = 100000;

    // allocate memory to the array
    size_t size = N * sizeof(int);
    int *a;
    cudaMallocManaged(&a, size);

    // initialize the array on the host
    initArray(a, N);

    // set grid dimension
    size_t threads = 256;
    size_t blocks = (N + threads - 1) / threads;

    doubleElement<<<blocks, threads>>>(a, N);

    checkCuda(cudaDeviceSynchronize());
    cudaFree();

    return 0;
}