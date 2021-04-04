#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cassert>


using std::begin;
using std::end;
using std::cout;
using std::endl;

__global__
void vectorAdd(int *d_a, int *d_b, int *d_c, int N) {
    int TID = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(TID < N)
        d_c[TID] = d_a[TID] + d_b[TID];

    // if the grid dimension (threads * blocks) is smaller than N, we can use a stride loop
    /*int stride = gridDim.x * blockDim.x
    for(int i=TID; i<N; i+=stride)
        d_c[TID] = d_a[TID] + d_b[TID]; */
}

void init(int *a, int *b, int N){
    for(int i=0; i<N; i++){
        a[i] = 10;
        b[i] = 10;
    }
}

void checkResult(int *a, int *b, int *c, int N){
    for(int i=0; i < N; i++)
        assert(c[i] = a[i] + b[i]);
    cout << "SUCCESS" <<endl;
}

int main(){
    int N = 1<<16; // Array size of 2^16 (65536 elts)
    size_t bytes = N * sizeof(int); // the storage have to take N elts of size int
    
    int *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, bytes);
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);

    init(h_a, h_b, N);


    // create some storage to welcome the vectors on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // now copy initialized vector to the device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    size_t threads = 1024;
    size_t blocks = (N + threads - 1) / threads; // to ensure that we will have enough blocks (padding)

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    checkResult(h_a, h_b, h_c, N);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    return 0;
}