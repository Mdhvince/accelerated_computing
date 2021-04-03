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

void checkResult(std::vector<int> a, std::vector<int> b, std::vector<int> c){

    for(int i=0; i < c.size(); i++)
        assert(c[i] = a[i] + b[i]);
}

int main(void){
    int N = 1<<16; // Array size of 2^16 (65536 elts)
    size_t bytes = N * sizeof(int); // the storage have to take N elts of size int
    
    
    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> c(N);
    // initialize a and b vectors on CPU with random numbers
    std::genrate(begin(a), end(a), [](){
        return std::rand() % 100;
    });
    std::genrate(begin(b), end(b), [](){
        return std::rand() % 100;
    });

    // create some storage to welcome the vectors on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // now copy initialized vector to the device
    cudaMemcpy(&d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    int threads = 1024;
    int blocks = (N + threads - 1) / threads; // to ensure that we will have enough blocks (padding)

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    checkResult(a, b, c);
    cout << "SUCCESS" <<endl;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}