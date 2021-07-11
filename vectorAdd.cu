#include <iostream>
#include <vector>
#include <cassert>



__global__
void vectorAdd(int *d_a, int *d_b, int *d_c, const int N) {
    int thread_id_in_grid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for(int i=thread_id_in_grid; i<N; i+=stride)
        d_c[thread_id_in_grid] = d_a[thread_id_in_grid] + d_b[thread_id_in_grid];
}

void initCPU(int *a, int *b, const int N){
    for(int i=0; i<N; i++){
        a[i] = 76;
        b[i] = 837;
    }
}

void checkResult(int *a, int *b, int *c, const int N){
    for(int i=0; i < N; i++){
        assert(c[i] = a[i] + b[i]); 
    }
    std::cout << "SUCCESS" <<std::endl;
}

int main(){

    int *h_a, *h_b, *h_c = nullptr;
    int *d_a, *d_b, *d_c = nullptr;

    const int N = 1<<16;
    size_t bytes = N * sizeof(int);

    size_t threads = 1024;
    size_t blocks = (N + threads - 1) / threads;              // to ensure that we will have enough blocks (padding)
    
    cudaMallocHost(&h_a, bytes);                              // cudaMemcpy() call is faster (higher bandwidth) when using cudaMallocHost() instead of malloc()
    cudaMallocHost(&h_b, bytes);
    cudaMallocHost(&h_c, bytes);
    initCPU(h_a, h_b, N);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    

    checkResult(h_a, h_b, h_c, N);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);

    return 0;
}