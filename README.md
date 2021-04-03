#### Notes
Compile and run : `nvcc -o out dim1operqtions.cu -run`  
Profiling : `nsys profile --stats=true ./out`  after the `--stats=true` if we want to generate a report we can add `-o myreport`  
  
APOD Design Cycle : Assess, Parallelize, Optimize, Deploy.  

#### Possible Optimizations
- Change the execution context
- Set the number of blocks `= (N + threads - 1) / threads`
- Set a grid with a number of blocks that is a multiple of the number of streaming multiprocessors (SMs)
- Initialize data on GPU when possible in order to reduce the number of migrations (DtoH or HtoD) and even Page faults.
- Asynchronous Memory Prefetching is very efficient and reduce considerably the number of operations and kernels runtime.
  
    
```cpp
int deviceId;
cudaGetDevice(&deviceId);

cudadeviceProp props;
cudaGetDeviceProperties(&props, deviceId);

//Number of SMs can be accessed by
props.multiProcessorCount;

// Async prefetching before lauching a kernel
cudaMemPrefetchAsync(arr, size, deviceId); // arr is an initialized array

// Async prefetching before lauching a function
cudaMemPrefetchAsync(arr, size, cudaCpuDeviceId);

```
many more Device Info can be found in the doc (cudaDeviceProp Reference)

#### Other Optimizations
- Use Cuda stream for concurrent kernel execution
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);
myKernel<<<blocks, threads, 0, stream>>>();
cudaStreamDestroy(stream);
```
cudaMallocManaged combined with memory prefetching can be very efficient. But a more advance technic can be use to be even more efficient. A more manual way. Instead of tranfering data from DtoH or HtoD (done automatically by cudaMallocManaged) we can use manual copy using `cudaMalloc` and `cudaMallocHost` & `cudaMemcpy` wich embed the cudaDeviceSynchronize (no need to it).  

  
```cpp
int *host_arr, *device_arr;

cudaMalloc(&device_arr, size); // the array is immediately on the available GPU device. No need to prefetch

cudaMallocHost(&host_arr, size); // the array is immediately on the CPU. No need to prefetch

// Do some computations on CPU : ie initialize the array

// Then we can to copy the array to the device in order to do computation on GPU.
cudaMemcpy(device_arr, host_arr, size, cudaMemcpyHostToDevice);

// Do computations on GPU

// Then we can back to CPU for other operations
cudaMemcpy(host_arr, device_arr, size, cudaMemcpyDeviceToHost);

cudaFree(device_arr);
cudaFreeHost(host_arr);
```
Sometimes we need even more optimizations. For example if we are doing computation on an array on GPU, we have to wait for all the elements of the array to be computed to tranfer the data on CPU. A good improvement can be to do the computation into a chunked array, and tranfer all chunks asynchronously.
  
```cpp
int N = 2<<24
int size = N*sizeof(int);

int *host_arr, *device_arr;

cudaMalloc(&device_arr, size); 
cudaMallocHost(&host_arr, size);

const int nbChunks = 4;
int chunckedN = N / nbChunks;
int chunkedSize = size / nbChunks;

// perform operation for each chunk on non-0 stream
for(i=0; i<nbChunks; i++){
    
    // get the index of the chuncked part of the array
    int chunckedIndex = i * chunkedN;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // async copy and do computation on GPU
    cudaMemcpyAsync(&device_arr[chunckedIndex],
                    &host_arr[chunckedIndex],
                    chunkedSize,
                    cudaMemcpyHostToDevice,
                    stream);
    
    // while async copy do computation on chuncked arr on GPU
    myKernel<<<blocks, threads, 0, stream>>>(&device_arr[chunckedIndex], chunckedN)
    cudaStreamDestroy(stream);

    // if needed async copy back for each chunks to CPU.
}

```