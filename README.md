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
  
    
```cu
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
```cu
cudaStream_t stream;
cudaStreamCreate(&stream);
myKernel<<<blocks, threads, 0, stream>>>();
cudaStreamDestroy(stream);
```
