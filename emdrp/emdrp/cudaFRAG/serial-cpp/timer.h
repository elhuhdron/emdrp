/*************************************************************************
 * Author - Rutuja
 * Date - 2/27/2016
 * Header 
 *************************************************************************/

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

struct GpuTimer{
    cudaEvent_t start;
    cudaEvent_t stop;
    GpuTimer(){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer(){
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void Start(){
        cudaEventRecord(start, 0);
    }
    void Stop(){
        cudaEventRecord(stop, 0);
    }
    float Elapsed(){
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

