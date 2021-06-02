/*
 * Author: Hailiang Zhang
 * Date: 3/1/2016
 */


// code guard
#ifndef H_CUDA_UTIL
#define H_CUDA_UTIL

// macros
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <iomanip>


//////////////////////////////////////////
// Error handlings
//////////////////////////////////////////
#define CALL_CUDA( err ) \
{ \
    cudaError_t cudaErr = err; \
    if (cudaErr != cudaSuccess) \
    { printf("cuda Error: \"%s\" in %s at line %d\n", cudaGetErrorString(cudaErr), __FILE__, __LINE__); exit(EXIT_FAILURE); } \
}


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

#endif
