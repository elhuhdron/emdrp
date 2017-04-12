/******************************************************************************
 * Author - Rutuja
 * Date - 03/22/2017
 * 
 * ****************************************************************************/
#include "cudaUtil.h"
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "arrayobject.h"


__global__ void create_rag(const int* const gpu_watershed, const npy_intp* const gpu_steps, const int n_steps, const int num_labels, 
                           const int num_pixels, int* gpu_edges, int* gpu_labels, int* d_count, int* gpu_edge_test);


__global__ void create_Labelrag(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps, const npy_int n_steps, 
                                const npy_uint32 num_labels, const npy_uint32 label_jump, const unsigned int start_label, 
                                const npy_uint32 num_pixels, unsigned int* gpu_edges, unsigned int* gpu_labels, npy_int32* d_count, 
                                npy_uint8* gpu_edge_test);

__global__ void initialize_edge_test(npy_uint8* gpu_edge_test);

__device__ bool search(int* gpu_edges, int* gpu_labels, const int count,const int edge, const int label, const bool found);


__global__ void sort(const int n_pixels, const int* const gpu_list, const int size, int* final_order);

__global__ void create_unique(const int* const edges,const int* const labels,const int count, int* gpu_uniquelabels);
