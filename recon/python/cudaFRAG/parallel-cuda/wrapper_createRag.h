/**********************************************************************************
 * Author - Rutuja
 * Date - 03/22/2017
 * 
 * ********************************************************************************/
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "arrayobject.h"
 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/unique.h>

#include "kernel_createFrag.h"
typedef thrust::device_ptr<int> intlabelptr;
typedef thrust::device_vector<int> IntVector;
typedef IntVector::iterator IntIterator;
typedef thrust::tuple< IntIterator, IntIterator >  IntIteratorTuple;
typedef thrust::zip_iterator< IntIteratorTuple >  ZipIterator;


extern void wrapper_createRag(const int* const gpu_watershed, const npy_intp* const gpu_steps, const int n_pixels,
                              const int n_labels, const int n_steps, int* gpu_edges, int* gpu_labels, 
                              const int blockdim, int* d_count, int* gpu_edge_test);


extern void wrapper_createLabelRag(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps, const npy_uint32 n_pixels,
                                   const npy_uint32 n_labels, const npy_uint32 label_jump, const unsigned int start_label, 
                                   const npy_int n_steps, unsigned int* gpu_edges, unsigned int* gpu_labels, 
                                   const npy_int blockdim, npy_int32* d_count, npy_uint8* gpu_edge_test);

extern void wrapper_initialize_edge_test(npy_uint8* gpu_edge_test, const npy_int blockdim, const npy_uint32 edge_size_test);

extern void wrapper_post_process(const int n_pixels, const int* const edges, 
                                 const int* const labels,const int count, int* gpu_uniquelabels, 
                                 const int blockdim);

extern void wrapper_sort(const int n_pixlels, int* gpu_list, int size, int* final_order);
