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
extern void wrapper_createRag(const int* const gpu_watershed, const npy_intp* const gpu_steps, const int n_pixels,
                              const int n_labels, const int n_steps, int* gpu_edges, int* gpu_labels, 
                              const int blockdim, int* d_count, int* gpu_edge_test);


extern void wrapper_createLabelRag(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps, const npy_uint32 n_pixels,
                                   const npy_uint64 n_labels, const npy_uint32 label_jump, const unsigned int start_label, 
                                   const npy_int n_steps, unsigned int* gpu_edges, unsigned int* gpu_labels, 
                                   const npy_int blockdim, npy_uint32* d_count, npy_uint8* gpu_edge_test, 
                                   const int* const watershed_grid_shape, const int* const gpu_grid);

extern void wrapper_get_borders(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps_edges,
                                const npy_intp* const gpu_steps_borders, const unsigned int* const gpu_edges,
                                unsigned int* gpu_borders, npy_uint32* gpu_count_edges, 
                                const npy_intp* const gpu_subind_edges, const npy_intp* const gpu_subind_borders,
                                npy_int n_subind_edges , npy_int n_subind_borders, npy_int n_steps_borders , npy_int n_steps_edges,
                                const int* const gpu_grid, const int* const grid, const npy_int blockdim, const npy_int cnst_size,
                                const npy_int border_max_size, unsigned char* gpu_tile_chk);


extern void wrapper_initialize_edge_test(npy_uint8* gpu_edge_test, const npy_int blockdim, const npy_uint64 edge_test_size,
                                 const npy_uint64 n_labels, const npy_uint32 label_jump);

extern void wrapper_sort(const int n_pixlels, int* gpu_list, int size, int* final_order);

extern void wrapper_get_borders_nearest_neigh(const unsigned int* const d_watershed, const npy_intp* const d_steps_edges, 
                                              npy_uint32 *d_borders, const npy_uint32* const d_edges, npy_int blockdim,
                                              const int* const d_grid, const int* const grid, const npy_uint32 n_voxels, 
                                              const npy_int n_steps, const npy_int tmp_edge_size, const npy_uint32 border_size,
                                              const npy_uint32 n_supervox, const npy_uint jump);


