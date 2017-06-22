/******************************************************************************
 * Author - Rutuja
 * Date - 03/22/2017
 * 
 * ****************************************************************************/
#include "cudaUtil.h"
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "arrayobject.h"


__global__ void create_Labelrag(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps, const npy_int n_steps, 
                                const npy_uint64 num_labels, const npy_uint32 label_jump, const unsigned int start_label, 
                                const npy_uint32 num_pixels, unsigned int* gpu_edges, unsigned int* gpu_labels, npy_uint32* d_count, 
                                npy_uint8* gpu_edge_test, const int* const gpu_grid_shape);

__global__ void initialize_edge_test(npy_uint8* gpu_edge_test, const npy_uint64 n_labels, const npy_uint64 size);


__global__ void get_borders(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps_edges, 
                            const npy_intp* const gpu_steps_borders, const unsigned int* const gpu_edges,
                            unsigned int* gpu_borders, npy_uint32* gpu_count_edges, const npy_intp* const gpu_subind_edges,
                            const npy_intp* const gpu_subind_borders, const npy_int n_subind_edges, const npy_int n_subind_borders, 
                            const npy_int n_steps_borders, const npy_int n_steps_edges, const int* const gpu_grid, 
                            const int blockdim, const int blockdim_padded, const npy_int cnst_size, const npy_int border_size,
                            unsigned char* gpu_tile_chk);

__device__ void get_dilation(const npy_intp* const steps, const int num_steps, const unsigned int* const subind_steps,
                             const npy_int n_subind_steps, const int* const dilate_idx, const int* const curr_idx, 
                             const unsigned int lab, const unsigned int edge_value, const unsigned int x_global, 
                             const unsigned int y_global , const unsigned int z_global, unsigned int start, 
                             const int* const grid_shape, unsigned int* boundary, const npy_int brd_size, const int* const dila,
                             unsigned char* gpu_tile_chk);

__global__ void get_nearest_neigh(const unsigned int* const watershed, const npy_intp* const steps,           
                                  npy_uint32* borders, const npy_uint32* const edges, const int* const grid,
                                  const npy_uint32 n_vox, const npy_int n_steps, const npy_int tmp_edge_size,        
                                  const npy_uint32 brder_size, const npy_uint32 n_labels, const npy_uint jmp);



/*__global__ void get_nearest_neigh(const unsigned int* const watershed, const npy_intp* const steps,           
                                  npy_uint32* borders, const npy_uint32* const edges, const int* const grid,
                                  const npy_uint32 n_vox, const npy_int n_steps, const npy_int tmp_edge_size,        
                                  const npy_uint32 brder_size, const npy_uint32 frst_lbl, const npy_uint32 last_lbl, const npy_uint jmp);
*/


//__global__ void sort(const int n_pixels, const int* const gpu_list, const int size, int* final_order);

//__global__ void create_unique(const int* const edges,const int* const labels,const int count, int* gpu_uniquelabels);
