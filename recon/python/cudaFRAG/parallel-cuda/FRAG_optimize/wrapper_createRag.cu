/************************************************************************
*Author - Rutuja
*Date - 03/26/2017

*************************************************************************/
#include "wrapper_createRag.h"

extern void wrapper_createLabelRag(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps, const npy_uint32 n_pixels,
                                   const npy_uint64 n_labels, const npy_uint32 label_jump, const unsigned int start_label, 
                                   const npy_int n_steps, unsigned int* gpu_edges, unsigned int* gpu_labels, const npy_int blockdim, 
                                   npy_uint32* d_count, npy_uint8* gpu_edge_test, const int* const  watershed_grid_shape, 
                                   const int* const gpu_grid){

     dim3 dim_grid((watershed_grid_shape[2]/blockdim)+1, (watershed_grid_shape[1]/blockdim) +1, (watershed_grid_shape[0]/blockdim) + 1), dim_block(blockdim,blockdim,blockdim);
     //printf("%d %d %d", dim_grid.x, dim_grid.y, dim_grid.z);
     // launch the kernel 
     create_Labelrag<<<dim_grid,dim_block>>>(gpu_watershed, gpu_steps, n_steps, n_labels, label_jump, start_label,
                                             n_pixels, gpu_edges, gpu_labels, d_count, gpu_edge_test, gpu_grid);
     cudaDeviceSynchronize();

     // get the error
     CALL_CUDA(cudaGetLastError());

}


extern void wrapper_get_borders(const unsigned int* gpu_watershed, const npy_intp* const gpu_steps_edges, 
                                const npy_intp* const gpu_steps_borders, const unsigned int* const gpu_edges, 
                                unsigned int* gpu_borders, npy_uint32* gpu_count_edges, 
                                const npy_intp* const gpu_subind_edges, const npy_intp* const gpu_subind_borders,
                                npy_int n_subind_edges , npy_int n_subind_borders, npy_int n_steps_borders , npy_int n_steps_edges,
                                const int* const gpu_grid, const int* const grid, const npy_int blockdim, const npy_int cnst_size,
                                const npy_int border_max_size, unsigned char* gpu_tile_chk)
{
     dim3 dim_grid((grid[2]/blockdim)+1, (grid[1]/blockdim)+1, (grid[0]/blockdim)+1), dim_block(blockdim, blockdim, blockdim);

     int blockdim_padded = 2;
     int shared_mem_size = (int)std::pow((blockdim+(blockdim_padded*2)),3) + n_subind_edges + n_subind_borders;
     //launch the kernel
     get_borders<<<dim_grid, dim_block, shared_mem_size*sizeof(unsigned int)>>>(gpu_watershed, gpu_steps_edges, gpu_steps_borders, 
                                               gpu_edges, gpu_borders, gpu_count_edges, gpu_subind_edges, gpu_subind_borders,
                                               n_subind_edges, n_subind_borders, n_steps_borders, n_steps_edges, gpu_grid, 
                                               blockdim, blockdim_padded, cnst_size, border_max_size, gpu_tile_chk);
     //cudaDeviceSynchronize();

     // get the error
     CALL_CUDA(cudaGetLastError());
}


extern void wrapper_initialize_edge_test(npy_uint8* gpu_edge_test, const npy_int blockdim, const npy_uint64 edge_test_size,
                                         const npy_uint64 n_labels, const npy_uint32 label_jump){

     dim3 dim_grid((n_labels/blockdim) + 1 , (label_jump/blockdim) + 1), dim_block(blockdim, blockdim);
     //printf("\n%d, %d", dim_grid.x, dim_grid.y);
     initialize_edge_test<<<dim_grid, dim_block>>>(gpu_edge_test, n_labels, edge_test_size);
     cudaDeviceSynchronize();
     
     // check for error
     CALL_CUDA(cudaGetLastError());

}

extern void wrapper_get_borders_nearest_neigh(const unsigned int* const d_watershed, const npy_intp* const d_steps_edges, 
                                              npy_uint32 *d_borders, const npy_uint32* const d_edges, npy_int blockdim,
                                              const int* const d_grid, const int* const grid, const npy_uint32 n_voxels, 
                                              const npy_int n_steps, const npy_int tmp_edge_size, const npy_uint32 border_size,
                                              const npy_uint32 n_supervox, const npy_uint jump){

     dim3 dim_grid((grid[0]/blockdim) + 1 , (grid[1]/blockdim) + 1, (grid[2]/blockdim) +1 ), dim_block(blockdim, blockdim, blockdim);
     get_nearest_neigh<<<dim_grid, dim_block>>>(d_watershed, d_steps_edges, d_borders, d_edges, 
                                                d_grid, n_voxels, n_steps, tmp_edge_size, border_size, n_supervox, jump);
     cudaDeviceSynchronize();

     // check for error
     CALL_CUDA(cudaGetLastError());
    
}

// attempts at performing post processing on gpu side- but slower than host side

/*extern void wrapper_post_process(const int n_pixels, const int* edges, const int* labels,const int count, int* gpu_uniquelabels, const int blockdim){

     dim3 dim_grid((count/blockdim)+1), dim_block(blockdim);
 
    // launch the kernel 
     create_unique<<<dim_grid,dim_block>>>(edges, labels, count, gpu_uniquelabels);
     cudaDeviceSynchronize();

     // get the error
     CALL_CUDA(cudaGetLastError());
 
}


extern void wrapper_sort(const int n_pixels, int* gpu_list, int size, int* final_order)
{

     dim3 dim_grid((size/128)+1), dim_block(128);
     // launch the kernel 
     sort<<<dim_grid,dim_block>>>(n_pixels, gpu_list,size, final_order);
     cudaDeviceSynchronize();

     // get the error
     CALL_CUDA(cudaGetLastError());
}*/

