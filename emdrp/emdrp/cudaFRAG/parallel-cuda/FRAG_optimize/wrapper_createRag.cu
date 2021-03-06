/************************************************************************
* Author - Rutuja
* Date - 03/26/2017
* Function - Wrapper Functions calling cuda kernels.
*************************************************************************/
#include "wrapper_createRag.h"

extern void wrapper_createLabelRag(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps, const npy_uint32 n_pixels,
                                   const npy_uint64 n_labels, const npy_uint32 label_jump, const unsigned int start_label, 
                                   const npy_int n_steps, unsigned int* gpu_edges, unsigned int* gpu_labels, 
                                   const npy_int blockdim, npy_uint32* d_count, npy_uint8* gpu_edge_test, 
                                   const int* const  watershed_grid_shape, const int* const gpu_grid){

     dim3 dim_grid((watershed_grid_shape[2]/blockdim)+1, (watershed_grid_shape[1]/blockdim) +1, (watershed_grid_shape[0]/blockdim) + 1), dim_block(blockdim,blockdim,blockdim);
     // launch the kernel 
     create_Labelrag<<<dim_grid,dim_block>>>(gpu_watershed, gpu_steps, n_steps, n_labels, label_jump, start_label,
                                             n_pixels, gpu_edges, gpu_labels, d_count, gpu_edge_test, gpu_grid);
     cudaDeviceSynchronize();

     // get the error
     CALL_CUDA(cudaGetLastError());

}

extern void wrapper_createBorder(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps, const npy_uint32 n_pixels,
                                 const npy_uint64 n_labels, const npy_int n_steps, unsigned int* gpu_edges, unsigned int* gpu_labels,
                                 unsigned int* gpu_borders, const npy_int blockdim, npy_uint32* d_count, 
                                 const int* const  watershed_grid_shape, const int* const gpu_grid){

    dim3 dim_grid((watershed_grid_shape[2]/blockdim)+1, (watershed_grid_shape[1]/blockdim) +1, (watershed_grid_shape[0]/blockdim) + 1), dim_block(blockdim,blockdim,blockdim);

    create_borders<<<dim_grid, dim_block>>>(gpu_watershed, gpu_steps, n_steps, n_labels, n_pixels, gpu_edges, gpu_labels,
                                            d_count, gpu_grid, gpu_borders);

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
                                              const int* const d_grid, const int* const grid, 
                                              const npy_uint32 n_voxels, const npy_int n_steps, const npy_int tmp_edge_size, 
                                              const npy_uint32 border_size, const npy_uint32 n_supervox, const npy_uint jump){

     dim3 dim_grid((grid[0]/blockdim) + 1 , (grid[1]/blockdim) + 1, (grid[2]/blockdim) +1 ), dim_block(blockdim, blockdim, blockdim);
     get_nearest_neigh<<<dim_grid, dim_block>>>(d_watershed, d_steps_edges, d_borders, d_edges, 
                                                d_grid, n_voxels, n_steps, tmp_edge_size, border_size, n_supervox, jump);
     cudaDeviceSynchronize();

     // check for error
     CALL_CUDA(cudaGetLastError());
    
}

extern void wrapper_reinit_brdrcnt(npy_uint32* d_borders, npy_uint32 batchsize, npy_int blockdim, npy_uint jump, npy_uint32 batch_brdrs){

    dim3 dim_grid((batch_brdrs/blockdim)+1, (jump/blockdim) +1), dim_block(blockdim,blockdim);

    reinit_brdrcnt<<<dim_grid, dim_block>>>(d_borders, batchsize, batch_brdrs);

    cudaDeviceSynchronize();

    // check for error
    CALL_CUDA(cudaGetLastError());

}

/*extern void wrapper_sort(npy_uint32* d_borders, const npy_uint32 begin, const npy_uint32 part, const npy_uint32 edge_index){

    int num_blocks = (part - begin)/128;
    int *d_in = (int*)malloc(sizeof(int)*(part-begin));
    d_in = NULL;;
    int *d_out = (int*)malloc(sizeof(int)*(part-begin));
    d_out = NULL;
    BlockSortKernel<128, 16><<<num_blocks, 128>>>(d_in, d_out);
    
}*/
