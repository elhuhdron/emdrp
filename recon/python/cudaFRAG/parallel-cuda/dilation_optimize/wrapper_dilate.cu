/**************************************************************************************
 Author - Rutuja Patil
 Date - 05/08/2017

**************************************************************************************/

#include "wrapper_dilate.h"
#include "kernel_dilate.h"

extern void wrapper_dilate(const unsigned char* const d_in_binary_mask, bool* d_out_binary_mask, 
                           /*const unsigned char* const d_structuring_element,*/ const int* const  d_grid, const int* const grid,
                           const npy_int32 n_voxels_in, const npy_int32 n_voxels_out, const npy_int n_struct_elements, const int blockdim){

   

     dim3 dim_grid((grid[2]/blockdim)+1, (grid[1]/blockdim) +1, (grid[0]/blockdim) + 1), dim_block(blockdim,blockdim,blockdim);
     //printf("%d %d %d", dim_grid.x, dim_grid.y, dim_grid.z);
     
     // launch the kernel
     int shared_mem_size = blockdim*blockdim*blockdim; 
     kernel_dilate<<<dim_grid,dim_block,  shared_mem_size*sizeof(unsigned char)>>>(d_in_binary_mask, d_out_binary_mask, 
                                                                                  /*d_structuring_element,*/ n_voxels_in, 
                                                                                  n_voxels_out, n_struct_elements, d_grid, blockdim);
     cudaDeviceSynchronize();

     // get the error
     CALL_CUDA(cudaGetLastError());

}

