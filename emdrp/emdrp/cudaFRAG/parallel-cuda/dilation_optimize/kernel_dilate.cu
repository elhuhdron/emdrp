/***************************************************************************************

* Author - Rutuja
* Date - 08/05/2017

*****************************************************************************************/

#include "kernel_dilate.h"


__global__ void kernel_dilate(const unsigned char* const input_binary_map, bool* output_dilated_map, 
                              /*const unsigned char* const mask,*/ const npy_int32 num_input, const npy_int32 num_output,
                              const npy_int num_mask, const int* const  grid_shape, const int blockdim){

     extern __shared__ unsigned char curr_svox[];
     unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
     unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
     unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
     unsigned int idx = i*grid_shape[1]*grid_shape[2] + j*grid_shape[2] + k;
     unsigned int shared_idx = threadIdx.z*blockdim*blockdim + threadIdx.y*blockdim + threadIdx.x;   
 
     if(idx < num_input){
     
         curr_svox[shared_idx] = input_binary_map[idx];
 
     }
     else{

         curr_svox[shared_idx] = 0;

     }

     __syncthreads();

     int mask_idx;
     if(idx < num_input){
 
         if(curr_svox[shared_idx] == 1){
            
            output_dilated_map[idx] = true;
             
            for(int l = -1;l < 2; l++){
              
                for(int m = -1;m < 2;m++){
            
                    for(int n = -1;n < 2;n++){
                         
                        mask_idx = (i+l)*grid_shape[1]*grid_shape[2] + (j+m)*grid_shape[2] + k+n;
                        output_dilated_map[mask_idx] = true;
                                 
                     }

                 }
             }
         }
     }

}

