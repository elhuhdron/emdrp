/********************************************************************************
* Author - Rutuja
* Date - 03/22/2017

********************************************************************************/
#include "kernel_createFrag.h"

#define BLOCK 16

__global__ void create_rag(const int* const gpu_watershed, const npy_intp* const gpu_steps, const int n_steps, const int num_labels,
                           const int num_pixels, int* gpu_edges, int* gpu_labels, int*  d_count, int* gpu_edge_test){

     int watershed_idx = blockIdx.x*blockDim.x + threadIdx.x;
     int label_val = gpu_watershed[watershed_idx]; 
     if(watershed_idx < num_pixels && label_val != 0){
         for(int step = 0; step < n_steps; step++){
             int edge_value = gpu_watershed[watershed_idx + gpu_steps[step]];
             int index = (label_val-1)*num_labels + edge_value-1;
             if(edge_value > label_val){
                 // getting the value returned by atomicAdd in a variable and then using
                 // it gives correct values. The atomicAdd of d_count[0] in place gives
                 // wrong resutls.
                    //int found  = 0;
                    int start_index =  num_labels*(label_val-1)  - (((label_val-1)*(label_val-2))/2);
                    //int end_index = start_index + num_labels - (label_val - 1) ;
      
                    if(gpu_edge_test[start_index + (edge_value - label_val)] == 0){  
                          int edge_increment = atomicAdd(&d_count[0],1);
                          gpu_edges[edge_increment] = edge_value;
                          gpu_labels[edge_increment] = label_val;   
                          gpu_edge_test[start_index + (edge_value - label_val)] = 1;
                    }        
                 
             }
         }
     }
}

__global__ void create_Labelrag(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps, const npy_int n_steps, 
                                const npy_uint64 num_labels, const npy_uint32 label_jump, const unsigned int start_label, 
                                const npy_uint32 num_pixels, unsigned int* gpu_edges, unsigned int* gpu_labels, 
                                npy_uint32* d_count, npy_uint8* gpu_edge_test, const int* const gpu_grid_shape){

     unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
     unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
     unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
     unsigned int watershed_idx = i*gpu_grid_shape[1]*gpu_grid_shape[2] + j*gpu_grid_shape[2] + k; 
     unsigned int label_val = gpu_watershed[watershed_idx];
     unsigned long int index;
     unsigned int factor = start_label/label_jump;
   
     if(watershed_idx < num_pixels && label_val != 0 && label_val < (start_label + label_jump) && label_val >= start_label){
         for(unsigned int step = 0; step < n_steps; step++){
             unsigned int edge_value = gpu_watershed[watershed_idx + gpu_steps[step]];
             if(label_val <= label_jump){
                 index = (label_val-1)*num_labels + edge_value-1;
             } else{
                 index = (label_val - (factor*label_jump)-1)*num_labels + edge_value-1;
                 //if(label_val == 146755 && edge_value == 147243)
                   //printf("%lu ", (label_val - start_label)*num_labels);
             }
             if(edge_value > label_val){
                 // getting the value returned by atomicAdd in a variable and then using
                 // it gives correct values. The atomicAdd of d_count[0] in place gives
                 // wrong resutls.
                 if(gpu_edge_test[index] == 0){
                     unsigned int edge_increment = atomicAdd(&d_count[0],1);
                     gpu_edges[edge_increment] = edge_value;
                     gpu_labels[edge_increment] = label_val;
                     gpu_edge_test[index] = 1;
                 }

             }
         }
     }
}



__global__ void initialize_edge_test(npy_uint8* gpu_edge_test, const npy_uint64 n_labels, const npy_uint64 size){


   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
   unsigned long int idx = j*n_labels + i;

   if(idx < size){ 
     gpu_edge_test[idx] = 0;
   }

}


__global__ void sort(const int n_pixels, const int* const gpu_list, const int size, int* final_order){

    int cnt = 0;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;   
    int val = gpu_list[idx];
    int same_cnt = 0;
    if(idx < size)
    for(int j = 0 ; j< size;j++){

       if(val > gpu_list[j]){

          cnt++;
       }
    }

    final_order[cnt] = val;

}

__global__ void create_unique(const int* const edges,const int* const labels, const int count, int* gpu_uniquelabels){


    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int val_edge = edges[idx];
    int val_label = labels[idx];;
    if(idx < count){

        if(idx != 0){
  
            if(val_edge == edges[idx-1] && val_label == labels[idx-1]){
            
                gpu_uniquelabels[idx] = 1;
            }
                    
        }
    }

}

