/********************************************************************************
* Author - Rutuja
* Date - 03/22/2017
* Kernel -  Kernel functions to create RAG and calculate the borders
********************************************************************************/

//#include <cub/cub.cuh>
//#include <cub/block/block_load.cuh>
//#include <cub/block/block_store.cuh>
//#include <cub/block/block_radix_sort.cuh>
//using namespace cub

#include "kernel_createFrag.h"

#define BLOCK 16



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
    unsigned int tmp = 0;
    bool do_add = false; 

    if(watershed_idx < num_pixels && label_val != 0 && label_val < (start_label + label_jump) && label_val >= start_label){
        
        for(unsigned int step = 0; step < n_steps; step++){
            
            unsigned int edge_value = gpu_watershed[watershed_idx + gpu_steps[step]];
            
            if(label_val <= label_jump){
            
                index = (label_val-1)*num_labels + edge_value-1;
             
            } else {
            
                index = (label_val - (factor*label_jump)-1)*num_labels + edge_value-1;

            }
            
            /*if(edge_value < label_val && edge_value != 0){    
               
                if(gpu_edge_test[index] == 0){
                     unsigned int edge_increment = atomicAdd(&d_count[0],1);
                     gpu_edges[edge_increment] = label_val;
                     gpu_labels[edge_increment] = edge_value;
                     gpu_edge_test[index] = 1;
                 }
               
             }*/
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

__global__ void create_borders(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps, const npy_int n_steps,
                               const npy_uint64 num_labels, const npy_uint32 num_pixels, unsigned int* gpu_edges, unsigned int* gpu_labels,
                               npy_uint32* d_count, const int* const gpu_grid_shape, unsigned int* gpu_borders){

     unsigned int i = blockIdx.z*blockDim.z + threadIdx.z;
     unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
     unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;
     unsigned int watershed_idx = i*gpu_grid_shape[1]*gpu_grid_shape[2] + j*gpu_grid_shape[2] + k;
     unsigned int label_val = gpu_watershed[watershed_idx];
     unsigned int border_vox;
     unsigned int edge_value;

     if(watershed_idx < num_pixels && label_val != 0){
         
         for(unsigned int step = 0; step < n_steps; step++){
         
             edge_value = gpu_watershed[watershed_idx + gpu_steps[step]];
             border_vox = watershed_idx + gpu_steps[step];

             if(edge_value > label_val){
             
                  // getting the value returned by atomicAdd in a variable and then using
                 // it gives correct values. The atomicAdd of d_count[0] in place gives wrong resutls.
                 unsigned int edge_increment = atomicAdd(&d_count[0],1);
                 gpu_edges[edge_increment] = edge_value;
                 gpu_labels[edge_increment] = label_val;
                 gpu_borders[edge_increment] = border_vox;
                 
             }
         }
     }

}


__global__ void get_borders(const unsigned int* const gpu_watershed, const npy_intp* const gpu_steps_edges,
                            const npy_intp* const gpu_steps_borders, const unsigned int* const gpu_edges,
                            unsigned int* gpu_borders, npy_uint32* gpu_count_edges, const npy_intp* const gpu_subind_edges,
                            const npy_intp* const gpu_subind_borders, const npy_int n_subind_edges,
                            const npy_int n_subind_borders, const npy_int n_steps_borders, const npy_int n_steps_edges, 
                            const int* const gpu_grid, const int blockdim, const int blockdim_padded, const npy_int cnst_size,
                            const npy_int border_size, unsigned char* gpu_tile_chk)
{

     npy_uint32 tmp_edges[100];
     unsigned int edge_val;
     npy_int dilation_index[3];
     npy_int curr_index[3];
     int dilation_cur[81];
     unsigned int counter = 0;

     extern __shared__ unsigned int bounding_box[];

     unsigned int idx_x = blockIdx.z*blockDim.z + threadIdx.z;
     unsigned int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
     unsigned int idx_z = blockIdx.x*blockDim.x + threadIdx.x;
     unsigned int watershed_idx = idx_x*gpu_grid[1]*gpu_grid[2] + idx_y*gpu_grid[2] + idx_z;
     unsigned int label = gpu_watershed[watershed_idx];

     unsigned int i;
     unsigned int j;
     unsigned int k;
     int watershed_shared_index; 
     unsigned int cnt_edges = 0;
     //tmp_edges[0] = 0;
     unsigned int start_index = 0;
     unsigned int store_index = 0;

     int shared_edge_jump = ((blockDim.z+(blockdim_padded*2))*(blockDim.y+(blockdim_padded*2))
                           *(blockDim.x+(blockdim_padded*2)));

   
     //copy watershed global memory into shared memory
     if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){

         for(int a = -(blockdim_padded) ;a < (blockdim+blockdim_padded); a++){

             for(int b = -(blockdim_padded); b < (blockdim+blockdim_padded); b++){

                 for(int c = -(blockdim_padded) ;c < (blockdim+blockdim_padded); c++){

                     watershed_shared_index = (blockDim.z+(blockdim_padded*2))*(blockDim.z+(blockdim_padded*2))*
                                   (threadIdx.z+blockdim_padded+a) + 
                                   (blockDim.y+(blockdim_padded*2))*(threadIdx.y+blockdim_padded+b) + 
                                   threadIdx.x+blockdim_padded+c;  
                     i = blockIdx.z*blockDim.z + threadIdx.z + a;
                     j = blockIdx.y*blockDim.y + threadIdx.y + b;
                     k = blockIdx.x*blockDim.x + threadIdx.x + c;
                     if(i > 0 && i < (gridDim.z*blockDim.z) && j > 0 && j < (gridDim.y*blockDim.y) 
                      && k > 0 && k < (gridDim.x*blockDim.x)){ 
                     
                        watershed_idx = i*gpu_grid[1]*gpu_grid[2] + j*gpu_grid[2] + k; 
                        bounding_box[watershed_shared_index] = gpu_watershed[watershed_idx];

                     }
                     else{
                     
                        bounding_box[watershed_shared_index] = 0;
                     }
                 }
             }
         }
        
         for(int edge_step = 0 ;edge_step < (n_subind_edges); edge_step++){
         
             bounding_box[shared_edge_jump + edge_step] = gpu_subind_edges[edge_step];

         } 

         for(int border_step = 0;border_step < (n_subind_borders); border_step++){
           
             bounding_box[shared_edge_jump + n_subind_edges + border_step] = gpu_subind_borders[border_step];
         }

     }
     
     __syncthreads();

     for(int fill_id = 0; fill_id < n_steps_edges; fill_id++){
   
        gpu_tile_chk[fill_id] = 0;  
     }

     if(label != 0){
  
       // store the edges of the current label in a temporary structure
       cnt_edges = gpu_edges[(label-1)*cnst_size];
       
       for(int s = 2; s < cnt_edges;s++){

           tmp_edges[s-1] = gpu_edges[(label-1)*cnst_size + s];
       } 
       
       tmp_edges[0] = cnt_edges-2;

       // calculate the indices of the edge 
       for(int cnt_id = 0;cnt_id < (label-1);cnt_id++){

          start_index += gpu_edges[cnt_id*cnst_size];

       }
       start_index =  start_index - (2*(label-1));

       watershed_shared_index = (blockDim.z+(blockdim_padded*2))*(blockDim.z+(blockdim_padded*2))*
                              (threadIdx.z+blockdim_padded) +
                              (blockDim.y+(blockdim_padded*2))*(threadIdx.y+blockdim_padded) +
                               threadIdx.x+blockdim_padded;


       curr_index[0] = threadIdx.z + blockdim_padded;
       curr_index[1] = threadIdx.y + blockdim_padded;
       curr_index[2] = threadIdx.x + blockdim_padded;

       for(int y= 0;y < n_steps_edges; y++){
      
           dilation_cur[y*3+0] = curr_index[0] +  bounding_box[shared_edge_jump + y*3 + 0];
           dilation_cur[y*3+1] = curr_index[1] +  bounding_box[shared_edge_jump + y*3 + 1];
           dilation_cur[y*3+2] = curr_index[2] +  bounding_box[shared_edge_jump + y*3 + 2];

       }
       for(int step = 0; step < n_steps_borders; step++){
           edge_val = bounding_box[watershed_shared_index + gpu_steps_borders[step]];
           counter = 0;
           store_index = 0;
           if(edge_val > label && edge_val != 0){
              
               while(counter < tmp_edges[0]){
                   if(edge_val == tmp_edges[counter+1]){
                   
                       dilation_index[0] = threadIdx.z + blockdim_padded + bounding_box[shared_edge_jump + n_subind_edges + step*3 + 0];
                       dilation_index[1] = threadIdx.y + blockdim_padded + bounding_box[shared_edge_jump + n_subind_edges + step*3 + 1];
                       dilation_index[2] = threadIdx.x + blockdim_padded + bounding_box[shared_edge_jump + n_subind_edges + step*3 + 2];
                    
                       store_index = start_index + counter;
                      
                       if(gpu_borders[store_index*gpu_count_edges[1]] == label){
                           
                           if(gpu_borders[store_index*gpu_count_edges[1] + 1] == edge_val){
 
                               assert(true);

                           }else{

                               assert(false);
                           }

                       }else{
 
                           printf("%d-%d-%d-%d ",label,edge_val,gpu_borders[(store_index)*gpu_count_edges[1]], store_index);
                           assert(false);
                       }

                       //get the indices  that form a border with this edge
                       get_dilation(gpu_steps_edges, n_steps_edges, &bounding_box[shared_edge_jump], n_subind_edges, dilation_index,
                                    curr_index, label, edge_val, idx_x ,idx_y, idx_z, store_index, gpu_grid, gpu_borders, border_size,
                                    dilation_cur, gpu_tile_chk);
                        
                       break;
                   }else{

                     counter++;
                   }
               }
           }      
       } 
       for(int step = 0; step < n_steps_edges; step++){
          edge_val = bounding_box[watershed_shared_index + gpu_steps_edges[step]];
          counter = 0;
          store_index = 0;

          if(edge_val >  label){

              while(counter <  tmp_edges[0]){

                   if(edge_val == tmp_edges[counter+1]){

                      dilation_index[0] = threadIdx.z + blockdim_padded + bounding_box[shared_edge_jump + step*3 + 0];
                      dilation_index[1] = threadIdx.y + blockdim_padded + bounding_box[shared_edge_jump + step*3 + 1];
                      dilation_index[2] = threadIdx.x + blockdim_padded + bounding_box[shared_edge_jump + step*3 + 2];
                      store_index = start_index + counter;

                      if(gpu_borders[store_index*gpu_count_edges[1]] == label){
                          
                          if(gpu_borders[store_index*gpu_count_edges[1] + 1] == edge_val){
                     
                              assert(true);
 
                          }else{

                              assert(false);
                          }
 
                      }else{

                          printf("%d-%d-%d-%d ",label,edge_val,gpu_borders[(store_index)*gpu_count_edges[1]], store_index);
                          assert(false);
                      }
                   
                     //get the indices  that form a border with this edge
                      get_dilation(gpu_steps_edges, n_steps_edges, &bounding_box[shared_edge_jump], n_subind_edges, dilation_index,
                                  curr_index, label, edge_val, idx_x ,idx_y, idx_z, store_index, gpu_grid, gpu_borders, border_size,
                                  dilation_cur, gpu_tile_chk);
                      break;

                   }else{
                     
                     counter++;
                   }
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

__device__ void get_dilation(const npy_intp* const steps, const int num_steps, const unsigned int* const subind_steps, 
                             const npy_int n_subind_steps, const int* const dilate_idx, const int* const curr_idx, 
                             const unsigned int lab, const unsigned int edge_value, const unsigned int x_global, 
                             const unsigned int y_global , const unsigned int z_global, unsigned int start, 
                             const int* const grid_shape, unsigned int* boundary, const npy_int brd_size, const int* const dila_2,
                             unsigned char* tile_test){

   int dila_1[81];
   int offset[3];
   unsigned int border_ind;
   bool do_add = true;
   unsigned int cnt = 0;
   unsigned int k=0; 
   int test_idx = 0;
    
   for(int step = 0; step < num_steps; step++){
   
       dila_1[step*3+0] = dilate_idx[0] + subind_steps[step*3+0];
       dila_1[step*3+1] = dilate_idx[1] + subind_steps[step*3+1];
       dila_1[step*3+2] = dilate_idx[2] + subind_steps[step*3+2];
    
       //dila_2[step*3+0] = curr_idx[0] + subind_steps[step*3+0];
       //dila_2[step*3+1] = curr_idx[1] + subind_steps[step*3+1];
       //dila_2[step*3+2] = curr_idx[2] + subind_steps[step*3+2];

 
   }

   __syncthreads();

   for(int m = 0;m < num_steps;m++){
       
       for(int n = 0;n < num_steps;n++){                  
       
           if(dila_1[m*3] == dila_2[n*3] && dila_1[m*3+1] == dila_2[n*3+1] && dila_1[m*3+2] == dila_2[n*3+2]){ 
              
               do_add = true;
               offset[0] = dila_2[n*3] - curr_idx[0];     
               offset[1] = dila_2[n*3+1] - curr_idx[1];
               offset[2] = dila_2[n*3+2] - curr_idx[2];
               test_idx = offset[0]*3*3 + offset[1]*3 + offset[2];
               if(test_idx > 13 || test_idx < -13)
                   assert(false);
                
               border_ind = (x_global+offset[0])*grid_shape[2]*grid_shape[1] + (y_global+offset[1])*grid_shape[2] + z_global+offset[2];
               assert(boundary[start*brd_size + 2] < brd_size);
               cnt = boundary[start*brd_size + 2];
               k = atomicAdd(&boundary[start*brd_size + 2] , 1);
               boundary[start*brd_size  + k] = border_ind;
                
           }    
       }         
   }
}

__global__ void get_nearest_neigh(const unsigned int* const watershed, const npy_intp* const steps, 
                                  npy_uint32* borders, const npy_uint32* const edges, const int* const grid,  
                                  const npy_uint32 n_vox, const npy_int n_steps, const npy_int tmp_edge_size, 
                                  const npy_uint32 brder_size, const npy_uint32 n_labels, const npy_uint jmp){
    

    unsigned int idx_x = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int idx_y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int idx_z = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int watershed_idx = idx_x*grid[1]*grid[2] + idx_y*grid[2] + idx_z;
    unsigned int label = watershed[watershed_idx];
    unsigned int edge_value;
    unsigned int border_ind;
    unsigned int border_vox;
    unsigned int start_index = 0;
    unsigned int store_index = 0;
    unsigned int cnt = 0;
    unsigned int k = 0;


    //if(label != 0 && label >= n_labels && label < (n_labels + jmp) && watershed_idx < n_vox){
    if(label != 0 && label >= n_labels && label <= jmp && watershed_idx < n_vox){

        // calculate the index of the edge in the datastructure
        for(int cnt_id = n_labels-1;cnt_id < label-1;cnt_id++){

            start_index += edges[cnt_id*tmp_edge_size];

        }
        start_index =  start_index - (2*(label-n_labels));
              
        for(int step = 0; step < n_steps; step++){
         
            edge_value = watershed[steps[step]+ watershed_idx];
            store_index = 0;
            if(edge_value > label){
                
                for(int index = 0;index < tmp_edge_size-2;index++){

                    // the edge list is stored with the first two columns dedicated to coulmn[0] - count of number of edges for a
                    // label and column[1] - label 
                    if(edges[(label-1)*tmp_edge_size + index + 2] == edge_value){
                      store_index = start_index + index;
                      break;
                    }
                }
   
                border_ind = steps[step]+ watershed_idx;
                border_vox = watershed_idx;
                cnt = borders[store_index*brder_size + 2];
                //assert check
                if(cnt > brder_size){

                    printf("%d ", cnt);
                }

                k = atomicAdd(&borders[store_index*brder_size + 2], 1);
                borders[store_index*brder_size + k] = border_ind;
                k = atomicAdd(&borders[store_index*brder_size + 2], 1);
                borders[store_index*brder_size + k] = border_vox;
            }               
        }
    }
}



__global__ void reinit_brdrcnt(npy_uint32* d_borders, const npy_uint32 bchsize, const npy_uint32 btch_brdrs){

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned long int idx = j*btch_brdrs + i;
  
    //printf("%d " , gridDim.x*blockDim.x);
    if(idx < bchsize && i == 2){

        d_borders[j*btch_brdrs + i] = 3; 

    }

}

/*template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(int *d_in, int *d_out)
{
    // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
    typedef cub::BlockLoad<
        int*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_TRANSPOSE> BlockLoadT;
    typedef cub::BlockStore<
        int*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_TRANSPOSE> BlockStoreT;
    typedef cub::BlockRadixSort<
        int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoadT::TempStorage       load; 
        typename BlockStoreT::TempStorage      store; 
        typename BlockRadixSortT::TempStorage  sort;
    } temp_storage; 
    // Obtain this block's segment of consecutive keys (blocked across threads)
    int thread_keys[ITEMS_PER_THREAD];
    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);   
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);
    
    __syncthreads();    // Barrier for smem reuse
    // Collectively sort the keys
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);
    __syncthreads();    // Barrier for smem reuse
    // Store the sorted segment 
    BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
}*/

/*__launch_bounds__ (BLOCK_THREADS);

// Kernels
//---------------------------------------------------------------------
__global__ void BlockSortKernel(
    Key         *d_in,          // Tile of input
    Key         *d_out,         // Tile of output
    clock_t     *d_elapsed)     // Elapsed cycle count of block scan
{
    enum { TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD };
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef BlockLoad<Key, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    // Specialize BlockRadixSort type for our thread block
    typedef BlockRadixSort<Key, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    // Shared memory
    __shared__ union
    {
        typename BlockLoadT::TempStorage        load;
        typename BlockRadixSortT::TempStorage   sort;
    } temp_storage;
    // Per-thread tile items
    Key items[ITEMS_PER_THREAD];
    // Our current block's offset
    int block_offset = blockIdx.x * TILE_SIZE;
    // Load items into a blocked arrangement
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, items);
    // Barrier for smem reuse
    __syncthreads();
    // Start cycle timer
    clock_t start = clock();
    // Sort keys
    BlockRadixSortT(temp_storage.sort).SortBlockedToStriped(items);
    // Stop cycle timer
    clock_t stop = clock();
    // Store output in striped fashion
    StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_out + block_offset, items);
    // Store elapsed clocks
    if (threadIdx.x == 0)
    {
        d_elapsed[blockIdx.x] = (start > stop) ? start - stop : stop - start;
    }
}*/
