/************************************************************************
*Author - Rutuja
*Date - 03/26/2017

*************************************************************************/
#include "wrapper_createRag.h"

extern void wrapper_createRag(const int* const gpu_watershed, const npy_intp* const gpu_steps, const int n_pixels,
                              const int n_labels, const int n_steps,
                              int* gpu_edges, int* gpu_labels, const int blockdim, int* d_count, int* gpu_edge_test){

     dim3 dim_grid((n_pixels/blockdim)+1), dim_block(blockdim); 
     // launch the kernel 
     create_rag<<<dim_grid,dim_block>>>(gpu_watershed, gpu_steps, n_steps, n_labels,
                                        n_pixels, gpu_edges, gpu_labels, d_count, gpu_edge_test);
     cudaDeviceSynchronize();
     
     // get the error
     CALL_CUDA(cudaGetLastError());

}


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


extern void wrapper_initialize_edge_test(npy_uint8* gpu_edge_test, const npy_int blockdim, const npy_uint64 edge_test_size,
                                         const npy_uint64 n_labels, const npy_uint32 label_jump){

     dim3 dim_grid((n_labels/blockdim) + 1 , (label_jump/blockdim) + 1), dim_block(blockdim, blockdim);
     //printf("\n%d, %d", dim_grid.x, dim_grid.y);
     initialize_edge_test<<<dim_grid, dim_block>>>(gpu_edge_test, n_labels, edge_test_size);
     cudaDeviceSynchronize();
     
     // check for error
     CALL_CUDA(cudaGetLastError());

}



extern void wrapper_post_process(const int n_pixels, const int* edges, const int* labels,const int count, int* gpu_uniquelabels, const int blockdim){

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
}

    /*intlabelptr dev_dataptr = thrust::device_pointer_cast(labels);
    intlabelptr dev_edgeptr = thrust::device_pointer_cast(edges);
    thrust::device_vector<int> d_vec(dev_dataptr, dev_dataptr + count[0]);
    thrust::device_vector<int> d_vec2(dev_edgeptr, dev_edgeptr + count[0]);
    thrust::sort_by_key(d_vec.begin(), d_vec.end(), d_vec2.begin());
    //thrust::sort(thrust::make_zip_iterator( thrust::make_tuple( d_vec.begin(), d_vec2.begin() ) ),
      //                                   thrust::make_zip_iterator( thrust::make_tuple( d_vec.end(), d_vec2.end() ) ) );

    ZipIterator newEnd = thrust::unique( thrust::make_zip_iterator( thrust::make_tuple( d_vec.begin(), d_vec2.begin() ) ),
                                         thrust::make_zip_iterator( thrust::make_tuple( d_vec.end(), d_vec2.end() ) ) );

    IntIteratorTuple endTuple = newEnd.get_iterator_tuple();


    d_vec.erase( thrust::get<0>( endTuple ), d_vec.end() );
    d_vec2.erase( thrust::get<1>( endTuple ), d_vec2.end() );
    printf("%d" , d_vec.size());;*/

