/******************************************************************************************
*  Author - Rutuja
* Date - 08/05/2017
*
* *****************************************************************************************/

#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "arrayobject.h"



__global__ void kernel_dilate(const unsigned char* const input_binary_map, bool* output_dilated_map, 
                             /*const unsigned char* const mask,*/ const npy_int32 num_input, const npy_int32 num_output,
                              const npy_int num_mask, const int* const  grid_shape, const int blockdim);
