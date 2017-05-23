/**************************************************************************
 * Author - Rutuja
 * Date  - 05/08/2017
 * ************************************************************************/

#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "arrayobject.h"
#include "cudaUtil.h"

extern void wrapper_dilate(const unsigned char* const d_in_binary_mask, bool* d_out_binary_mask, 
                           /*const unsigned char* const d_structuring_element,*/ const int* const  d_grid, const int* const grid,
                           const npy_int32 n_voxels_in, const npy_int32 n_voxels_out, const npy_int n_struct_elements, const int blockdim);
