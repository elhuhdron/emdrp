/********************************************************
 Author - Rutuja
 Date - 2/27/2016
 Extension to convert the python data structures 
 to cpp data structures and calling the cuda wrappers
 
********************************************************/
//C extension includes
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "arrayobject.h"

//system includes
#include <iostream>
#include <stdlib.h>

//cuda includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//local includes
#include "dilation.h"
#include "wrapper_dilate.h"
#include "cudaUtil.h"
#include "timer.h"

static PyMethodDef _dilation_Methods[] = {
    // EM data extensions
    {"dilate", dilate, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _pyCext in compile and linked

// https://docs.python.org/3.3/howto/cporting.html
// http://python3porting.com/cextensions.html

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "dilation_Extension",           /* m_name */
        NULL,                           /* m_doc */
        -1,                             /* m_size */
        _dilation_Methods,              /* m_methods */
        NULL,                           /* m_reload */
        NULL,                           /* m_traverse */
        NULL,                           /* m_clear */
        NULL                            /* m_free */
};
PyMODINIT_FUNC
PyInit__dilation_Extension(void)
#else
void init_dilation_Extension()
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    (void) Py_InitModule("_dilation_extension", _dilation_Methods);
    //PyObject *module = Py_InitModule("myextension", myextension_methods);
#endif

    import_array();  // Must be present for NumPy.  Called first after above line.

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

static PyObject *dilate(PyObject *self, PyObject *args){

    GpuTimer timer4;
    timer4.Start();
    PyArrayObject *input_binary_mask;
    PyArrayObject *input_structuring_element;
    npy_int input_iterations; 
    PyArrayObject *gridsize;
    PyArrayObject *output_binary_mask;
    npy_int blockdim, n_struct_elements, n_grid;
    npy_intp *dims;
    npy_int32 n_voxels_in, n_voxels_out;
    bool verbose = false; 

    if (!PyArg_ParseTuple(args, "OOOiiO", &input_binary_mask, &input_structuring_element, &output_binary_mask, &input_iterations, &blockdim, &gridsize))
      return NULL;
  
    unsigned char *h_in_binary_mask;
    h_in_binary_mask = (unsigned char*)PyArray_DATA(input_binary_mask);
    dims = PyArray_DIMS(input_binary_mask);
    n_voxels_in = dims[0]*dims[1]*dims[2];
    if(verbose) std::cout << "number of watershed pixels" << int(h_in_binary_mask[34]) <<   std::endl;

    //get the number of orthogonal jumps to make for defining adjacent pixels in the volume
    // use the numpy data type as the int* != npy_intp* or even npy_int* != npy_intp*  
    unsigned char *h_structuring_element;
    h_structuring_element = (unsigned char*)PyArray_DATA(input_structuring_element);
    dims = PyArray_DIMS(input_structuring_element);
    n_struct_elements = dims[0]*dims[1]*dims[2];
    if(verbose) std::cout<< "number of strucutring elements" << int(h_structuring_element[4]) << std::endl;

    //get the shape of the grid (used only in case of launching 3D kernel, usefulness not explored)
    int *grid;
    grid = (int*)PyArray_DATA(gridsize);
    dims = PyArray_DIMS(gridsize);
    n_grid = dims[0];
    if(verbose)  std::cout << "shape of grid" << grid[0] << " " << grid[1] << " " << grid[2] <<  std::endl;

    bool *h_out_binary_mask;
    h_out_binary_mask = (bool*)PyArray_DATA(output_binary_mask);
    dims = PyArray_DIMS(output_binary_mask);
    n_voxels_out = dims[0]*dims[1]*dims[2];
    if(verbose) std::cout << "number of watershed pixels" << n_voxels_out <<  std::endl;
   
    //initialize the timers 
    GpuTimer timer1;
    GpuTimer timer2;
    float time_memtransfer=0.0;
    float time_kernelProcessing=0.0;

    //Set the gpu to use for the application
    CALL_CUDA(cudaSetDevice(0));

    //get device properties
    cudaDeviceProp prop;
    CALL_CUDA(cudaGetDeviceProperties(&prop, 0));
    unsigned int threads = prop.maxThreadsDim[1];
    unsigned int max_blocks[3];
    max_blocks[0] = prop.maxGridSize[0];
    max_blocks[1] = prop.maxGridSize[1];
    max_blocks[2] = prop.maxGridSize[2];

    if(verbose) std::cout << "the max number of threads in each direction and max number of blocks in each direction " << threads 
                        << " " << max_blocks[0] << " " << max_blocks[1] << " " << max_blocks[2] << std::endl;


    //gpu variables
    unsigned char* d_in_binary_mask;
    bool* d_out_binary_mask;
    unsigned char* d_structuring_element;
    int* d_grid;

    // allocate and transfer memory onto the gpu
    timer1.Start();
    CALL_CUDA(cudaMalloc((void**)&d_in_binary_mask, n_voxels_in*sizeof(unsigned char)));
    CALL_CUDA(cudaMalloc((void**)&d_out_binary_mask, n_voxels_out*sizeof(bool)));
    CALL_CUDA(cudaMemcpy(d_out_binary_mask, h_out_binary_mask, n_voxels_out*sizeof(bool), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&d_structuring_element, n_struct_elements*sizeof(unsigned char)));
    CALL_CUDA(cudaMemcpy(d_structuring_element, h_structuring_element, n_struct_elements*sizeof(unsigned char), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&d_grid, n_grid*sizeof(int)));
    CALL_CUDA(cudaMemcpy(d_grid, grid, n_grid*sizeof(int), cudaMemcpyHostToDevice));
    //timer1.Stop();
    //time_memtransfer = timer1.Elapsed()/1000;
    
  
    timer2.Start();
    for(int iter = 0; iter < input_iterations; iter++){
       
        CALL_CUDA(cudaMemcpy(d_in_binary_mask, h_in_binary_mask, n_voxels_in*sizeof(unsigned char),cudaMemcpyHostToDevice));
        wrapper_dilate(d_in_binary_mask, d_out_binary_mask, d_grid, grid, //d_structuring_element
                       n_voxels_in, n_voxels_out, n_struct_elements, blockdim);
        CALL_CUDA(cudaMemcpy(h_out_binary_mask, d_out_binary_mask, n_voxels_out*sizeof(unsigned char), cudaMemcpyDeviceToHost));
        if(iter > 1){
            
            memcpy(h_in_binary_mask, h_out_binary_mask, n_voxels_out*sizeof(unsigned char));
        }
        
    }
    timer2.Stop();
    time_kernelProcessing = timer2.Elapsed()/1000;
   
    timer4.Stop();
    float time_total = timer4.Elapsed()/1000;
    //std::cout << time_total << std::endl;
    //std::cout << "The memory transfer time is: " << time_memtransfer << " seconds" << std::endl;
    //std::cout << "The kernel processing time is: " << time_kernelProcessing << " seconds" << std::endl;
    //std::cout << "the total time is: " << time_kernelProcessing + time_memtransfer  << std::endl;

    return Py_BuildValue("i",1);

}
