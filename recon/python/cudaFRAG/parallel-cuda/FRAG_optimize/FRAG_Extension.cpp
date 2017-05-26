/******************************************************
 * Author - Rutuja
 * Date - 2/27/2016
 * Extension to convert the python data structures 
 * to cpp data structures and calling the cuda wrappers
*******************************************************/
//C extension includes
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "arrayobject.h"

//system includes
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <tuple>

//cuda includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
//#include <thrust/zip_iterator.h>

//local include
#include "FRAG_Extension.h"
#include "cudaUtil.h"
#include "timer.h"
#include "wrapper_createRag.h"

// Methods Table
static PyMethodDef _FRAG_ExtensionMethods[] = {
    // EM data extensions
    {"build_frag", build_frag, METH_VARARGS},
    {"build_frag_borders", build_frag_borders, METH_VARARGS},
    {"build_frag_borders_nearest_neigh", build_frag_borders_nearest_neigh, METH_VARARGS},
    {NULL, NULL}     /* Sentinel - marks the end of this structure */
};

/* ==== Initialize the C_test functions ====================== */
// Module name must be _pyCext in compile and linked

// https://docs.python.org/3.3/howto/cporting.html
// http://python3porting.com/cextensions.html

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_FRAG_extension",           /* m_name */
        NULL,                       /* m_doc */
        -1,                         /* m_size */
        _FRAG_ExtensionMethods,      /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL                 /* m_free */
};
PyMODINIT_FUNC
PyInit__FRAG_extension(void)
#else
void init_FRAG_extension()
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    (void) Py_InitModule("_FRAG_extension", _FRAG_ExtensionMethods);
    //PyObject *module = Py_InitModule("myextension", myextension_methods);
#endif

    import_array();  // Must be present for NumPy.  Called first after above line.

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}


// redundant code
bool pairCompare(const std::tuple<int, int>& firstElem, const std::tuple<int, int>& secondElem) {
    if(std::get<0>(firstElem) < std::get<0>(secondElem)){
       return true;
    }else if(std::get<0>(secondElem) < std::get<0>(firstElem)){
        return false;
    }else{
        return std::get<1>(firstElem) < std::get<1>(secondElem);
    }
    
    //return std::get<1>(firstElem) < std::get<0>(secondElem);
}

// Method to extract data into C structures 
static PyObject *build_frag(PyObject *self, PyObject *args){

    PyArrayObject *input_watershed;
    PyArrayObject *input_steps;
    PyArrayObject *input_edges;
    PyArrayObject *input_edge_test;
    PyArrayObject *input_count;
    PyArrayObject *gridsize; 
    int edgelist_size, n_grid;
    npy_intp *dims;
    npy_int32 n_steps, blockdim, n_size;
    npy_uint32 num_pixels, batch_edge_size, label_jump;
    npy_uint64 n_supervoxels;
    bool verbose;
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "OliOiOiOiOOi", &input_watershed, &n_supervoxels, &edgelist_size, &input_edges, &verbose, &input_steps, &blockdim, &gridsize, &label_jump, &input_count, &input_edge_test, &batch_edge_size)) 
       return NULL;

    // get arguments in PythonArrayObject to access data through C data structures
    // get watershed unraveled array
  
    unsigned int *watershed_r;
    watershed_r = (unsigned int*)PyArray_DATA(input_watershed);
    dims = PyArray_DIMS(input_watershed);
    num_pixels = dims[0]*dims[1]*dims[2];
    if(verbose) std::cout << "number of watershed pixels" << num_pixels <<  std::endl;
    
    //get the number of orthogonal jumps to make for defining adjacent pixels in the volume
    // use the numpy data type as the int* != npy_intp* or even npy_int* != npy_intp*  
    npy_intp* steps;
    steps = (npy_intp*)PyArray_DATA(input_steps);
    dims = PyArray_DIMS(input_steps);
    n_steps = dims[0]; 
    if(verbose) std::cout<< "number_of steps and number of supervoxels " << n_steps << " " << n_supervoxels <<  std::endl;

    //allocate edges
    npy_int32* edges;
    edges = (npy_int32*)PyArray_DATA(input_edges);
  
     //check if there is an overflow for space allocated to hybrid adjacency matrix
    unsigned long int max_int = std::pow(2,64);
    unsigned long int size = max_int/n_supervoxels;
    if(verbose) std::cout << "limit for label_count to not exceed: " << size << std::endl;
    assert(label_jump < size); 
    assert(edgelist_size >= batch_edge_size); 
    unsigned long int edge_test_size = n_supervoxels*label_jump;
    if(verbose) std::cout << "the size of partial adjacency matrix on the gpu: " << edge_test_size << " " << label_jump << std::endl; 

    //int comp_size = num_supervoxels + (num_supervoxels*(num_supervoxels-1))/2;

    //allocate edge test data structure
    npy_uint8* edge_test;
    edge_test = (npy_uint8*)PyArray_DATA(input_edge_test);
    dims = PyArray_DIMS(input_edge_test);
    npy_uint64 k = dims[0];
    if(verbose) std::cout << " size_ of hybrid adjacency matrix " << k << std::endl;
    npy_uint32 *count_edges = (npy_uint32*)PyArray_DATA(input_count);

    unsigned int* h_labels = (unsigned int*)malloc(edgelist_size*sizeof(unsigned int));
    unsigned int* h_edges = (unsigned int*)malloc(edgelist_size*sizeof(unsigned int));
    h_labels[0] = 0;
    h_edges[0] = 0;
  
    //get the shape of the grid (used only in case of launching 3D kernel, usefulness not explored)
    int *grid; 
    grid = (int*)PyArray_DATA(gridsize);
    dims = PyArray_DIMS(gridsize);
    n_grid = dims[0];

    if(verbose)  std::cout << "shape of grid" << n_grid << grid[0] << " " << grid[1]  << " " << grid[2] << std::endl;
     
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

    if(verbose) std::cout << "the max number of threads in each direction and max number of blocks in each direction " 
                          << threads << " " << max_blocks[0] << " " << max_blocks[1] << " " << max_blocks[2] << std::endl;


    // initialize the timers
    GpuTimer timer1;
    GpuTimer timer2;
    GpuTimer timer3;
    float time_memtransfer=0.0;
    float time_kernelProcessing=0.0;
    float time_postprocessing=0.0;
      
    // initailize gpu variables 
    unsigned int* gpu_watershed;
    npy_intp* gpu_steps;
    npy_uint32* gpu_count_edges;
    unsigned int* gpu_edges;
    unsigned int* gpu_labels;
    npy_uint8* gpu_edge_test;
    int* gpu_grid;
 
    timer1.Start();
    CALL_CUDA(cudaMalloc((void**)&gpu_watershed,num_pixels*sizeof(unsigned int)));
    CALL_CUDA(cudaMemcpy(gpu_watershed,watershed_r,num_pixels*sizeof(unsigned int),cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_steps,n_steps*sizeof(npy_intp)));
    CALL_CUDA(cudaMemcpy(gpu_steps,steps,n_steps*sizeof(npy_intp),cudaMemcpyHostToDevice)); 
    CALL_CUDA(cudaMalloc((void**)&gpu_count_edges,sizeof(npy_uint32)));         
    CALL_CUDA(cudaMalloc((void**)&gpu_edges, (batch_edge_size)*sizeof(unsigned int)));
    CALL_CUDA(cudaMalloc((void**)&gpu_labels, (batch_edge_size)*sizeof(unsigned int)));
    CALL_CUDA(cudaMalloc((void**)&gpu_edge_test, (edge_test_size)*sizeof(npy_uint8)));
    CALL_CUDA(cudaMemcpy(gpu_edge_test, edge_test, (edge_test_size)*sizeof(npy_uint8), cudaMemcpyHostToDevice));  
    CALL_CUDA(cudaMalloc((void**)&gpu_grid, (n_grid)*sizeof(int)));
    CALL_CUDA(cudaMemcpy(gpu_grid, grid, (n_grid)*sizeof(int), cudaMemcpyHostToDevice));

    timer1.Stop();
    time_memtransfer = timer1.Elapsed()/1000;
    std::cout << " only the one time transfer: "  << time_memtransfer << std::endl; 
    //create the rag for the current label
    std::vector<std::tuple<unsigned int,unsigned int>> list_of_edges;
    float init = 0;
    for(unsigned long int label = 1; label <= n_supervoxels; label += label_jump){
       
        count_edges[0] = 0; 
        timer1.Start();
        CALL_CUDA(cudaMemcpy(gpu_count_edges,&count_edges[0], sizeof(npy_uint32),cudaMemcpyHostToDevice));
        CALL_CUDA(cudaMemcpy(gpu_labels,h_labels,(batch_edge_size)*sizeof(unsigned int),cudaMemcpyHostToDevice));
        CALL_CUDA(cudaMemcpy(gpu_edges,h_edges, (batch_edge_size)*sizeof(unsigned int),cudaMemcpyHostToDevice));
        timer1.Stop();
        time_memtransfer += timer1.Elapsed()/1000;
   
        timer2.Start();
        unsigned int start_label = (unsigned int)label;
        /*wrapper_createRag(gpu_watershed, gpu_steps, num_pixels, 
                      label_jump, n_steps, gpu_edges, 
                      gpu_labels, blockdim, gpu_count_edges, gpu_edge_test);*/
        wrapper_createLabelRag(gpu_watershed, gpu_steps, num_pixels, n_supervoxels,
                      label_jump, start_label, n_steps, gpu_edges,
                      gpu_labels, blockdim, gpu_count_edges, gpu_edge_test, grid, gpu_grid);
        wrapper_initialize_edge_test(gpu_edge_test, blockdim, edge_test_size, n_supervoxels, label_jump);
        timer2.Stop();
        time_kernelProcessing += timer2.Elapsed()/1000;
        
        timer1.Start();
        CALL_CUDA(cudaMemcpy(&count_edges[0],gpu_count_edges,sizeof(npy_uint32),cudaMemcpyDeviceToHost));
        CALL_CUDA(cudaMemcpy(h_labels,gpu_labels,(batch_edge_size)*sizeof(unsigned int),cudaMemcpyDeviceToHost));
        CALL_CUDA(cudaMemcpy(h_edges,gpu_edges,(batch_edge_size)*sizeof(unsigned int),cudaMemcpyDeviceToHost));
        timer1.Stop();
        time_memtransfer += timer1.Elapsed()/1000;
        // serial post processing on cpu -- to remove all the duplicates that might be added due to asynchronous functioning 
       //std::cout << "batch wise: " << count_edges[0] << std::endl;
        assert(batch_edge_size > count_edges[0]);
        for(unsigned int i = 0 ;i < count_edges[0];i++){
            list_of_edges.push_back(std::make_tuple(h_labels[i], h_edges[i]));
        }

    }
    timer3.Start();
    std::sort(list_of_edges.begin(), list_of_edges.end()); //paircompare to specify order
    auto last =  std::unique(list_of_edges.begin(), list_of_edges.end());
    list_of_edges.erase(last, list_of_edges.end());
    timer3.Stop();
    time_postprocessing = timer3.Elapsed()/1000;
    count_edges[0] = list_of_edges.size();
    
    
    unsigned int cnt = 0;
    assert(edgelist_size > count_edges[0]);
    std::vector<std::tuple<unsigned int,unsigned int>>::iterator i = list_of_edges.begin();
    for(i = list_of_edges.begin(); i != list_of_edges.end();i++){
            edges[cnt*2 + 0] = std::get<0>(*i);
            edges[cnt*2 + 1] = std::get<1>(*i);
            cnt++;
    }
    
    std::cout << "the edges generated: " << count_edges[0] << std::endl;
    CALL_CUDA(cudaFree(gpu_watershed));
    CALL_CUDA(cudaFree(gpu_steps));
    CALL_CUDA(cudaFree(gpu_edge_test));

    
    free(h_labels);
    free(h_edges);
 
    std::cout << "The memory transfer time is: " << time_memtransfer << " seconds" << std::endl;
    std::cout << "The kernel processing time is: " << time_kernelProcessing << " seconds" << std::endl;
    std::cout << "The post processing is: " << time_postprocessing << " seconds" << std::endl;
    std::cout << "the total rag creation time is" << time_memtransfer + time_kernelProcessing + time_postprocessing << " seconds" << std::endl;   
    //return 
    return Py_BuildValue("i",1);
}

static PyObject *build_frag_borders(PyObject *self, PyObject *args){

    PyArrayObject *input_watershed;
    PyArrayObject *input_edges;
    PyArrayObject *input_steps;
    PyArrayObject *input_count;
    PyArrayObject *input_borders;
    PyArrayObject *input_steps_border;
    PyArrayObject *gridsize;
    PyArrayObject *input_subind_edges;
    PyArrayObject *input_subind_borders;
    npy_uint32 n_supervoxels;
    int verbose;
    npy_intp* dims;
    npy_intp *n_voxels_dim;
    npy_intp *n_borders_dim;
    npy_intp *n_edges_dim;
    npy_uint32 n_voxels;
    npy_uint64 n_borders;
    npy_uint32 n_edges;
    npy_int n_steps_edges, n_steps_borders, blockdim, n_subind_edges, n_subind_borders;
    npy_int d_size;

    // parse arguments
    if (!PyArg_ParseTuple(args,"OlOOOiOOOOiOi", &input_watershed, &n_supervoxels, &input_edges, &input_borders, &input_count, &verbose, &input_subind_edges, &input_subind_borders, &input_steps, &input_steps_border, &blockdim, &gridsize, &d_size))
       return NULL;

    // get the watershed voxels
    unsigned int *watershed = (unsigned int*)PyArray_DATA(input_watershed);
    dims = PyArray_DIMS(input_watershed);
    n_voxels_dim = dims;
    n_voxels = n_voxels_dim[0]*n_voxels_dim[1]*n_voxels_dim[2];
    if(verbose) std::cout << "number of watershed pixels" << n_voxels << " " << d_size  << " " << n_voxels_dim[0] << " " << n_voxels_dim[1] << " " <<  n_voxels_dim[2] << std::endl;

    // necessary to typecast "steps" with npy_intp* ,otherwise we get wrong results
    // get steps for 1X dilation
    npy_intp *steps_edges = (npy_intp*)PyArray_DATA(input_steps);
    dims = PyArray_DIMS(input_steps);
    n_steps_edges = dims[0];
    if (verbose) std::cout << "number of steps " << n_steps_edges << " " << steps_edges[1] << " " << steps_edges[2] << " " 
                           << steps_edges[25] << std::endl;

    //get steps for 2X dilation 
    npy_intp *steps_border = (npy_intp*)PyArray_DATA(input_steps_border);
    dims = PyArray_DIMS(input_steps_border);
    n_steps_borders = dims[0];
    if (verbose) std::cout << "number of steps " << n_steps_borders << " " << steps_border[0] << " " << steps_border[1] << " " 
                           << steps_border[25] << std::endl;

    // sub_indices for 1X steps
    npy_intp *subind_edges = (npy_intp*)PyArray_DATA(input_subind_edges);
    dims = PyArray_DIMS(input_subind_edges);
    n_subind_edges = dims[0]*dims[1];
    if (verbose) std::cout << "number of steps " << n_subind_edges << " " << subind_edges[1] << " " << subind_edges[2] << " " 
                           << subind_edges[25] << std::endl;

    //sub_indices for 2X steps
    npy_intp *subind_borders = (npy_intp*)PyArray_DATA(input_subind_borders);
    dims = PyArray_DIMS(input_subind_borders);
    n_subind_borders = dims[0]*dims[1];
    if (verbose) std::cout << "number of steps " << n_subind_borders << " " << subind_borders[0] << " " << subind_borders[1] << " " 
                           << subind_borders[2] << std::endl;

    //get the edges and borders 
    npy_uint32 *edges = (npy_uint32*)PyArray_DATA(input_edges);
    dims = PyArray_DIMS(input_edges);
    n_edges_dim = dims;
    n_edges = n_edges_dim[0]*n_edges_dim[1];
    if(verbose) std::cout << "size of edges: " << n_edges << " " << edges[0] << " " << edges[100] << std::endl;

    npy_int32 *count = (npy_int32*)PyArray_DATA(input_count);
    if(verbose) std::cout << "the number of edges" << count[0] << std::endl; 

    npy_uint32 *borders = (npy_uint32*)PyArray_DATA(input_borders);
    dims = PyArray_DIMS(input_borders);
    n_borders_dim = dims;
    n_borders = n_borders_dim[0]*n_borders_dim[1];
    if(verbose) std::cout << "size of borders: " << n_borders << " " <<  borders[0] << " " <<  borders[1] <<" " <<  borders[2] << std::endl;
    // passing the total number of borders that a edge can have
    count[1] = n_borders_dim[1];
 
    //get the shape of the grid (used only in case of launching 3D kernel, usefulness not explored)
    int *grid;
    grid = (int*)PyArray_DATA(gridsize);
    dims = PyArray_DIMS(gridsize);
    int n_grid = dims[0];
    if(verbose)  std::cout << "shape of grid" << n_grid << grid[0] << " " << grid[1]  << " " << grid[2] << std::endl; 

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

    if(verbose) std::cout << "the max number of threads in each direction and max number of blocks in each direction "
                          << threads << " " << max_blocks[0] << " " << max_blocks[1] << " " << max_blocks[2] << std::endl;


    unsigned char* tile_check = (unsigned char*)malloc(n_steps_edges*sizeof(unsigned char));
    for(int id = 0;id < n_steps_edges;id++){
         tile_check[id] = 0;

    }

    // initialize the timers
    GpuTimer timer1;
    GpuTimer timer2;
    GpuTimer timer3;
    float time_memtransfer=0.0;
    float time_kernelProcessing=0.0;
    float time_postprocessing=0.0;

    // initailize gpu variables 
    unsigned int* gpu_watershed;
    npy_intp* gpu_steps_edges;
    npy_intp* gpu_steps_borders;
    npy_intp* gpu_subind_edges;
    npy_intp* gpu_subind_borders;
    npy_uint32* gpu_count_edges;
    npy_uint32 *gpu_edges;
    npy_uint32 *gpu_borders;
    unsigned char* gpu_tile_check;
    int* gpu_grid;

    timer1.Start();
    CALL_CUDA(cudaMalloc((void**)&gpu_watershed,n_voxels*sizeof(unsigned int)));
    CALL_CUDA(cudaMemcpy(gpu_watershed, watershed, n_voxels*sizeof(unsigned int),cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_steps_edges, n_steps_edges*sizeof(npy_intp)));
    CALL_CUDA(cudaMemcpy(gpu_steps_edges, steps_edges, n_steps_edges*sizeof(npy_intp), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_steps_borders, n_steps_borders*sizeof(npy_intp)));
    CALL_CUDA(cudaMemcpy(gpu_steps_borders, steps_border, n_steps_borders*sizeof(npy_intp), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_subind_edges, n_subind_edges*sizeof(npy_intp)));
    CALL_CUDA(cudaMemcpy(gpu_subind_edges, subind_edges, n_subind_edges*sizeof(npy_intp), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_subind_borders, n_subind_borders*sizeof(npy_intp)));
    CALL_CUDA(cudaMemcpy(gpu_subind_borders, subind_borders, n_subind_borders*sizeof(npy_intp), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_count_edges,2*sizeof(npy_uint32)));
    CALL_CUDA(cudaMemcpy(gpu_count_edges, count, 2*sizeof(npy_uint32), cudaMemcpyHostToDevice));    
    CALL_CUDA(cudaMalloc((void**)&gpu_edges, (n_edges)*sizeof(npy_uint32)));
    CALL_CUDA(cudaMemcpy(gpu_edges, edges, (n_edges)*sizeof(npy_uint32), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_borders, (n_borders)*sizeof(npy_uint32)));
    CALL_CUDA(cudaMemcpy(gpu_borders, borders, (n_borders)*sizeof(npy_uint32), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_grid, (n_grid)*sizeof(int)));
    CALL_CUDA(cudaMemcpy(gpu_grid, grid, (n_grid)*sizeof(int), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_tile_check, (n_steps_edges)*sizeof(unsigned char)));
    CALL_CUDA(cudaMemcpy(gpu_tile_check, tile_check, (n_steps_edges)*sizeof(unsigned char), cudaMemcpyHostToDevice));


    timer1.Stop();
    time_memtransfer = timer1.Elapsed()/1000;
    std::cout << " only the one time transfer: "  << time_memtransfer << std::endl; 
 
    timer2.Start();
    wrapper_get_borders(gpu_watershed, gpu_steps_edges, gpu_steps_borders, gpu_edges, gpu_borders, gpu_count_edges,
                        gpu_subind_edges, gpu_subind_borders, n_subind_edges, n_subind_borders,  
                        n_steps_borders, n_steps_edges, gpu_grid, grid, blockdim, d_size, n_borders_dim[1], gpu_tile_check);
    timer2.Stop();
    time_kernelProcessing = timer2.Elapsed()/1000;

    timer1.Start();
    CALL_CUDA(cudaMemcpy(count, gpu_count_edges, 2*sizeof(npy_uint32), cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(borders, gpu_borders, (n_borders)*sizeof(npy_uint32), cudaMemcpyDeviceToHost));
    timer1.Stop();
    time_memtransfer += timer1.Elapsed()/1000;
 
    //postprocessing
    npy_uint64 edge_index = 0;
    npy_uint64 begin = 0;
    npy_uint64 end = 0;
    std::cout << "post_processing_started" << std::endl;
    timer3.Start();
    for(unsigned int edge_size = 0; edge_size < count[0] ; edge_size++){
         edge_index = edge_size * n_borders_dim[1];
         begin = edge_index + 3;
         end = edge_index + borders[edge_index + 2];
         std::vector<npy_uint32> indices(borders + begin, borders + end);
         std::sort(indices.begin() , indices.end());
         if(edge_size == 3){
             std::cout << indices.size() << std::endl;}
         auto last = std::unique(indices.begin(), indices.end());
         indices.erase(last, indices.end());
         if(edge_size == 3)
           std::cout << indices.size() <<std::endl;
         indices.erase(std::remove(indices.begin(), indices.end(), 0), indices.end());
         borders[edge_index + 2] = indices.size();
         if(edge_size == 3){
           std::cout << borders[edge_index + 2] << std::endl;}
         std::copy(indices.begin(), indices.end(),borders + edge_index+3);

    }
    timer3.Stop();
    time_postprocessing = timer3.Elapsed()/1000;

    std::cout << "The memory transfer time is: " << time_memtransfer << " seconds" << std::endl;
    std::cout << "The kernel processing time is: " << time_kernelProcessing << " seconds" << std::endl;
    std::cout << "The post processing is: " << time_postprocessing << " seconds" << std::endl;
    std::cout << "the total rag creation time is" << time_memtransfer + time_kernelProcessing + time_postprocessing << " seconds" << std::endl;


    //return 
    return Py_BuildValue("i",1);
      
}

// get border features using the nearest neighour similar to GALA.
static PyObject *build_frag_borders_nearest_neigh(PyObject *self, PyObject *args){

    PyArrayObject *input_watershed;
    PyArrayObject *input_borders;
    PyArrayObject *input_steps;
    PyArrayObject *input_count;
    npy_uint32 n_supervoxels;
    bool verbose;
    npy_intp* dims;
    npy_intp *n_voxels_dim;
    npy_intp *n_borders_dim;
    npy_uint32 n_voxels;
    npy_uint64 n_borders;
    npy_int n_steps;
    npy_int tmp_edge_size;


//parse arguments
    if (!PyArg_ParseTuple(args,"OiOOiOi", &input_watershed, &n_supervoxels, &input_borders, &input_count, &verbose, &input_steps, &tmp_edge_size))
        return NULL;

     // get the watershed voxels
    unsigned int *h_watershed = (unsigned int*)PyArray_DATA(input_watershed);
    dims = PyArray_DIMS(input_watershed);
    n_voxels_dim = dims;
    n_voxels = n_voxels_dim[0]*n_voxels_dim[1]*n_voxels_dim[2];
    if(verbose) std::cout << "number of watershed pixels" << n_voxels << " " << n_voxels_dim[0] << " " << n_voxels_dim[1] << " " <<  n_voxels_dim[2] << std::endl;

    // necessary to typecast "steps" with npy_intp* ,otherwise we get wrong results
    // get steps for checking neighborhood 
    npy_intp *h_steps_edges = (npy_intp*)PyArray_DATA(input_steps);
    dims = PyArray_DIMS(input_steps);
    n_steps = dims[0];
    if (verbose) std::cout << "number of steps " << n_steps << " " << h_steps_edges[1] << " " << h_steps_edges[2] << " " << h_steps_edges[25] << std::endl;

    //get the edges and borders 
    //npy_uint32 *h_edges = (npy_uint32*)PyArray_DATA(input_edges);
    npy_int32 *h_count = (npy_int32*)PyArray_DATA(input_count);

    // get the structure to store borders
    npy_uint32 *h_borders = (npy_uint32*)PyArray_DATA(input_borders);
    dims = PyArray_DIMS(input_borders);
    n_borders_dim = dims;
    n_borders = n_borders_dim[0]*n_borders_dim[1];

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

    if(verbose) std::cout << "the max number of threads in each direction and max number of blocks in each direction "
                          << threads << " " << max_blocks[0] << " " << max_blocks[1] << " " << max_blocks[2] << std::endl;


    // initialize the timers
    GpuTimer timer1;
    GpuTimer timer2;
    GpuTimer timer3;
    float time_memtransfer=0.0;
    float time_kernelProcessing=0.0;
    float time_postprocessing=0.0;

    // gpu_variables 
    unsigned int *d_watershed;
    npy_intp *d_steps_edges;

    timer1.Start();
    CALL_CUDA(cudaMalloc((void**)&h_watershed,n_voxels*sizeof(unsigned int)));
    CALL_CUDA(cudaMemcpy(d_watershed, h_watershed, n_voxels*sizeof(unsigned int),cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&h_steps_edges, n_steps*sizeof(npy_intp)));
    CALL_CUDA(cudaMemcpy(d_steps_edges, h_steps_edges, n_steps*sizeof(npy_intp), cudaMemcpyHostToDevice));

    //return 
    return Py_BuildValue("i",1);
 
}
 //test code to measure time taken to sort
   
    /*int* list = (int*)malloc(1000000*sizeof(int));
    int* final_order = (int*)malloc(1000000*sizeof(int));
    for(int g = 0; g < 1000000; g++){
       list[g] = rand()%1000000;
    }
    int size = 1000000;
    int* gpu_list;
    int* gpu_final_order;
 
    GpuTimer timer4;
    CALL_CUDA(cudaMalloc((void**)&gpu_list, (size)*sizeof(int)));
    CALL_CUDA(cudaMemcpy(gpu_list, list, (size)*sizeof(int), cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_final_order, (size)*sizeof(int)));
    CALL_CUDA(cudaMemcpy(gpu_final_order, final_order, (size)*sizeof(int), cudaMemcpyHostToDevice));
    timer4.Start();
    wrapper_sort(num_pixels, gpu_list, size, gpu_final_order);
    CALL_CUDA(cudaMemcpy(final_order, gpu_final_order, size*sizeof(int),cudaMemcpyDeviceToHost));
    timer4.Stop();
    float time_sort = timer4.Elapsed()/1000;*/

 // test code to test time taken for unique on gpu
   /* int* gpu_uniquelabels;
    int* unique_labels = (int*)malloc(cpu_count[0]*sizeof(int));
    for(int l = 0; l < cpu_count[0]; l++){
        unique_labels[l] = 0;
    }
    CALL_CUDA(cudaMalloc((void**)&gpu_uniquelabels, (cpu_count[0])*sizeof(int)));
    CALL_CUDA(cudaMemcpy(gpu_uniquelabels, unique_labels, (cpu_count[0])*sizeof(int), cudaMemcpyHostToDevice));
    wrapper_post_process(num_pixels, gpu_edges, gpu_labels, cpu_count[0], gpu_uniquelabels, blockdim);
    CALL_CUDA(cudaMemcpy(unique_labels, gpu_uniquelabels, (cpu_count[0])*sizeof(int), cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(h_labels,gpu_labels,(edgelist_size)*sizeof(int),cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(h_edges,gpu_edges,(edgelist_size)*sizeof(int),cudaMemcpyDeviceToHost));*/
   /* std::vector<std::tuple<int,int>> list_of_edges;
    for(int i = 0 ;i < cpu_count[0];i++){
       if(unique_labels[i] == 0)
        list_of_edges.push_back(std::make_tuple(h_labels[i], h_edges[i]));
    } */

