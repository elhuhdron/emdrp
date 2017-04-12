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
    npy_uint32 num_pixels, n_supervoxels, label_jump, batch_edge_size;
    bool verbose;
    
    // parse arguments
    if (!PyArg_ParseTuple(args, "OiiOiOiOiOOi", &input_watershed, &n_supervoxels, &edgelist_size, &input_edges, &verbose, &input_steps, &blockdim, &gridsize, &label_jump, &input_count, &input_edge_test, &batch_edge_size)) 
       return NULL;

    // get arguments in PythonArrayObject to access data through C data structures
    // get watershed unraveled array
  
    unsigned int *watershed_r;
    watershed_r = (unsigned int*)PyArray_DATA(input_watershed);
    dims = PyArray_DIMS(input_watershed);
    num_pixels = dims[0]*dims[1]*dims[2];
    if(verbose) std::cout << "number of watershed pixels" << num_pixels << " " << watershed_r[39459] <<  std::endl;
    
    //get the number of orthogonal jumps to make for defining adjacent pixels in the volume
    // use the numpy data type as the int* != npy_intp* or even npy_int* != npy_intp*  
    npy_intp* steps;
    steps = (npy_intp*)PyArray_DATA(input_steps);
    dims = PyArray_DIMS(input_steps);
    n_steps = dims[0]; 
    if(verbose) std::cout<< "number_of steps" << n_steps << std::endl;

    //allocate edges
    npy_int32* edges;
    edges = (npy_int32*)PyArray_DATA(input_edges);
  
     //check if there is an overflow for space allocated to hybrid adjacency matrix
    unsigned max_int = std::pow(2,32);
    unsigned int size = max_int/n_supervoxels;
    assert(label_jump < size);  
    unsigned int edge_test_size = n_supervoxels*label_jump; 
    //int comp_size = num_supervoxels + (num_supervoxels*(num_supervoxels-1))/2;

    //allocate edge test data structure
    npy_uint8* edge_test;
    edge_test = (npy_uint8*)PyArray_DATA(input_edge_test);
    npy_int32 *count_edges = (npy_int32*)PyArray_DATA(input_count);

    unsigned int* h_labels = (unsigned int*)malloc(edgelist_size*sizeof(unsigned int));
    unsigned int* h_edges = (unsigned int*)malloc(edgelist_size*sizeof(unsigned int));
    h_labels[0] = 0;
    h_edges[0] = 0;
  
    //get the shape of the grid (used only in case of launching 3D kernel, usefulness not explored)
    int *grid; 
    grid = (int*)PyArray_DATA(gridsize);
    dims = PyArray_DIMS(gridsize);
    n_grid = dims[0];
    if(verbose)  std::cout << "shape of grid" << grid[0] << " " << batch_edge_size << std::endl;
     
    //Set the gpu to use for the application
    CALL_CUDA(cudaSetDevice(1));

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
    npy_int32* gpu_count_edges;
    unsigned int* gpu_edges;
    unsigned int* gpu_labels;
    npy_uint8* gpu_edge_test;
    
    timer1.Start();
    CALL_CUDA(cudaMalloc((void**)&gpu_watershed,num_pixels*sizeof(unsigned int)));
    CALL_CUDA(cudaMemcpy(gpu_watershed,watershed_r,num_pixels*sizeof(unsigned int),cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc((void**)&gpu_steps,n_steps*sizeof(npy_intp)));
    CALL_CUDA(cudaMemcpy(gpu_steps,steps,n_steps*sizeof(npy_intp),cudaMemcpyHostToDevice)); 
    CALL_CUDA(cudaMalloc((void**)&gpu_count_edges,sizeof(npy_int32)));            
    CALL_CUDA(cudaMalloc((void**)&gpu_edges, (batch_edge_size)*sizeof(unsigned int)));
    CALL_CUDA(cudaMalloc((void**)&gpu_labels, (batch_edge_size)*sizeof(unsigned int)));
    CALL_CUDA(cudaMalloc((void**)&gpu_edge_test, (edge_test_size)*sizeof(npy_uint8)));
    CALL_CUDA(cudaMemcpy(gpu_edge_test, edge_test, (edge_test_size)*sizeof(npy_uint8), cudaMemcpyHostToDevice));
    timer1.Stop();
    time_memtransfer = timer1.Elapsed()/1000;
    std::cout << " only the one time transfer: "  << time_memtransfer << std::endl; 
    //create the rag for the current label
    std::vector<std::tuple<unsigned int,unsigned int>> list_of_edges;
    GpuTimer timer4;
    float init= 0;
    for(unsigned int label = 1; label <= n_supervoxels; label += label_jump){
       
        //timer4.Start();
        //memset(edge_test, 0 , edge_test_size*sizeof(npy_uint8));
        //timer4.Stop();
        //init += timer4.Elapsed()/1000;
        count_edges[0] = 0; 
        timer1.Start();
        CALL_CUDA(cudaMemcpy(gpu_count_edges,&count_edges[0], sizeof(npy_int32),cudaMemcpyHostToDevice));
        CALL_CUDA(cudaMemcpy(gpu_labels,h_labels,(batch_edge_size)*sizeof(unsigned int),cudaMemcpyHostToDevice));
        CALL_CUDA(cudaMemcpy(gpu_edges,h_edges, (batch_edge_size)*sizeof(unsigned int),cudaMemcpyHostToDevice));
        timer1.Stop();
        time_memtransfer += timer1.Elapsed()/1000;

        timer2.Start();
        unsigned int start_label = label;
        /*wrapper_createRag(gpu_watershed, gpu_steps, num_pixels, 
                      label_jump, n_steps, gpu_edges, 
                      gpu_labels, blockdim, gpu_count_edges, gpu_edge_test);*/
        wrapper_createLabelRag(gpu_watershed, gpu_steps, num_pixels, n_supervoxels,
                      label_jump, start_label, n_steps, gpu_edges,
                      gpu_labels, blockdim, gpu_count_edges, gpu_edge_test);
        wrapper_initialize_edge_test(gpu_edge_test, blockdim, edge_test_size);

        timer2.Stop();
        time_kernelProcessing += timer2.Elapsed()/1000;
  
        timer1.Start();
        CALL_CUDA(cudaMemcpy(&count_edges[0],gpu_count_edges,sizeof(npy_int32),cudaMemcpyDeviceToHost));
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
    std::cout << "time just required to init: " << init << std::endl;
    timer3.Start();
    std::sort(list_of_edges.begin(), list_of_edges.end()); //paircompare to specify order
    auto last =  std::unique(list_of_edges.begin(), list_of_edges.end());
    list_of_edges.erase(last, list_of_edges.end());
    timer3.Stop();
    time_postprocessing = timer3.Elapsed()/1000;
    count_edges[0] = list_of_edges.size();
    
    if(verbose){
        unsigned int cnt = 0;
        assert(edgelist_size > count_edges[0]);
        std::vector<std::tuple<unsigned int,unsigned int>>::iterator i = list_of_edges.begin();
        for(i = list_of_edges.begin(); i != list_of_edges.end();i++){
            edges[cnt*2 + 0] = std::get<0>(*i);
            edges[cnt*2 + 1] = std::get<1>(*i);
            cnt++;
        }
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

