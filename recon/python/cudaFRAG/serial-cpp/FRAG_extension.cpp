/****************************************************
 * Author - Rutuja
 * Date - 2/27/2016
 * Extension to convert the python data structures 
 * to cpp data structures. 
******************************************************/
//C extension includes
#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#include "arrayobject.h"

//system includes
#include <iostream>
#include <time.h>
#include <tuple>
#include <set>
#include <vector>
#include <algorithm>
#include <assert.h>

#include "timer.h"
#include "FRAG_extension.h"
#define DEBUG_NEW new(__FILE__, __LINE__)

// Methods Table
static PyMethodDef _frag_ExtensionMethods[] = {
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
        NULL,                /* m_doc */
        -1,                  /* m_size */
        _frag_ExtensionMethods,      /* m_methods */
        NULL,                /* m_reload */
        NULL,
        NULL,
        NULL
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
    (void) Py_InitModule("_FRAG_extension", _frag_ExtensionMethods);
    //PyObject *module = Py_InitModule("myextension", myextension_methods);
#endif

    import_array();  // Must be present for NumPy.  Called first after above line.

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}

// Method to extract data into C structures 
static PyObject *build_frag(PyObject *self, PyObject *args){

    PyArrayObject *input_watershed;
    PyArrayObject *input_edges;
    PyArrayObject *input_steps;
    PyArrayObject *input_count;
    PyArrayObject *input_edge_test;
    npy_int n_steps, connectivity;
    npy_uint32 n_voxels, n_supervoxels, size_of_edges, label_jump;
    npy_intp* dims;
    npy_intp* n_voxels_dim;
    int verbose, adjacencyMatrix;
 
   
    // parse arguments
    if (!PyArg_ParseTuple(args, "OiiiOiOiiOO", &input_watershed, &n_supervoxels, &connectivity, &size_of_edges, &input_edges, &verbose, &input_steps, &adjacencyMatrix, &label_jump, &input_count, &input_edge_test)) 
       return NULL;

    // get arguments in PythonArrayObject to access data through C data structures
    // get watershed unraveled array
    unsigned int *watershed;
    watershed = (unsigned int*)PyArray_DATA(input_watershed);
    dims = PyArray_DIMS(input_watershed);
    n_voxels_dim = dims;
    n_voxels = n_voxels_dim[0]*n_voxels_dim[1]*n_voxels_dim[2];
    if(verbose) std::cout << "number of watershed pixels" << n_voxels << " " << n_voxels_dim[0] << " " << n_voxels_dim[1] << " " <<  n_voxels_dim[2] << std::endl;

    // necessary to typecast "steps" with npy_intp* ,otherwise we get wrong results
    npy_intp *steps;
    steps = (npy_intp*)PyArray_DATA(input_steps);
    dims = PyArray_DIMS(input_steps);
    n_steps = dims[0];
    if (verbose) std::cout << "number of steps " << n_steps << " " << steps[1] << " " << steps[2] << " " << steps[25] << std::endl;
     
    std::vector<std::tuple<int,int>> list_of_edges;
    
    // get the hybrid_adjacency matrix data structure and the edges and count for edges
    npy_uint8* edge_test;
    edge_test = (npy_uint8*)PyArray_DATA(input_edge_test); 
    npy_int32 *edges = (npy_int32*)PyArray_DATA(input_edges);
    npy_int32 *count = (npy_int32*)PyArray_DATA(input_count);

    //check if there is an overflow for space allocated to hybrid adjacency matrix
    unsigned max_int = std::pow(2,32);
    unsigned int size = max_int/n_supervoxels;
    assert(label_jump < size);
    
    GpuTimer timer1;
    timer1.Start();
    npy_uint32 label;
    npy_uint32 index;
    npy_uint32 edge_value;
    float initi;
    //creation of rag
    for(unsigned int start_label = 1; start_label < n_supervoxels; start_label += label_jump){
        GpuTimer timer4;
        timer4.Start();
        memset(edge_test, 0 , (n_supervoxels*label_jump)*sizeof(npy_uint8));
        timer4.Stop();
        initi += timer4.Elapsed()/1000;
        for(unsigned int vox = 0; vox < n_voxels; vox++){
            label = watershed[vox];
            if(label !=0 && label < (start_label + label_jump) && label >= start_label){ 
                for(int step = 0;step < n_steps;step++){
                    edge_value = watershed[vox + steps[step]];
                    if(edge_value > label){
                        if(adjacencyMatrix){
                            if(label <= label_jump){
                                index = (label-1)*n_supervoxels + edge_value - 1;
                            } else{
                                index = (label - start_label)*n_supervoxels + edge_value - 1;
                            }  
                          
                            if(edge_test[index] == 0){
                                list_of_edges.push_back(std::tuple<int,int>(label,watershed[vox + steps[step]]));
                                edge_test[index] = 1;
                            }  
                          
                        }else{
                            list_of_edges.push_back(std::tuple<int,int>(label,watershed[vox + steps[step]]));
                        }
                    }      
                }
            }
        }
    }
  
    //post processing 
    std::sort(list_of_edges.begin(), list_of_edges.end()); 
    auto last = std::unique(list_of_edges.begin(), list_of_edges.end());
    list_of_edges.erase(last, list_of_edges.end()); 
    count[0] = list_of_edges.size();
    std::cout << "total edges generated for this volume: " <<  count[0] << std::endl; 
    timer1.Stop();
    float total_time = timer1.Elapsed()/1000;
    std::cout << "total rag_creation time: " << total_time << std::endl;
    std::cout << "time taken to reinitialize test edge: " << initi << std::endl;

    // check if the size of edges is enough to accomodate the edges generated
    assert(size_of_edges > count[0]);

    // copy the generated edges into the appropriate data structure
    std::vector<std::tuple<int,int>>::iterator i = list_of_edges.begin();
    unsigned int cnt = 0;
    for(i = list_of_edges.begin(); i != list_of_edges.end();i++){
            edges[cnt* 2 + 0] = std::get<0>(*i);
            edges[cnt* 2 + 1] = std::get<1>(*i);
            cnt++;
    }

    

    return Py_BuildValue("i",1);
    
}


