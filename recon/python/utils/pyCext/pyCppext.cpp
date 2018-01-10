/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2018 Paul Watkins, National Institutes of Health / NINDS
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// cpp extensions for python using numpy for processing EM data.

#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include "arrayobject.h"
#include "pyCext.h"

#include <iostream>
#include <vector>
#include <assert.h>

// a RAG graph element, using the "array-of-lists" method for storing the RAG.
typedef struct {
        npy_uint32 value;
        std::vector<npy_uint64> *border_voxels;
        npy_intp last_border;
    } RAG;

// new Method to create RAG.
// method is a bit faster than adjacency matrix because the RAGs are always very sparse.
static PyObject *frag_with_borders(PyObject *self, PyObject *args) {

    PyArrayObject *input_watershed, *input_steps;
    npy_uint32 *watershed, label, edge_value, cvalue;
    npy_intp *shp, n_voxels, n_steps, edge_count=0;
    npy_int32 *steps;
    npy_uint32 n_supervoxels, nalloc_rag, nalloc_borders; 
    int min_step, max_step;
    std::vector<RAG> *cedges;
    std::vector<npy_uint64> *cborder;
    RAG crag;
   
    // parse arguments
    if (!PyArg_ParseTuple(args, "O!IO!iiII", &PyArray_Type, &input_watershed, &n_supervoxels, 
                          &PyArray_Type, &input_steps, &min_step, &max_step, &nalloc_rag, &nalloc_borders)) 
       return NULL;

    // get arguments in PythonArrayObject to access data through C data structures
    // supervoxels or "watershed" label input
    shp = PyArray_DIMS(input_watershed);
    n_voxels = PyArray_SIZE(input_watershed);
    watershed = (npy_uint32 *) PyArray_DATA(input_watershed);

    // integers specifying where to look relative to current voxel    
    n_steps = PyArray_SIZE(input_steps);
    steps = (npy_int32 *) PyArray_DATA(input_steps);

    // allocate data structure for storing edges
    std::vector<RAG> **sparse_edges = new std::vector<RAG>* [n_supervoxels];
    for( npy_uint32 i=0; i < n_supervoxels; i++ ) {
        sparse_edges[i] = new std::vector<RAG>; sparse_edges[i]->reserve(nalloc_rag);
    }
    
    // creation of rag
    std::vector<RAG>::iterator it;
    std::vector<npy_uint64>::iterator jt;
    min_step = (min_step < 0 ? -min_step : 0); max_step = (max_step > 0 ? max_step : 0);
    for( npy_int64 vox = min_step, cvox; vox < n_voxels - max_step; vox++ ) {
        label = watershed[vox];
        if( label != 0 ) { 
            // iterate "steps" which is a list of relative indices (C-order) where to search for neighbors
            for(int step = 0; step < n_steps; step++) {
                cvox = vox + steps[step]; edge_value = watershed[cvox];
                if( edge_value != 0 && edge_value != label ) { // do not add any self-directed edges
                        
                    // only store "triangular" matrix so edges are not duplicated (RAG is not directed)
                    if( edge_value < label ) {
                        cedges = sparse_edges[label-1]; cvalue = edge_value;
                    } else {
                        cedges = sparse_edges[edge_value-1]; cvalue = label;
                    }

                    // search for the edge
                    // NOTE: initially found that keeping the edge list in descending order is significantly faster.
                    //     that is  it->value <= cvalue  instead of  it->value >= cvalue
                    //   likely this is a combination of C-order and the labeling order from the watershed.
                    for( it = cedges->begin(); it != cedges->end(); it++ ) 
                        if( it->value <= cvalue ) break;
                        
                    // store the edge if not already there
                    if( it == cedges->end() || it->value != cvalue ) {
                        //cborder = NULL;
                        //if( get_borders ) { 
                            cborder = new std::vector<npy_uint64>; 
                            cborder->reserve(nalloc_borders);
                            crag.last_border = 0;
                        //}
                        crag.value = cvalue; crag.border_voxels = cborder;
                        it = cedges->insert(it, crag); edge_count++;
                    } else {
                        cborder = it->border_voxels;
                    }
                        
                    // store the border voxels
                    //if( get_borders ) {
                        // method without last border for reference.
                        //for( jt = cborder->begin(); jt != cborder->end(); jt++ ) 
                        //    if( *jt <= cvox ) break;
                        //if( jt == cborder->end() || *jt != cvox ) 
                        //    jt = cborder->insert(jt, cvox);
                        //// NOTE: this is optimized assuming that cvox > vox
                        //for( /*jt = cborder->begin()*/; jt != cborder->end(); jt++ ) 
                        //    if( *jt <= vox ) break;
                        //if( jt == cborder->end() || *jt != vox ) 
                        //    cborder->insert(jt, vox);

                        // NOTE: this is optimized assuming both that vox is always incrementating
                        //   and that cvox is always positive. This means that for an increasing list the current
                        //   voxel always has to be past the location the last voxel was inserted in the border list.
                        for( jt = cborder->begin() + it->last_border; jt != cborder->end(); jt++ ) 
                            if( *jt >= vox ) break;
                        if( jt == cborder->end() || *jt != vox ) 
                            jt = cborder->insert(jt, vox);
                        it->last_border = jt - cborder->begin();
                        // NOTE: this is optimized assuming that cvox > vox
                        for( /*jt = cborder->begin()*/; jt != cborder->end(); jt++ ) 
                            if( *jt >= cvox ) break;
                        if( jt == cborder->end() || *jt != cvox ) 
                            cborder->insert(jt, cvox);
                    //}
                }
            }
        }
    }
    //std::cout << "edges " << edge_count << std::endl;

    // create a list of edges to return in numpy array format
    npy_intp eshp[2]; eshp[0] = edge_count; eshp[1] = 2;
    PyArrayObject *list_of_edges = (PyArrayObject *) PyArray_Empty(2, eshp, PyArray_DescrFromType(NPY_UINT32), 0);
    npy_uint32 *edges = (npy_uint32 *) PyArray_DATA(list_of_edges);
    npy_intp cnt=0, bcnt;
    npy_intp bshp[1];
    // create a python list of borders that is parallel to edge list
    //PyObject* border_list = get_borders ? PyList_New(edge_count) : NULL;
    PyObject* border_list = PyList_New(edge_count);
    PyArrayObject *list_of_borders;
    npy_uint64 *borders;
    
    // copy the generated edges into the numpy array, copy border voxels into parallel list, free allocated memory
    for( npy_uint32 i=0; i < n_supervoxels; i++ ) {
        std::vector<RAG> *cedges = sparse_edges[i];
        for( std::vector<RAG>::iterator it = cedges->begin(); it != cedges->end(); it++, cnt++ ) {
            // NOTE: order that the edges are stored "upper or lower" triagular above, affects order here.
            //edges[2*cnt] = i+1; edges[2*cnt+1] = it->value; 
            edges[2*cnt] = it->value; edges[2*cnt+1] = i+1; 
            //if( get_borders ) {
                cborder = it->border_voxels; bshp[0] = cborder->size();
                list_of_borders = (PyArrayObject *) PyArray_Empty(1, bshp, PyArray_DescrFromType(NPY_UINT64), 0);
                borders = (npy_uint64 *) PyArray_DATA(list_of_borders); bcnt = 0;
                for( std::vector<npy_uint64>::iterator jt = cborder->begin(); jt != cborder->end(); jt++, bcnt++ )
                    borders[bcnt] = *jt;
                delete cborder;
                PyList_SetItem(border_list, cnt, (PyObject*) list_of_borders);
            //}
        }
        delete sparse_edges[i];
    }
    delete[] sparse_edges;
    //assert( cnt == edge_count );

    //return (PyObject *) list_of_edges; // to return single object
    return Py_BuildValue("OO", (PyObject *) list_of_edges, border_list);
}
