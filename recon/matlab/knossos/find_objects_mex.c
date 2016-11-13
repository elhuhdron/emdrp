/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Paul Watkins, National Institutes of Health / NINDS
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

// Had to create an efficient version of:
//    props = regionprops(Vlbls, 'boundingbox');
// which can be done in O(nvoxels) but somehow that escaped the great folks at mathworks.
// Hard coded for 3d volumes and uint32 labels.

#include "stdint.h"
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  uint32_T *lbls, nlabels, label;
  int64_T *label_mins, *label_maxs, subs[3];
  size_t numel, m, n, nz, i, j, k;
  mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
  
  //function [label_mins, label_maxs] = find_objects(labels, nlabels)

  // labels MUST uint32 and the same size.
  numel = mxGetNumberOfElements(prhs[0]);
  if( ndims == 3 ) {
    m = mxGetM(prhs[0]); n = (mxGetDimensions(prhs[0]))[1]; nz = (mxGetDimensions(prhs[0]))[2];
  } else {
    mexPrintf("mex_find_objects: labels must be 3d, not %d dims\n",ndims);
    return;
  }
  lbls = (uint32_T*)mxGetData(prhs[0]); // must be uint32
  nlabels = *((uint32_T*)mxGetData(prhs[1])); // must be uint32
  
  // allocate the maxes and mins (defines bounding box) for each label
  label_mins = mxMalloc(nlabels*3 * sizeof(int64_T));
  label_maxs = mxMalloc(nlabels*3 * sizeof(int64_T));

  // initialize maxes to -1 and mins beyond label size
  for( i=0; i < nlabels; i++ ) {
    label_mins[i*3] = n+1; label_mins[i*3+1] = m+1; label_mins[i*3+2] = nz+1;
    label_maxs[i*3] = -1; label_maxs[i*3+1] = -1; label_maxs[i*3+2] = -1;
  }

  if( nlabels > 0 ) { // corner case
    // iterate over all voxels in the volume
    for( i = 0; i < numel; i++ ) {
      subs[0] = i % m; subs[1] = (i / m) % n; subs[2] = i / m / n; // ind2sub for 3d
      
      label = lbls[i];
      if( label > 0 && label <= nlabels ) {
        label--; // to use it as an index
        for( j = 0; j < 3; j++ ) {
          subs[j]++; // be consistent with matlab, start at 1
          if( subs[j] < label_mins[label*3+j] ) label_mins[label*3+j] = subs[j];
          if( subs[j] > label_maxs[label*3+j] ) label_maxs[label*3+j] = subs[j];
        }
      }
    }
  }
  
  plhs[0] = mxCreateNumericMatrix(0, 0, mxINT64_CLASS, mxREAL);
  mxSetData(plhs[0], label_mins); mxSetM(plhs[0], 3); mxSetN(plhs[0], nlabels);
  plhs[1] = mxCreateNumericMatrix(0, 0, mxINT64_CLASS, mxREAL);
  mxSetData(plhs[1], label_maxs); mxSetM(plhs[1], 3); mxSetN(plhs[1], nlabels);
} // mexFunction
