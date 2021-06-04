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

/* C extensions for python using numpy for processing EM data.
 * Structure based on:
 *      http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays
 */

/* ==== Defines =================================== */
#define LBLS_ND   3

/* ==== Prototypes =================================== */

// .... Python callable EM data extensions ..................

static PyObject *label_affinities(PyObject *self, PyObject *args);
static PyObject *binary_warping(PyObject *self, PyObject *args);
static PyObject *type_components(PyObject *self, PyObject *args);
static PyObject *remove_adjacencies(PyObject *self, PyObject *args);
static PyObject *label_overlap(PyObject *self, PyObject *args);

// .... Helper functions for EM data extensions ..................
npy_intp get_misclass_points(const npy_bool *src, const npy_bool *tgt, const npy_bool *msk, npy_intp numel,
        npy_intp *pts, npy_intp *tmp_pts);
int get_nbhd_patch(npy_bool *patch, const npy_bool *src, npy_int x, npy_int y, npy_int z, npy_int m, npy_int n,
        npy_int nz);
npy_uint32 get_simpleLUTind_from_patch(npy_bool *patch);
