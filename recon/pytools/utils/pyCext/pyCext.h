/* C extensions for python using numpy for processing EM data.
 * Structure based on:
 *      http://wiki.scipy.org/Cookbook/C_Extensions/NumPy_arrays
 *
 * Created pwatkins 14 Apr 2015
 *
 *
 */

/* ==== Defines =================================== */
#define LBLS_ND   3

/* ==== Prototypes =================================== */

// .... Python callable EM data extensions ..................

static PyObject *label_affinities(PyObject *self, PyObject *args);
static PyObject *binary_warping(PyObject *self, PyObject *args);
static PyObject *merge_supervoxels(PyObject *self, PyObject *args);
static PyObject *type_components(PyObject *self, PyObject *args);

// .... Helper functions for EM data extensions ..................
npy_intp get_misclass_points(const npy_bool *src, const npy_bool *tgt, const npy_bool *msk, npy_intp numel, 
        npy_intp *pts, npy_intp *tmp_pts);
int get_nbhd_patch(npy_bool *patch, const npy_bool *src, npy_int x, npy_int y, npy_int z, npy_int m, npy_int n, 
        npy_int nz);
npy_uint32 get_simpleLUTind_from_patch(npy_bool *patch);


