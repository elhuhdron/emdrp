/*************************************************************************
 *
 *
 *
 * ************************************************************************/


static PyObject *build_frag(PyObject *self, PyObject *args);
static PyObject *build_frag_borders(PyObject *self, PyObject *args);
void get_dilation(unsigned int* dila_1, unsigned int* dila_2, npy_intp* steps, npy_int n_steps, 
                  unsigned int dila_index1, unsigned int dila_index2, npy_uint32* boundary, npy_uint64 start_index, npy_uint32 border_dim);

