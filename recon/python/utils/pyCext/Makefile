
# Python include directory. This should contain the file Python.h, among others.
#PYTHON_INCLUDE_PATH = /usr/local/Anaconda/envs/anaconda_2.4.1_py35/include/python3.5m
PYTHON_INCLUDE_PATH = $(HOME)/anaconda3/include/python3.5m

# Python library directory and link version. This should contain libpythonx.x, among others.
#PYTHON_LIBRARY_PATH = /usr/local/Anaconda/envs/anaconda_2.4.1_py35/lib
PYTHON_LIBRARY_PATH = $(HOME)/anaconda3/lib
PYTHON_VERSION = 3.5m

# Numpy include directory. This should contain the file arrayobject.h, among others.
#NUMPY_INCLUDE_PATH = /usr/local/Anaconda/envs/anaconda_2.4.1_py35/lib/python3.5/site-packages/numpy/core/include/numpy
NUMPY_INCLUDE_PATH = $(HOME)/anaconda3/lib/python3.5/site-packages/numpy/core/include/numpy

# ---- Link --------------------------- 
_pyCext.so:  pyCext.o 
	gcc -L$(PYTHON_LIBRARY_PATH) -shared pyCext.o -o _pyCext.so -lpython$(PYTHON_VERSION)

# ---- gcc C compile ------------------
pyCext.o:  pyCext.c pyCext.h
	gcc  -c -fPIC pyCext.c -O3 -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH)

clean:
	rm -rf *.o *.so *.pyc

