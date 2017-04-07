# compilers and flags
CC=g++
CCFLAGS=-fPIC -g -std=c++11

# CUDA include Directory
CUDA_INSTALL_PATH=/usr/local/cuda
CUDA_INC_PATH= $(CUDA_INSTALL_PATH)/include
CUDA_LIB_PATH= $(CUDA_INSTALL_PATH)/lib64

#Python include directory. This should contain the file Python.h, among others.
PYTHON_INCLUDE_PATH = $(HOME)/anaconda3/include/python3.5m

# Numpy include directory. This should contain the file arrayobject.h, among others.
NUMPY_INCLUDE_PATH = $(HOME)/anaconda3/lib/python3.5/site-packages/numpy/core/include/numpy

# Python library directory and link version. This should contain libpythonx.x, among others.
PYTHON_LIBRARY_PATH = $(HOME)/anaconda3/lib
PYTHON_VERSION = 3.5m

# ---- Link --------------------------- 
_FRAG_extension.so: FRAG_extension.o
	$(CC) -L$(CUDA_LIB_PATH) -L$(PYTHON_LIBRARY_PATH) -shared FRAG_extension.o -o _FRAG_extension.so -lpython$(PYTHON_VERSION) -lcudart
	#install_name_tool -change libpython$(PYTHON_VERSION).dylib $(PYTHON_LIBRARY_PATH)/libpython$(PYTHON_VERSION).dylib _rag_Extension.so

# ---- C++ compile ------------------
FRAG_extension.o: FRAG_extension.cpp 
	$(CC) $(CCFLAGS) -c FRAG_extension.cpp -O3 -I$(CUDA_INC_PATH) -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH)

clean:
	rm -rf *.o *.so *.pyc                                                            
