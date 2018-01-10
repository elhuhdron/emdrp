
# compilers and flags
CC=g++
CCFLAGS=-fPIC -g -std=c++11 -O3

# Python include directory. This should contain the file Python.h, among others.
PYTHON_INCLUDE_PATH = $(HOME)/anaconda3/include/python3.6m

# Numpy include directory. This should contain the file arrayobject.h, among others.
NUMPY_INCLUDE_PATH = $(HOME)/anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy

# Python library directory and link version. This should contain libpythonx.x, among others.
PYTHON_LIBRARY_PATH = $(HOME)/anaconda3/lib
PYTHON_VERSION = 3.6m

# ---- Link --------------------------- 
_pyCext.so:  pyCext.o pyCppext.o
	$(CC) -L$(PYTHON_LIBRARY_PATH) -shared pyCext.o pyCppext.o -o _pyCext.so -lpython$(PYTHON_VERSION)
	#install_name_tool -change libpython$(PYTHON_VERSION).dylib $(PYTHON_LIBRARY_PATH)/libpython$(PYTHON_VERSION).dylib _pyCext.so

# ---- Compile ------------------
pyCext.o:  pyCext.c pyCext.h
	$(CC) $(CCFLAGS) -c pyCext.c -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH)

pyCppext.o:  pyCppext.cpp pyCext.h
	$(CC) $(CCFLAGS) -c pyCppext.cpp -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH)

clean:
	rm -rf *.o *.so *.pyc

