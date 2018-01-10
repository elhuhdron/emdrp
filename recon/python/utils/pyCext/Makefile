
# compilers and flags
CC=gcc
CPPC=g++
CFLAGS=-fPIC -g -O3
CPPFLAGS=-fPIC -g -std=c++11 -O3

# Python include directory. This should contain the file Python.h, among others.
PYTHON_INCLUDE_PATH = $(HOME)/anaconda3/include/python3.6m

# Numpy include directory. This should contain the file arrayobject.h, among others.
NUMPY_INCLUDE_PATH = $(HOME)/anaconda3/lib/python3.6/site-packages/numpy/core/include/numpy

# Python library directory and link version. This should contain libpythonx.x, among others.
PYTHON_LIBRARY_PATH = $(HOME)/anaconda3/lib
PYTHON_VERSION = 3.6m

# ---- Link --------------------------- 
_pyCext.so:  pyCext.o pyCppext.o
	$(CPPC) -L$(PYTHON_LIBRARY_PATH) -shared pyCext.o pyCppext.o -o _pyCext.so -lpython$(PYTHON_VERSION)
	#install_name_tool -change libpython$(PYTHON_VERSION).dylib $(PYTHON_LIBRARY_PATH)/libpython$(PYTHON_VERSION).dylib _pyCext.so

# ---- Compile ------------------
pyCext.o:  pyCext.c pyCext.h
	$(CC) $(CFLAGS) -c pyCext.c -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH)

pyCppext.o:  pyCppext.cpp pyCext.h
	$(CPPC) $(CPPFLAGS) -c pyCppext.cpp -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH)

clean:
	rm -rf *.o *.so *.pyc

