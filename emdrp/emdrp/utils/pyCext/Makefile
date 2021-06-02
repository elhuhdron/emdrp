
# compilers and flags
CC=gcc
CPPC=g++
CFLAGS=-fPIC -O3
CPPFLAGS=-fPIC -std=c++11 -O3

# Python include directory. This should contain the file Python.h, among others.
PYTHON_INCLUDE_PATH = $(HOME)/anaconda3/include/python3.8

# Numpy include directory. This should contain the file arrayobject.h, among others.
NUMPY_INCLUDE_PATH = $(HOME)/anaconda3/lib/python3.8/site-packages/numpy/core/include/numpy

# Python library directory and link version. This should contain libpythonx.x, among others.
PYTHON_LIBRARY_PATH = $(HOME)/anaconda3/lib
PYTHON_VERSION = 3.8

# ---- Link --------------------------- 
_pyCext.so _pyCppext.so:  pyCext.o pyCppext.o
	$(CC) -L$(PYTHON_LIBRARY_PATH) -shared pyCext.o -o _pyCext.so -lpython$(PYTHON_VERSION)
	$(CPPC) -L$(PYTHON_LIBRARY_PATH) -shared pyCppext.o -o _pyCppext.so -lpython$(PYTHON_VERSION)

# ---- Compile ------------------
pyCext.o:  pyCext.c pyCext.h
	$(CC) $(CFLAGS) -c pyCext.c -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH)

pyCppext.o:  pyCppext.cpp pyCppext.h
	$(CPPC) $(CPPFLAGS) -c pyCppext.cpp -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH)

clean:
	rm -rf *.o *.so *.pyc __pycache__

