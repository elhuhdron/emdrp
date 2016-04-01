
# Python include directory. This should contain the file Python.h, among others.
#PYTHON_INCLUDE_PATH = /usr/include/python2.6
#PYTHON_INCLUDE_PATH = /usr/include/python2.7
PYTHON_INCLUDE_PATH = $(HOME)/anaconda3/include/python3.5m

# Python library directory and link version. This should contain libpythonx.x, among others.
PYTHON_LIBRARY_PATH = $(HOME)/anaconda3/lib
PYTHON_VERSION = 3.5m

# Numpy include directory. This should contain the file arrayobject.h, among others.
#NUMPY_INCLUDE_PATH = /usr/lib64/python2.6/site-packages/numpy/core/include/numpy
#NUMPY_INCLUDE_PATH = /Library/Python/2.7/site-packages/numpy/core/include/numpy
NUMPY_INCLUDE_PATH = $(HOME)/anaconda3/lib/python3.5/site-packages/numpy/core/include/numpy

# ---- Link --------------------------- 
_pyCext.so:  pyCext.o 
	gcc -L$(PYTHON_LIBRARY_PATH) -shared -lpython$(PYTHON_VERSION) pyCext.o -o _pyCext.so 
	#install_name_tool -change libpython3.5m.dylib ~/anaconda3/lib/libpython3.5m.dylib _pyCext.so

# ---- gcc C compile ------------------
pyCext.o:  pyCext.c pyCext.h
	gcc  -c -fPIC pyCext.c -O4 -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH)

clean:
	rm -rf *.o *.so *.pyc

