LIB = lib/

# compilers and flags
CC=g++
NVCC=nvcc

NVCCFLAGS=-Xcompiler -fPIC -std=c++11 -DCUCC_ID="nVidia" -DBLD_PROCEDURE -DWITH_GSL -DHAVE_INLINE -DWITH_CUDA -DWITH_MPI -DPARALLEL -DWITH_PARALLEL_IO 
#NVCCFLAGS=--compiler-bindir $(CC) -fPIC

CUDA_ARCH= -g -G -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_52,code=sm_52
CCFLAGS= -fPIC -g -std=c++11

CUDA_INSTALL_PATH=/usr/local/cuda-8.0
CUDA_INC_PATH= $(CUDA_INSTALL_PATH)/include 
CUDA_LIB_PATH= $(CUDA_INSTALL_PATH)/lib64 
#CUB_INC_PATH = /home/patilra/cub-1.7.0/

#Python include directory. This should contain the file Python.h, among others.
PYTHON_INCLUDE_PATH = $(HOME)/anaconda3/include/python3.5m

# Numpy include directory. This should contain the file arrayobject.h, among others.
NUMPY_INCLUDE_PATH = $(HOME)/anaconda3/lib/python3.5/site-packages/numpy/core/include/numpy

# Python library directory and link version. This should contain libpythonx.x, among others.
PYTHON_LIBRARY_PATH = $(HOME)/anaconda3/lib #-L$(HOME)/gits/emdrp/recon/python/cudaFrag/parallel-cuda/lib
PYTHON_VERSION = 3.5m
EXT_LIB = _FRAG_extension.so
CUDA_OBJS= wrapper_createRag kernel_createFrag
CPP_OBJS= FRAG_Extension
CUDA_OBJECTS=$(addsuffix .o,$(addprefix $(LIB),$(CUDA_OBJS)))
CPP_OBJECTS=$(addsuffix .o, $(addprefix $(LIB),$(CPP_OBJS)))
CUDA_LIB=libCuda.a 
all: $(EXT_LIB)

# ---- C++ compile ------------------
$(CPP_OBJECTS): $(LIB)%.o:%.cpp %.h
	$(CC) -I./ -I$(CUDA_INC_PATH) -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) $(CCFLAGS) -O3 -c $< -o $@     

$(CUDA_OBJECTS): $(LIB)%.o:%.cu %.h
	$(NVCC) -I./ -I$(PYTHON_INCLUDE_PATH) -I$(NUMPY_INCLUDE_PATH) $(NVCCFLAGS) $(CUDA_ARCH) -c $< -o $@ 	


$(EXT_LIB): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(CC) --shared $(CPP_OBJECTS) $(CUDA_OBJECTS) -o $(EXT_LIB) -L$(CUDA_LIB_PATH) -L$(PYTHON_LIBRARY_PATH) -lpython$(PYTHON_VERSION) -lcudart
	#install_name_tool -change libpython$(PYTHON_VERSION).dylib $(PYTHON_LIBRARY_PATH)/libpython$(PYTHON_VERSION).dylib _rag_Extension.so




.PHONY: clean
clean:
	rm -rf $(CUDA_OBJECTS) $(CPP_OBJECTS) ./*.so
