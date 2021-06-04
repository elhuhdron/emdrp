
#from distutils.core import setup, Extension
from setuptools import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs


include_dirs = get_numpy_include_dirs()
include_dirs[0] = include_dirs[0] + '/numpy'

pyCext_module = Extension('_pyCext',
                    sources = ['emdrp/emdrp/utils/pyCext/pyCext.c'],
                    extra_compile_args = ['-O3'],
                    include_dirs=include_dirs,
                    )

pyCppext_module = Extension('_pyCppext',
                    sources = ['emdrp/emdrp/utils/pyCext/pyCppext.cpp'],
                    extra_compile_args = ['-O3', '-std=c++11'],
                    include_dirs=include_dirs,
                    )


setup (name = 'emdrp',
       version = '0.1',
       author = 'Paul Watkins',
       author_email = 'pwatkins@gmail.com',
       url = 'https://github.com/elhuhdron/emdrp',
       license='MIT',
       packages=['emdrp'],
       package_dir={'': 'emdrp'},
       ext_modules=[
            pyCext_module,
            pyCppext_module
        ],
)
