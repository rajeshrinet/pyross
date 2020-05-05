#from setuptools import setup, find_packages, Extension
#from setuptools import find_packages

import numpy
import os, sys 
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths
extra_files = package_files('pyross/data')

setup(
    name='PyRoss',
    version='1.0.0',
    url='https://gitlab.com/rajeshrinet/pyross',
    author='The PyRoss team',
    license='MIT',
    description='python library for numerical simulation of infectious disease',
    long_description='pyross is a library for numerical simulation of infectious disease',
    platforms='works on all platforms (such as LINUX, macOS, and Microsoft Windows)',
    ext_modules=cythonize([ Extension("pyross/*", ["pyross/*.pyx"],
        include_dirs=[numpy.get_include()],
        )],
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    #packages = find_packages(),
    package_data={'': [extra_files],'pyross': ['*.pxd']},
    #include_package_data = True
#    packages=['pyross'],
)
