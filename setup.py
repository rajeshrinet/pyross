import numpy
import os, sys, os.path, tempfile, subprocess, shutil
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True


setup(
    name='PyRoss',
    version='1.0.0',
    url='https://gitlab.com/rajeshrinet/pyross',
    author='The PyRoss team',
    license='MIT',
    description='python library for numerical simulation of infectious disease',
    long_description='pyross is a library for numerical simulation of infectious disease',
    platforms='works on all platforms (eg LINUX, macOS, and Microsoft Windows)',
    ext_modules=cythonize([ Extension("pyross/*", ["pyross/*.pyx"],
        include_dirs=[numpy.get_include()],
        )],
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    packages=['pyross'],
    package_data={'pyross': ['*.pxd']}
)


