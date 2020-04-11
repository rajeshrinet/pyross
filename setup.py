import numpy
import os, sys, os.path, tempfile, subprocess, shutil
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True


def checkOpenmpSupport():
    """ Adapted from https://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
    """ 
    ompTest = \
    r"""
    #include <omp.h>
    #include <stdio.h>
    int main() {
    #pragma omp parallel
    printf("Thread %d, Total number of threads %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    """
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r'test.c'
    try:
        with open(filename, 'w') as file:
            file.write(ompTest)
        with open(os.devnull, 'w') as fnull:
            result = subprocess.call(['cc', '-fopenmp', filename],
                                     stdout=fnull, stderr=fnull)
    except:
        print("Failed to test for OpenMP support. Assuming unavailability");
        result = -1;
    os.chdir(curdir)
    shutil.rmtree(tmpdir) 
    if result == 0:
        return True
    else:
        return False


if checkOpenmpSupport() == True:
    ompArgs = ['-fopenmp']
else:
    ompArgs = None 




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
        extra_compile_args=ompArgs,
        extra_link_args=ompArgs 
        )],
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    packages=['pyross'],
    package_data={'pyross': ['*.pxd']}
)


