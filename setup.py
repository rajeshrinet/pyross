import numpy
import os, sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate=True


if 'darwin'==(sys.platform).lower():
    extension = Extension('pyross/*', ['pyross/*.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-mmacosx-version-min=10.9'],
        extra_link_args=['-mmacosx-version-min=10.9'],
    )
else:
    extension = Extension('pyross/*', ['pyross/*.pyx'],
        include_dirs=[numpy.get_include()],
    )


with open('requirements.txt', 'r') as rm:
    reqs = [l.strip() for l in rm]


setup(
    name='pyross',
    version='1.2.0',
    url='https://github.com/rajeshrinet/pyross',
    author='The PyRoss team',
    author_email = 'pyross@googlegroups.com',
    license='MIT',
    description='Infectious disease models in Python: inference, prediction and NPI',
    long_description='PyRoss is a numerical library that offers an integrated platform for \
                      inference, prediction and non-pharmaceutical interventions in \
                      age- and contact-structured epidemiological compartment models.',
    platforms='works on LINUX and macOS',
    ext_modules=cythonize([ extension ],
        compiler_directives={'language_level': sys.version_info[0]},
        ),
    libraries=[],
    packages=['pyross'],
    install_requires=reqs,
    package_data={'pyross': ['*.pxd']},
)
