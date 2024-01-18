from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("Evaluation/module_192_6_v1.pyx",compiler_directives={'language_level': "3"})
    
)
