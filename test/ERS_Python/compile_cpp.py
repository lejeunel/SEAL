from distutils.core import setup, Extension
import shutil
import glob
import os

module = Extension('ERSModule',
                    sources = ['erspy.cpp', 'MERCCInput.cpp', 'MERCOutput.cpp', 'MERCDisjointSet.cpp', 'MERCFunctions.cpp', 'MERCLazyGreedy.cpp'])

setup(name = 'PackageName', 
      version = '1.0',
      description = 'This is a Python wrapper for ERS',
      ext_modules = [module])

lib_dirname = glob.glob('./build/lib.*')[0]
so_fname = glob.glob(os.path.join(lib_dirname, 'ERSModule*.so'))[0]
      
shutil.copy2(so_fname, 'ERSModule.so')
