#!/usr/bin/env bash

cd src
echo "Compiling stnm kernels by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ..

# TODO: Combine build.py and setup.py
python build.py

# Cythonise cpu nms
python setup.py install
