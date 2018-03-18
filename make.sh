#!/usr/bin/env bash

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 \
	       -gencode arch=compute_70,code=sm_70 "

# Install dependecies
pip install -r requirements.txt


# Cythonise extensions
cd utils/nms
bash make.sh
cd ../..

# Compile RoI pooling
cd models/roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py
cd ../..

# Run tests
python -m unittest
