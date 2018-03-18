# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize
from setuptools import setup

ext_modules = [
    Extension(
        "cpu_nms",
        ["src/cpu_nms.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
        include_dirs=[np.get_include()]
    ),
]

setup(
    name='nms',
    ext_modules=cythonize(ext_modules),
)
