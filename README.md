# pytorch-detection

## Introduction
I wanted to have an object-detection framework based on PyTorch, but couldn't find anything good enough.
The closest I could find was [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch),
but I didn't like quite a few design decisions they made (such as wanting to support python2),
so I decided to make my own.

## Installation
This project uses [PyTorch](http://pytorch.org/) 3.1 and Python 3.6

I recommend using virtual environment to make sure correct versions of each package are installed.
I made a script which completes all steps required for the installation, run it with:
```bash
bash make.sh
```

### Dependecies
Dependencies are listed in `requirements.txt`, to install with `pip` use:
```bash
pip install -r requirements.txt
```

### Cython extensions
You also need to complile and install Cython extensions:
```bash
cd utils/nms
python setup.py install
cd ../..
```
