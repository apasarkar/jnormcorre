### jnormcorre
This is a Jax-accelerated implementation of the normcorre motion correction algorithm.
It allows you to motion correct neuroimaging data (calcium, voltage imaging, etc.) 
as fast as possible on GPUs/TPUs (also works well on CPUs).

See the following documentation for algorithm/parameter details, common use cases, and API info.  

## Citations

If you use this method, please cite the accompanying [paper](https://www.biorxiv.org/content/10.1101/2023.09.14.557777v1)

> _maskNMF: A denoise-sparsen-detect approach for extracting neural signals from dense imaging data_. (2023). A. Pasarkar\*, I. Kinsella, P. Zhou, M. Wu, D. Pan, J.L. Fan, Z. Wang, L. Abdeladim, D.S. Peterka, H. Adesnik, N. Ji, L. Paninski.


## Installation
Currently, this code is supported for Linux operating systems with python version at least 3.8. We primarily use [JAX](https://github.com/google/jax) for fast CPU/GPU/TPU execution and Just-In-Time compilation; see the Google installation instructions on how to install the appropriate version of JAX and JAXLIB for your specific hardware system. We require: 

```
python>=3.8
jax>=0.3.25
jaxlib>=0.3.25
```

To install this repository into your python environment from the source code, do the following (this approach is recommended for now): 
```
#Step 1: Install the appropriate version of jax for your hardware system 

#Step 2: Run the below lines of code
git clone https://github.com/apasarkar/jnormcorre.git
cd jnormcorre
pip install -e .
```

To install the most recently published version from PyPI, you can do: 

```
#Step 1: Install the appropriate version of jax for your hardware system 

#Step 2: Run below line
pip install jnormcorre
```

The package on PyPI comes with jax, but doing the installation in this order will allow you to control whether your version of jax is GPU/TPU compatible for your system. If you are only running on CPU, you can just skip to step 2. 


These are the original implementations:  
- Matlab Implementation of Normcorre: https://github.com/flatironinstitute/NoRMCorre#ref

- Python Implementation of Normcorre: https://github.com/flatironinstitute/CaImAn

## License
See License.txt for the details of the GPL license used here. 