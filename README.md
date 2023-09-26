### jnormcorre
This is a Jax-accelerated implementation of normcorre. 

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

The package on PyPI comes with jax, but doing the install in this order will allow you to control whether your version of jax is GPU/TPU compatible for your system. If you are only running on CPU, you can just skip to step 2. 


## Use Cases
This implementation can support both online and offline motion correction use cases. Here are some common ones: 

1. **Offline:** Given a full video, write out the motion corrected video as a new file.

2. **Offline + PMD Compression&Denoising**: Given a full video, estimate templates, and then perform on-the-fly registration (rigid and/or piecewise rigid) + compression. No need to save an intermediate registered movie - instead, reduce your data size by ~2 orders of magnitude and use this for downstream processing tasks. This is hyper efficient thanks to jax's composability mechanisms, and useful for large data analysis, online experiments and more. This setup is implemented in [PMD](https://github.com/apasarkar/localmd) and in the full [maskNMF](https://github.com/apasarkar/masknmf_full_pipeline) functional imaging analysis pipeline. 

3. **Online**: jnormcorre can be set up to adaptively estimate a template, and then take a single pass through a stream of new data, registering all newly observed frames and updating the template in the process

## Citations

If you use this method, please cite the accompanying [paper](https://www.biorxiv.org/content/10.1101/2023.09.14.557777v1)

> _maskNMF: A denoise-sparsen-detect approach for extracting neural signals from dense imaging data_. (2023). A. Pasarkar\*, I. Kinsella, P. Zhou, M. Wu, D. Pan, J.L. Fan, Z. Wang, L. Abdeladim, D.S. Peterka, H. Adesnik, N. Ji, L. Paninski.

These are the original implementations:  
- Matlab Implementation of Normcorre: https://github.com/flatironinstitute/NoRMCorre#ref

- Python Implementation of Normcorre: https://github.com/flatironinstitute/CaImAn

## License
See License.txt for the details of the GPL license used here. 