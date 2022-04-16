### jnormcorre
This is a Jax-accelerated implementation of normcorre. 

## Installation
This software is currently only tested on linux/unix systems, though wider usage may be possible. 

Todo: Add Info on installation via conda/mamba

# Conda installation for developers (ubuntu)
First install mamba: 

```
conda install -c conda-forge mamba
```

Next, navigate to the root of this repository and run the following command: 
```
mamba env update -f environments/devel_ubuntu.yml
```

This will create a conda environment, "jnormcorre_devel", which will contain all dependencies. 

## Use Case
This implementation is for offline motion correction only. The code is currently set up to use multicore processing via python multiprocessing. GPU/TPU parallelization support will be provided soon. 

## Citations
- Eftychios A. Pnevmatikakis and Andrea Giovannucci, NoRMCorre: An online algorithm for piecewise rigid motion correction of calcium imaging data, Journal of Neuroscience Methods, vol. 291, pp 83-94, 2017; doi: https://doi.org/10.1016/j.jneumeth.2017.07.031

- Matlab Implementation of Normcorre: https://github.com/flatironinstitute/NoRMCorre#ref

- Python Implementation of Normcorre: https://github.com/flatironinstitute/CaImAn




## License
See License.txt for the details of the GPL license used here. 
