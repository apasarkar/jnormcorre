import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="jnormcorre",
    version="0.0.6",
    description="Jax-accelerated implementation of normcorre",
    packages=setuptools.find_packages(),
    install_requires=["future","numpy", "scipy", "h5py", "tqdm", "matplotlib", "opencv-python", "tifffile", "typing", "torch", "pynwb", "pillow", "scikit-image", "jax", "jaxlib"],
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
)
