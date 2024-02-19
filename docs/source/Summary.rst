.. Summary

Summary
=======

jnormcorre is a jax-accelerated implementation of the `normcorre <https://www.sciencedirect.com/science/article/pii/S0165027017302753>`_ motion correction
algorithm. The motivation for designing this tool is to allow users to
process big neuroimaging data -- large FOV, high frame rate -- as efficiently
as possible on a variety of platforms (CPUs/GPUs/TPUs). By leveraging just-in-time
compilation and vectorization, jnormcorre allows for much faster processing
of neuroimaging data on any platform, but is especially fast on GPUs/TPUs.


**Use Cases**

Here are some of the common use cases supported by jnormcorre.

- **End-to-end offline analysis**: You've collected a dataset and want to motion correct and save out the results as a new file.

- **Custom template alignment**: You have a template already and want to align data to this template. Maybe you computed this template using existing functional imaging data (i.e. data from early in the imaging session) or using different info altogether (i.e. structural imaging data).

- **Real-time alignment**: You have a template and want to align data to this template in real time.

- **Fused Motion Correction and Downstream Analysis**: You want to motion correct frames of data on the fly and apply downstream data processing to it.





