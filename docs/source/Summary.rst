.. Summary

jnormcorre
==========

jnormcorre is a jax-accelerated implementation of the `normcorre <https://www.sciencedirect.com/science/article/pii/S0165027017302753>`_ motion correction
algorithm. The motivation for designing this tool is to allow users to
process big neuroimaging data -- large FOV, high frame rate -- as efficiently
as possible on a variety of platforms (CPUs/GPUs/TPUs). By leveraging just-in-time
compilation and vectorization, jnormcorre allows for much faster processing
of neuroimaging data on any platform, but is especially fast on GPUs/TPUs.

This software accompanies the following paper: `maskNMF <https://www.biorxiv.org/content/10.1101/2023.09.14.557777v1.full.pdf>`_

Use Cases
=========

Here are some of the common use cases/features supported by jnormcorre.

- **End-to-end offline analysis**:
    You've collected a dataset and want to motion correct and
    save out the results as a new file.


.. code-block:: console

    corrector = jnormcorre.motion_correction.MotionCorrect(lazy_dataset, max_shifts=max_shifts, frames_per_split=frames_per_split,
                                                    num_splits_to_process_rig=num_splits_to_process_rig,
                                                    niter_rig=niter_rig, pw_rigid=pw_rigid, strides=strides,
                                                    overlaps=overlaps, max_deviation_rigid=max_deviation_rigid,
                                                    num_splits_to_process_els=num_splits_to_process_els, min_mov=min_mov,
                                                    gSig_filt=gSig_filt)

    frame_corrector, output_file = corrector.motion_correct(
        template=template, save_movie=save_movie
    )

- **Custom template alignment**:
    You have a good template already and want to align data to this template. Maybe you computed this template using existing functional imaging data (i.e. data from early in the imaging session) or using different info altogether (i.e. structural imaging data).

.. code-block:: console

    #Construct registration object with your template
    frame_corrector_object = jnormcorre.motion_correction.frame_corrector(template, max_shifts,
                                            strides=strides, overlaps=overlaps,
                                                    max_deviation_rigid, min_mov=0)

    #Run registration on your_data
    registration_obj.register_frames(your_data, pw_rigid=pw_rigid)

- **Real-time alignment**:
    You have a good template and want to align data to this template in real time.
    The above code block also supports this case.




