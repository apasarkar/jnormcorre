API Documentation
=================

motion_correction Module
------------------------

frame_corrector Class
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: jnormcorre.motion_correction.frame_corrector
   :members: __init__, register_frames, rigid_function, pwrigid_function

MotionCorrect Class
^^^^^^^^^^^^^^^^^^^

.. autoclass:: jnormcorre.motion_correction.MotionCorrect
    :members: __init__, motion_correct

Rigid Motion Correction Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: jnormcorre.motion_correction.register_frames_to_template_rigid

.. autofunction:: jnormcorre.motion_correction.register_to_template_and_transfer_rigid

.. autofunction:: jnormcorre.motion_correction.register_frames_to_template_pwrigid

.. autofunction:: jnormcorre.motion_correction.register_to_template_and_transfer_pwrigid


jnormcorre.utils.lazy_array API
--------------------------------

.. autoclass:: jnormcorre.utils.lazy_array.lazy_data_loader
   :members:
   :show-inheritance:

   .. automethod:: dtype

   .. automethod:: shape

   .. automethod:: ndim

   .. automethod:: _compute_at_indices

   .. automethod:: __getitem__

jnormcorre.utils.registrationarrays
-----------------------------------

.. autoclass:: jnormcorre.utils.registrationarrays.TiffArray
   :members: __init__

.. autoclass:: jnormcorre.utils.registrationarrays.Hdf5Array
   :members: __init__

.. autoclass:: jnormcorre.utils.registrationarrays.RegistrationArray
   :members: __init__
