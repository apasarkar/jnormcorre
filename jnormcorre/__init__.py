from .utils.lazy_array import lazy_data_loader
from .utils.registrationarrays import (
    RegistrationArray,
    TiffArray,
    Hdf5Array,
    FilteredArray,
)
from jnormcorre.motion_correction import (
    FrameCorrector,
    MotionCorrect,
    register_frames_to_template_rigid,
    register_frames_to_template_pwrigid,
    register_to_template_and_transfer_rigid,
    register_to_template_and_transfer_pwrigid,
)
from . import spatial_filters
