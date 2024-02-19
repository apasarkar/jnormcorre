.. Algo

Algorithm Overview
==================

Jnormcorre implements a template-based motion correction algorithm from
normcorre.
It operates on 3D data (Time, X, Y). At a high level, it
works by:

1. **Estimating a template for the video.**
2. **Aligning various frames to this template.** This is done using either a rigid or piecewise rigid motion correction method (see below).
3. **Updating the template from step 1.**

Steps 1 - 3 are repeated iteratively across the frames of the movie.


Rigid Motion Correction
=======================
For each frame, we apply the optimal rigid shift
(horizontal shift of "x" units and vertical shift of "y" units) that best aligns the template and frame.

Key Parameters:
1. max_shifts
2. frames_per_split
3. num_splits_to_process_rig
4. niter_rig


Piecewise Rigid Motion Correction
=================================
Here we break each frame into overlapping square patches. We perform rigid motion correction between
each of these square patches and the corresponding region on the template. We then
"stitch" the alignment results across all patches together via interpolation.

Key Parameters:
1. pw_rigid
2. strides
3. overlaps
4. max_deviation_rigid
5. frames_per_split
6. num_splits_to_process_els
7. niter_els

Template Updates
================

Movie is partitioned into temporal "chunks" of, say, 1K frames. We run the rigid and/or
piecewise rigid motion correction algorithms on each chunk. For each chunk, we get a
local template (computed from the motion corrected results).

We pool all these templates together to create a global template by taking
the pixelwise median across all templates.


1p Processing
=============
Some modalities like 1-photon imaging have significant background (neuropil) contamination.
The spatially smooth background overpowers the signals, and this affects the template.

For this kind of data, we follow the modified pipeline:
(1) Spatially high-pass filter the template
(2) Spatially high-pass filter each frame
(3) Estimate the optimal shifts between 1 and 2.
(4) Apply the shifts to the "unfiltered" frames to register the data.

Steps (1) and (2) enhance the salient parts of the signal while suppressing the background.
See the figure for an example of this.

Key Parameters:
1. gSig_filt


Supported Data Formats
======================
This repo supports 2D imaging videos (Frames, X, Y).
Any data loader which implements the simple "lazy_array" interface; see (jnormcorre.utils.lazy_array).
works here. This is really a basic array-like interface, so numpy arrays, etc. are automatically
compatible. Tiff and HDF5 files are also currently supported.