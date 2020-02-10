Welcome to zeustools's documentation!
=====================================

ZEUStools is designed to make our lives easier in terms of loading and processing ZEUS-2 data. It is organized into several different submodules.

The main utility right now is the Calibration Pipeline, which is a script that loads the output of Bo's pipeline and produces spectra and data files in physical units. It can be run as a commandline script, and requires an INI file containing settings it should use.

There is also a library of general functions.

The :py:mod:`zeustools.bpio` submodule is designed for reading in the outputs of Bo's data reduction pipeline. 

The :py:mod:`zeustools.calibration` submodule contains functions for determining the real physical properties of our data.

The :py:mod:`zeustools.codystools` submodule contains the most useful functions designed by Cody, notably :func:`zeustools.codystools.createModelSnakeEntireArray`, which smooths the time-stream in Fourier space.

The :py:mod:`zeustools.rooneystools` submodule includes a myriad of functions Christopher thought were useful. Most useful in this category are the :py:func:`zeustools.rooneystools.nd_reject_outliers` function, which uses median absolute deviation to reject outliers in the time stream; the :class:`zeustools.rooneystools.ArrayMapper` class, which lets you input a physical detector position (spectral, spatial) and return that detector's logical position (mce_row, mce_column); and the :func:`zeustools.rooneystools.makeFileName` function, which will do all the date logic for you to return the correct filename, e.g. ``20191130/saturn_191130_0010``.

The :mod:`zeustools.pointing` module includes useful routines for use at APEX, including functions that process and plot raster pointing data, and automatically output the APECS calibration commands needed to properly respond to the pointing results.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   calibpipeline
   zeustools




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
