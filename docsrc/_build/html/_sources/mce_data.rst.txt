mce_data submodule
==================

The :mod:`.mce_data` (from `the MCE wiki <https://e-mode.phas.ubc.ca/mcewiki/index.php/Python_data_and_runfile_modules>`_) is a low-level interface for reading the MCE binary files. In our case these are files like ``saturn_191130_0030`` with no file extension. 

The library has been modified in two important ways: First, it had a hard limit on file size---it would refuse to open files larger than a certain size---which I have exceeded in the past, so I modified it to accept larger files. I also modified it to run on Python 3.

mce_data module
---------------------

.. automodule:: zeustools.mce_data
   :members:
   :undoc-members:
   :show-inheritance: