Misc. Submodules
==================


mce_data module
---------------------

The :mod:`.mce_data` (from `the MCE wiki <https://e-mode.phas.ubc.ca/mcewiki/index.php/Python_data_and_runfile_modules>`_) is a low-level interface for reading the MCE binary files. In our case these are files like ``saturn_191130_0030`` with no file extension. 

The library has been modified in two important ways: First, it had a hard limit on file size---it would refuse to open files larger than a certain size---which I have exceeded in the past, so I modified it to accept larger files. I also modified it to run on Python 3. Now it also supports reading in the chop data that is embedded in the MCE data files!

.. automodule:: zeustools.mce_data
   :members:
   :undoc-members:
   :show-inheritance:


leapseconds module
---------------------
The :mod:`.leapseconds` (from `this GitHub Gist <https://gist.github.com/zed/92df922103ac9deb1a05>`_) module provides helper functions for converting between GPS time and UTC time. This can be important because .ts files are written with GPS timestamps.

.. automodule:: zeustools.leapseconds
   :members:
   :undoc-members:
   :show-inheritance: