
==========================
Installation
==========================

Details regarding installation of IsaacGym can be found `here <https://developer.nvidia.com/isaac-gym>`_. **We currently support the `Preview Release 3` version of IsaacGym.**

Prerequisites
=========================

- Ubuntu 18.04 or 20.04.
- Python 3.6, 3.7 or 3.8.
- Minimum NVIDIA driver version:

  + Linux: 470

Setup the Python Packages
=========================

The code has been tested on Ubuntu 18.04 with Python 3.7. The minimum recommended NVIDIA driver
version for Linux is `470` (dictated by support of IsaacGym).

It uses `Anaconda <https://www.anaconda.com/>`_ to create virtual environments.
To install Anaconda, follow instructions `anaconda install <https://docs.anaconda.com/anaconda/install/linux/>`_ .

Ensure that Isaac Gym works on your system by running one of the examples from the ``python/examples`` 
directory, like ``joint_monkey.py``. Follow troubleshooting steps described in the Isaac Gym Preview 3 
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

.. code-block:: bash

    pip install -e .


