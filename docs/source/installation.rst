.. Installation instruction for SOPHT

Installation
============

This page walks you through the installation steps for ``SOPHT``.

Linux / Mac (Intel)
-------------------------

Step 1: Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You have the option to directly clone the public ``SOPHT`` repository
or create your own fork.

.. code-block:: console

    git clone https://github.com/SophT-Team/SophT.git

Step 2. Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using a python virtual environment, such as ``conda``,
``pyenv``, or ``venv``.
The following lines create and activate a ``conda`` environment called
``sopht-env``.

.. code-block:: console

    conda create --name sopht-env -y
    conda activate sopht-env

Step 3: Install the dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start with, install pip with ``python 3.10``.

.. code-block:: console

    conda install python=3.10

``SOPHT`` uses `poetry`_ as the designated dependency manager.
To install poetry, navigate to the root directory of ``SophT-Simulator`` and
do

.. code-block:: console

    make poetry-download

To test whether ``poetry`` is properly installed, do

.. code-block:: console

    poetry --version

After ``poetry`` is installed, we can proceed to install the rest of
the dependencies in the root directory

.. code-block:: console

    make install
    make pre-commit-install

Mac (ARM64)
-----------

To install ``SOPHT`` on Macs with ARM64 architecture (M1/M2), follow
the same steps as `Linux / Mac (Intel)`_, and append the following
commands

.. code-block:: console

    pip uninstall pyfftw
    conda install -c conda-forge pyfftw
    conda install numba

.. _poetry: https://python-poetry.org/
