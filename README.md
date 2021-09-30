Modular Deep Learning
=======================================================================================================

Code for realizing 4 different MoE architectures.
1. Expectation Model
2. Stochastic Model
3. Pre-softmax Model
4. EM Model

REQUIREMENTS
------------

1. ``Python 3.7`` 
2. ``Pytorch 1.6.0, optionally with Cuda 10.1`` 
3. Linux Operating System. It has been tested on Ubuntu and MacOS. 
4. Additional modules listed in ``requirements.txt``

INSTALLATION 
------------

In order to install the code locally please follow the steps below:

1. Clone this repository and go to the cloned directory.

2. Set the environment variable to point to your python executable:

   `export PYTHON=<path to python 3.7 executable>`

3. Run the following command to set up the environment:

   `make` on **Linux/Mac**

4. Activate the environment by running:

   `source mnn/bin/activate` on **Linux/Mac**


RUNNING JUPYTER NOTEBOOK
------------------------

1. Go to the notebooks directory

2. Run jupyter notebook (the mnn env is already loaded in the ipykernel) to access all the notebooks:

   `jupyter lab`

2. Start with the toy_classification.ipynb.

3. Select the mnn kernel.

4. You should now be able to run the notebooks.

