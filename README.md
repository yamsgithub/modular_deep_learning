Gated Modular Deep Learning
=======================================================================================================

This repository contains various experiments to understand and improve interpertability in gated modular neural networks. Currently I am using the various Mixture of Experts architecture models listed below for these experiments: 

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


RUNNING JUPYTER NOTEBOOK for WORKSHOP EXPERIMENTS
------------------------

1. Run the following script to start jupyter: 

   `./bin/run_notebooks.sh`

2. In the jupyter lab go to the notebooks folder which contains all the relevant notebooks 

3. Start with the toy_classification.ipynb.

4. Select the mnn kernel.

5. You should now be able to run the notebooks.

Contact
-------

For any questions or issues email: yamuna dot krishnamurthy at rhul.ac.uk


