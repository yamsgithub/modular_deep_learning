Neurips XAI Workshop 2021 Code Submission for Paper "Interpretability in Gated Modular Neural Networks"
=======================================================================================================

Code for reproducing results in the paper

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

   `source xai2021/bin/activate` on **Linux/Mac**


RUNNING JUPYTER NOTEBOOK
------------------------

1. Run jupyter notebook (the xai2021 env is already loaded in the ipykernel):

   `jupyter lab`

2. In the jupyter lab open the toy_classification.ipynb and mnist_classification.ipynb notebooks which contain the all the experiments from the paper.

3. Select the xai2021 kernel.

4. You should now be able to run the notebooks.

