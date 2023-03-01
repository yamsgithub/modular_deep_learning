Gated Modular Deep Learning
=======================================================================================================

This repository contains various experiments to search for good task decompositions in Mixture of Experts (MoE). Good task decompositions enable interpertability and transferability in gated modular neural networks. 

REQUIREMENTS
------------

1. ``Python 3.9`` 
2. ``Pytorch 1.10.1, optionally with Cuda 11.2`` 
3. Linux Operating System. It has been tested on Ubuntu and MacOS. 
4. Additional modules listed in ``requirements.txt``

INSTALLATION 
------------

In order to install the code locally please follow the steps below:

1. Clone this repository and go to the cloned directory.

2. Set the environment variable to point to your python executable:

   `export PYTHON=<path to python 3.9 executable>`

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

Publication
-------
Yamuna Krishnamurthy and Chris Watkins, [Interpretability in Gated Modular Neural Networks](https://xai4debugging.github.io/files/papers/interpretability_in_gated_modu.pdf), in Explainable AI approaches for debugging and diagnosis Workshop at Neural Information Processing (NeurIPS), Dec 2021


