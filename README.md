Gated Modular Deep Learning
=======================================================================================================

This repository contains implementations of different Mixture of Experts (MoE) models both existing and novel architectures and methods of training. MoE is a gated modular deep neural network architecture, shown in Figure 1. It consists of simple individual neural network modules called experts and another simple neural network called gate. The gate allocates samples to the experts during training and selects the expert specifialized for a sample during inference. The output of the MoE is some combination of the outputs of the individual experts. The experts and gate are usually trained end-to-end. Since ideally experts specialise in samples, during inference, for each sample only a few experts need to be evaluated and updated. This is called <em> conditional computation</em>.

![](figures/moe_expectation.png)
*Figure 1 Original MoE architecture with 3 experts and 1 gate. The output of the model is the expected sum of the outputs of the individual experts*

The goal of the various MoE models is to search for good and clean task decompositions among the experts. Good task decompositions enable interpertability and transferability in gated modular neural networks.




Requirements
------------

1. ``Python 3.9`` 
2. ``Pytorch 1.10.1, optionally with Cuda 11.2`` 
3. Linux Operating System. It has been tested on Ubuntu and MacOS. 
4. Additional modules listed in ``requirements.txt``

Installation 
------------

In order to install the code locally please follow the steps below:

1. Clone this repository and go to the cloned directory.

2. Set the environment variable to point to your python executable:

   `export PYTHON=<path to python 3.9 executable>`

3. Run the following command to set up the environment:

   `make` on **Linux/Mac**

4. Activate the environment by running:

   `source mnn/bin/activate` on **Linux/Mac**


Running Jupyter Notebook 
------------------------

1. Run the following script to start jupyter: 

   `./bin/run_notebooks.sh`

2. In the jupyter lab go to the notebooks folder which contains all the relevant notebooks 

3. Start with the toy_classification.ipynb.

4. Select the mnn kernel.

5. You should now be able to run the notebooks.

Contact
-------

email: yamuna dot krishnamurthy at rhul.ac.uk

Publication
-------

Yamuna Krishnamurthy and Chris Watkins, [Interpretability in Gated Modular Neural Networks](https://xai4debugging.github.io/files/papers/interpretability_in_gated_modu.pdf), in Explainable AI approaches for debugging and diagnosis Workshop at Neural Information Processing (NeurIPS), Dec 2021.


