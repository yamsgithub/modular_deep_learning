#!/bin/bash

# Generic options:
#SBATCH -J CIFAR100_orig
#SBATCH --account=p72053  # Run job under project <project>
#SBATCH --time=48:00:00  # Run for a max of 2 day

# Node resources:
# (choose between 1-4 gpus per node)

#SBATCH --partition=zen2_0256_a40x2   # Choose either "gpu" or "infer" node type        
#SBATCH --qos zen2_0256_a40x2 
#SBATCH --gres=gpu:1      # 4 GPUS per node (plus 100% of node CPU and RAM per GPU)

# Run commands:

# Place other commands here

cd /home/fs72053/yamuna_k/modular_deep_learning

export MNN_HOME=/home/fs72053/yamuna_k/modular_deep_learning:/home/fs72053/yamuna_k/modular_deep_learning/aaai_2022/src
echo "MNN_HOME  : $MNN_HOME"

export PYTHONPATH=$MNN_HOME
echo "PYTHONPATH: $PYTHONPATH"

conda init bash
conda deactivate
conda activate mnn

python aaai_2022/src/cifar100/cifar100_train_original_model.py $*

echo "end of job"
