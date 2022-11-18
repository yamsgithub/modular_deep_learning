#!/bin/bash

# Generic options:

#SBATCH --account=bdrap03  # Run job under project <project>
#SBATCH --time=05:00:00  # Run for a max of 2 day

# Node resources:
# (choose between 1-4 gpus per node)

#SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type        
#SBATCH --nodes=7 --ntasks-per-node=1 # Resources from a single node
#SBATCH --gres=gpu:4      # 4 GPUS per node (plus 100% of node CPU and RAM per GPU)

# Run commands:

# Place other commands here

cd /nobackup/projects/bdrap03/yamuna/modular_deep_learning/

export MNN_HOME=/nobackup/projects/bdrap03/yamuna/modular_deep_learning/
echo "MNN_HOME  : $MNN_HOME"

export PYTHONPATH=$MNN_HOME
echo "PYTHONPATH: $PYTHONPATH"

conda init bash
conda deactivate
conda activate mnn

python aaai_2022/src/cifar10/cifar10_with_attn_sample_sim_model_1_4.py -i $SLURM_ARRAY_TASK_ID

echo "end of job"
