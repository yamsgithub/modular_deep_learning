#!/bin/bash

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_from_no_gate_model.sh -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_from_no_gate_model.sh -k 1 -g gate_layers_top_k -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_from_no_gate_model.sh  -k 2 -g gate_layers_top_k -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_from_no_gate_model.sh -r 10 -M 5 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_from_no_gate_model.sh -k 1 -g gate_layers_top_k -r 10 -M 5 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_from_no_gate_model.sh -k 2 -g gate_layers_top_k -r 10 -M 5 -E 100
