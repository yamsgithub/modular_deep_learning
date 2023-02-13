#!/bin/bash

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_no_gate_model.sh -m cifar10_no_gate_self_information -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_no_gate_model.sh -m cifar10_no_gate_self_information -k 1 -g gate_layers_top_k -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_no_gate_model.sh -m cifar10_no_gate_self_information -k 2 -g gate_layers_top_k -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_no_gate_model.sh -m cifar10_no_gate_self_information -r 10 -M 5 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_no_gate_model.sh -m cifar10_no_gate_self_information -k 1 -g gate_layers_top_k -r 10 -M 5 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_no_gate_model.sh -m cifar10_no_gate_self_information -k 2 -g gate_layers_top_k -r 10 -M 5 -E 200
