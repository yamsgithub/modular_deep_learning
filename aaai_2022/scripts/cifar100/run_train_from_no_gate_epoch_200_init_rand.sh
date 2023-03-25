#!/bin/bash

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_from_no_gate_model.sh -m cifar100_no_gate_self_information -r 10 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_from_no_gate_model.sh -m cifar100_no_gate_self_information -k 1 -g gate_layers_top_k -r 10 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_from_no_gate_model.sh -m cifar100_no_gate_self_information -k 2 -g gate_layers_top_k -r 10 -M 20 -E 200



