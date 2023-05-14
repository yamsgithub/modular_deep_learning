#!/bin/bash

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_no_gate_model.sh -r 10 -M 20 -E 20

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_no_gate_model.sh -m cifar100_no_gate_entropy_argmax -mt moe_no_gate_entropy_model -r 10 -M 20 -E 200
