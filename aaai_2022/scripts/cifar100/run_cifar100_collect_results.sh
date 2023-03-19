#!/bin/bash

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_collect_results.sh -m cifar100_stochastic_rand_init -mt moe_stochastic_model -r 1 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_collect_results.sh -g gate_layers_top_k -k 1 -m cifar100_rand_init_top_1 -mt moe_top_k_model -r 1 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_collect_results.sh -g gate_layers_top_k -k 2 -m cifar100_rand_init_top_2 -mt moe_top_k_model -r 1 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_collect_results.sh -m cifar100_rand_init  -mt moe_expectation_model -r 1 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_collect_results.sh -m  cifar100_loss_gate_rand_init -r 10  -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_collect_results.sh -m cifar100_with_attn_stochastic_rand_init -mt moe_stochastic_model -r 1 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_collect_results.sh  -k 1 -m cifar100_with_attn_rand_init_top_1 -mt moe_top_k_model -r 1 -M 20 -E 200  

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_collect_results.sh -k 2 -m cifar100_with_attn_rand_init_top_2 -mt moe_top_k_model -r 1 -M 20 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_collect_results.sh -m cifar100_with_attn_rand_init -mt moe_expectation_model -r 1 -M 20 -E 200 

# sbatch aaai_2022/scripts/cifar100/schedule_cifar100_single_model.sh
