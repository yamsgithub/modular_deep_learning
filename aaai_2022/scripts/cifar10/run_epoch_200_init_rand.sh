#!/bin/bash

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_original_model.sh -m cifar10_stochastic_rand_init -mt moe_stochastic_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_original_model.sh -g gate_layers_top_k -k 1 -m cifar10_rand_init_top_1 -mt moe_top_k_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_original_model.sh -g gate_layers_top_k -k 2 -m cifar10_rand_init_top_2 -mt moe_top_k_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_original_model.sh -m cifar10_rand_init  -mt moe_expectation_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_loss_gate_model.sh -m  cifar10_loss_gate_rand_init -r 10  -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_with_attention.sh -m cifar10_with_attn_stochastic_rand_init -mt moe_stochastic_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_with_attention.sh  -k 1 -m cifar10_with_attention_rand_init_top_1 -mt moe_top_k_model -r 10 -M 10 -E 200  

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_with_attention.sh -k 2 -m cifar10_with_attention_rand_init_top_2 -mt moe_top_k_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_with_attention.sh -m cifar10_with_attention_rand_init -mt moe_expectation_model -r 10 -M 10 -E 200 

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_single_model.sh