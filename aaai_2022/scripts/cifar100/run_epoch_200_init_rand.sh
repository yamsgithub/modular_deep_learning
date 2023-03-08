#!/bin/bash

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -m cifar100_stochastic_rand_init -mt moe_stochastic_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -g gate_layers_top_k -k 1 -m cifar100_rand_init_top_1 -mt moe_top_k_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -g gate_layers_top_k -k 2 -m cifar100_rand_init_top_2 -mt moe_top_k_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -m cifar100_rand_init  -mt moe_expectation_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_without_reg_loss_gate_model.sh -m  cifar100_loss_gate_rand_init -r 10  -M 10 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_with_attention.sh -m cifar100_with_attn_stochastic_rand_init -mt moe_stochastic_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_with_attention.sh  -k 1 -m cifar100_with_attention_rand_init_top_1 -mt moe_top_k_model -r 10 -M 10 -E 200  

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_with_attention.sh -k 2 -m cifar100_with_attention_rand_init_top_2 -mt moe_top_k_model -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_with_attention.sh -m cifar100_with_attention_rand_init -mt moe_expectation_model -r 10 -M 10 -E 200 

sbatch aaai_2022/scripts/cifar100/schedule_cifar100_single_model.sh