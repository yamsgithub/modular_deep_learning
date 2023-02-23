#!/bin/bash

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_attn_model.sh -m cifar10_with_attention_rand_init_top_1 -mt moe_top_k_model -g gate_layers_top_k -k 1  -r 10 -M 10 -E 200  

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_attn_model.sh  -m cifar10_with_attention_rand_init_top_2 -mt moe_top_k_model -g gate_layers_top_k -k 2 -r 10 -M 10 -E 200

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_attn_model.sh -m cifar10_with_attention_rand_init -mt moe_expectation_model -r 10 -M 10 -E 200 

sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_attn_model.sh -m cifar10_with_attn_stochastic_rand_init -mt moe_stochastic_model -r 10 -M 10 -E 200 
