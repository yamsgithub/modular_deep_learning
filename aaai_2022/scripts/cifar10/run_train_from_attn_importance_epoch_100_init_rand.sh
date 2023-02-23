#!/bin/bash

for i in 0.2 0.4 0.6 0.8 1.0; 
    do
        sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_attn_model.sh -m cifar10_with_attn_rand_init -i $i -r 10 -M 10 -E 200; 
       sbatch aaai_2022/scripts/cifar10/schedule_cifar10_train_from_attn_model.sh  -m cifar10_with_attn_rand_init_top_2 -mt moe_top_k_model -g gate_layers_top_k -k 2 -i $i -r 10 -M 10 -E 200;
done