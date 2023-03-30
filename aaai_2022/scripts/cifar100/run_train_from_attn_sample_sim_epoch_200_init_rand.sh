#!/bin/bash

for i in 1e-5; 
    do for j in 1e-7 1e-6 1e-5 1e-4; 
        do 
        sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_from_attn_model.sh -m cifar100_with_attn_rand_init_wideres -mt moe_stochastic -ss $i -sd $j  -r 10 -M 20 -E 200; 
        
        sbatch aaai_2022/scripts/cifar100/schedule_cifar100_train_from_attn_model.sh -m cifar100_with_attn_rand_init_top_2_wideres -mt moe_top_k_model -g gate_layers_top_k -k 2 -ss $i -sd $j -r 10 -M 20 -E 200;
    done;
done

