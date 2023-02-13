#!/bin/bash
for i in 1e-6 1e-5 1e-4 1e-3; 
    do for j in 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1; 
        do 
        sbatch aaai_2022/scripts/cifar10/schedule_cifar10_original_model.sh -m cifar10_rand_init_wideres -ss $i -sd $j -D wideres_distance_funct -r 10 -M 10 -E 200; 
        
        sbatch aaai_2022/scripts/cifar10/schedule_cifar10_with_attention.sh -m cifar10_with_attn_rand_init_wideres -ss $i -sd $j -D wideres_distance_funct -r 10 -M 10 -E 200; 
        
        sbatch aaai_2022/scripts/cifar10/schedule_cifar10_original_model.sh -m cifar10_rand_init_top_2_wideres -g gate_layers_top_k -k 2 -mt moe_top_k_model -ss $i -sd $j  -D wideres_distance_funct -r 10 -M 10 -E 200;
        
        sbatch aaai_2022/scripts/cifar10/schedule_cifar10_with_attention.sh -m cifar10_with_attn_rand_init_top_2_wideres -k 2 -mt moe_top_k_model -ss $i -sd $j  -D wideres_distance_funct -r 10 -M 10 -E 200;
    done; 
done
