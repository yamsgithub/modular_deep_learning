#!/bin/bash 

for i in 1e-4; 
    do for j in 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1; 
        do 
         sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -m cifar100_rand_init_wideres -mt moe_stochastic_model -ss $i -sd $j -D wideres_distance_funct  -r 10 -M 20 -E 200; 
        
         # sbatch aaai_2022/scripts/cifar100/schedule_cifar100_with_attention.sh -m cifar100_with_attn_rand_init_wideres -mt moe_stochastic_model -ss $i -sd $j -D wideres_distance_funct  -r 10 -M 20 -E 200; 
        
         sbatch aaai_2022/scripts/cifar100/schedule_cifar100_original_model.sh -m cifar100_rand_init_top_2_wideres -g gate_layers_top_k -k 2 -mt moe_top_k_model -ss $i -sd $j  -D wideres_distance_funct -r 10 -M 20 -E 200;
        
        # sbatch aaai_2022/scripts/cifar100/schedule_cifar100_with_attention.sh -m cifar100_with_attn_rand_init_top_2_wideres -k 2 -mt moe_top_k_model -ss $i -sd $j -D wideres_distance_funct -r 10 -M 20 -E 200;
    done; 
done