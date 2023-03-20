#!/bin/bash 
# 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1
# 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2
for i in 1e-7 1e-6 1e-5 1e-4 1e-3; 
    do for j in 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1; 
        do         
            sbatch aaai_2022/scripts/cifar10/schedule_cifar10_collect_results.sh -m cifar10_with_attn_rand_init_wideres -ss $i -sd $j -f cifar10_top_k_results.csv -r 10 -M 10 -E 200; 
                
        sbatch aaai_2022/scripts/cifar10/schedule_cifar10_collect_results.sh  -m cifar10_with_attn_rand_init_top_2_wideres -k 2 -mt moe_top_k_model -ss $i -sd $j -f cifar10_top_k_results.csv -r 10 -M 10 -E 200;
    done; 
done
