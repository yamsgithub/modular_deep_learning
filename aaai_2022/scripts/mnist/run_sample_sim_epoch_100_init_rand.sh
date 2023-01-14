#!/bin/bash

for i in 1e-6 1e-5; 
    do for j in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 ; 
        do sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -m mnist_rand_init -ss $i -sd $j  -r 10 -M 10 -E 100; 
        sbatch aaai_2022/scripts/mnist/schedule_mnist_with_attention.sh -m mnist_with_attn_rand_init -ss $i -sd $j  -r 10 -M 10 -E 100; 
        # do sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -g gate_layers_top_k -k 2 -m mnist_sample_sim_rand_init_top_2 -mt moe_top_k_model -ss $i -sd $j -r 10 -M 10 -E 100;
        # sbatch aaai_2022/scripts/mnist/schedule_mnist_with_attention.sh  -k 2 -m mnist_with_attn_sample_sim_rand_init_top_2 -mt moe_top_k_model -ss $i -sd $j -r 10 -M 10 -E 100;
    done; 
done

