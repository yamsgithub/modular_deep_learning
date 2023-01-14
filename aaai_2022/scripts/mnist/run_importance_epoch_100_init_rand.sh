#!/bin/bash

for i in 0.2 0.4 0.6 0.8 1.0; 
    do sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -m mnist_importance_rand_init -i $i -r 10 -M 10 -E 100; 
       sbatch aaai_2022/scripts/mnist/schedule_mnist_with_attention.sh -m mnist_with_attn_importance_rand_init -i $i -r 10 -M 10 -E 100; 
       sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -g gate_layers_top_k -k 2 -m mnist_importance_rand_init_top_2 -mt moe_top_k_model -i $i -r 10 -M 10 -E 100;
       sbatch aaai_2022/scripts/mnist/schedule_mnist_with_attention.sh  -k 2 -m mnist_with_attn_importance_rand_init_top_2 -mt moe_top_k_model -i $i -r 10 -M 10 -E 100;
done