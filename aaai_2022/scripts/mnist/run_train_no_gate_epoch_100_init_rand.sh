#!/bin/bash

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_no_gate_model.sh -r 10 -M 10 -E 20

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_no_gate_model.sh -r 10 -M 5 -E 20

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_no_gate_model.sh -m 'mnist_no_gate_entropy_argmax' -mt moe_no_gate_entropy_model -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_train_no_gate_model.sh -m 'mnist_no_gate_entropy_argmax' -mt moe_no_gate_entropy_model -r 10 -M 5 -E 100



