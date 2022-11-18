from cifar100_original_moe_training import *

w_importance = 0.0
w_sample_sim_same = 1e-3
w_sample_sim_diff = 1e-2

model = 'cifar100_with_reg'

total_experts = 20

num_classes = 100

num_epochs = 40

runs = 5

train_original_model(model, cifar100_trainloader, cifar100_testloader, runs, T=[1.0]*num_epochs, 
                         w_importance=w_importance, w_sample_sim_same=w_sample_sim_same, 
                         w_sample_sim_diff=w_sample_sim_diff,
                         num_classes=num_classes, total_experts=total_experts, num_epochs=num_epochs)
