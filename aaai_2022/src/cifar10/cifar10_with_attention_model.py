from cifar10_moe_with_attention_training import *

model = 'cifar10_with_attention'

num_classes = 10

num_epochs = 40

total_experts = 5

runs = 10

train_with_attention(model, cifar10_trainloader, cifar10_testloader, runs, T=[1.0]*num_epochs, num_classes=num_classes, total_experts=total_experts, num_epochs=num_epochs)
