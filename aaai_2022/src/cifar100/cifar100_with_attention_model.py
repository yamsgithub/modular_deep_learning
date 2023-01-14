from cifar100_moe_with_attention_training import *

model = 'cifar100_with_attention'

num_classes = 100

num_epochs = 40

total_experts = 20

runs = 5

train_with_attention(model, cifar100_trainloader, cifar100_testloader, runs, T=[1.0]*num_epochs, 
                         num_classes=num_classes, total_experts=total_experts, num_epochs=num_epochs)
