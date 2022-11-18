from cifar10_original_moe_training import *

# expert resnet pre-trained
# gate resnet not pre-trained
model = 'cifar10_without_reg'

total_experts = 5

num_classes = 10

num_epochs = 40

runs = 10

train_original_model(model, cifar10_trainloader, cifar10_testloader, runs, T=[1.0]*num_epochs, num_classes=num_classes, total_experts=total_experts, num_epochs=num_epochs)
