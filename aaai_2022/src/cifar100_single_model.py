from cifar100_original_moe_training import *

model = 'cifar100_single_model'

num_epochs = 40

runs = 5

train_single_model(model, cifar100_trainloader, cifar100_testloader, num_classes, num_epochs, runs)
