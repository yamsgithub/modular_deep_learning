from cifar10_original_moe_training import *

model = 'cifar10_single_model'

num_classes = 10

num_epochs = 200

runs = 1

train_single_model(model, cifar10_trainloader, cifar10_testloader, num_classes, num_epochs, runs)
