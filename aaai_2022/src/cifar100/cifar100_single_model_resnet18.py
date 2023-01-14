from cifar100_original_moe_training_resnet18 import *

model = 'cifar_single_model_resnet18'

num_epochs = 40

runs = 5

train_single_model(model, cifar100_trainloader, cifar100_testloader, num_classes, num_epochs, runs)
