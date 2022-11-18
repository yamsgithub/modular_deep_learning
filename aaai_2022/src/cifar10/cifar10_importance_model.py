import argparse
from cifar10_original_moe_training import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='importance weight')
args = vars(parser.parse_args())
print('args', args)

w_importance = 0.0

if not args['i'] is None:
    w_importance = 0.2*int(args['i'])

# expert resnet pre-trained
# gate resnet not pre-trained
model = 'cifar10_with_reg'

total_experts = 5

num_classes = 10

num_epochs = 40

runs = 10

train_original_model(model, cifar10_trainloader, cifar10_testloader, runs, T=[1.0]*num_epochs, w_importance=w_importance, num_classes=num_classes, total_experts=total_experts, num_epochs=num_epochs)
