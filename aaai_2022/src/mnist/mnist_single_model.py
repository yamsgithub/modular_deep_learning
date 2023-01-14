import argparse

from mnist_original_moe_training import *
from single_model_training import *

m = 'mnist_single_model'
mt = single_model
total_experts = 10
num_epochs=20
runs = 1

parser = argparse.ArgumentParser()
parser.add_argument('-m', help='model name')
parser.add_argument('-r', help='number of runs')
parser.add_argument('-E', help='number of epochs')
args = vars(parser.parse_args())
print('args', args)

if not args['m'] is None:
    m = args['m']
if not args['r'] is None:
    runs = int(args['r'])
if not args['E'] is None:
    num_epochs = int(args['E'])

print('model name:', m)
print('runs:', runs)
print('Num epochs:', num_epochs)

num_classes = 10

# Paths to where the trained models, figures and results will be stored. You can change this as you see fit.
working_path = '/nobackup/projects/bdrap03/yamuna/modular_deep_learning/aaai_2022/src'
model_path = os.path.join(working_path, '../models/mnist')

train_single_model(m, mt, trainloader, testloader, runs=runs, num_classes=num_classes, 
                     num_epochs=num_epochs, model_path=model_path)