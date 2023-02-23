from cifar10_original_moe_training import *
from single_model_training import *

model = 'cifar10_single_model'

num_classes = 10

num_epochs = 1

runs = 10

# Paths to where the trained models, figures and results will be stored. You can change this as you see fit.
working_path = '/gpfs/data/fs71921/yamunak'
model_path = os.path.join(working_path, 'models/cifar10')

if not os.path.exists(model_path):
    os.mkdir(model_path)

train_single_model(model, single_model, cifar10_trainloader, cifar10_valloader, num_classes, num_epochs, runs, model_path)
