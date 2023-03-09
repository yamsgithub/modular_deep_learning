from cifar100_original_moe_training import *
from single_model_training import *

model = 'cifar100_single_model'

num_classes = 100

num_epochs = 200

runs = 10

# Paths to where the trained models, figures and results will be stored. You can change this as you see fit.
working_path = '/gpfs/data/fs72053/yamuna_k'
model_path = os.path.join(working_path, 'models/cifar100')

if not os.path.exists(model_path):
    os.mkdir(model_path)

train_single_model(model, single_model, cifar100_trainloader, cifar100_valloader, num_classes, num_epochs, runs, model_path)
