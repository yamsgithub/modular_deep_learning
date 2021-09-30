#!/usr/bin/env python
# coding: utf-8


import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from moe_models import moe_stochastic_model, moe_stochastic_loss


# # Pytorch datasets
# 
# The MNIST dataset is a famous and has been used as a testbed of machine learning algorithms for more than 25 years. 
# 
# Look at the torchvision documentation [here](https://pytorch.org/docs/stable/torchvision/index.html) to find out about other datasets included with Keras.  There is an interesting dataset called Fashion-MNIST which is a plug-in replacement for the MNIST dataset, but which may have very different properties (it is grey-scale images). 


import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transforms: Convert PIL image to tensors and normalize
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

def fmnist_target_transform(target):
    return target+10


# datasets
trainset_mnist = torchvision.datasets.MNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset_mnist = torchvision.datasets.MNIST('./data',
    download=True,
    train=False,
    transform=transform)

trainset_fmnist = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform,
    target_transform=fmnist_target_transform)
testset_fmnist = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform,
    target_transform=fmnist_target_transform)

# trainset_fmnist = torch.utils.data.Subset(trainset_fmnist, range(0,5000))
# testset_fmnist = torch.utils.data.Subset(testset_fmnist, range(0,1000))

# trainset_mnist = torch.utils.data.Subset(trainset_mnist, range(0,5000))
# testset_mnist = torch.utils.data.Subset(testset_mnist, range(0,1000))

trainset = torch.utils.data.ConcatDataset([trainset_mnist, trainset_fmnist])
testset = torch.utils.data.ConcatDataset([testset_mnist, testset_fmnist])


bs = 32 
# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                        shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                        shuffle=False)

# imade size and number of images
image, label = trainset.__getitem__(0)
print(image.shape, len(trainset))

del trainset, testset, trainset_fmnist, testset_fmnist, trainset_mnist, testset_mnist 

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.clf()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
print('Labels:', labels)
# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
imshow(img_grid, one_channel=True)


# Classifying these isn't easy: look how different the 4s are from each other, and the 5s. 
# 
# Take a moment to look at the images carefully: what do you think are the difficulties of classifying these with a neural net?  How large are the features that you would need to use? 

# # Neural network models for MNIST classification
# 
# It is best not to use the entire MNIST dataset as a training set because you will be training for the whole of the practical class. You will learn much more by training repeatedly on smaller subsets, and examining the effect of using different models and parameters. 

class ExpertModel(nn.Module):
    def __init__(self):
        super(ExpertModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2,2) 
        self.fc1 = nn.Linear(32*13*13, 100) # this is a pure linear transform
        self.fc2 = nn.Linear(100, 20) # this is a pure linear transform
        self.bn = nn.BatchNorm1d(100)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32*13*13)
        x = F.relu(self.bn(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=1)

        return x

class GateModel(nn.Module):
    def __init__(self, num_experts):
        super(GateModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2,2) 
        self.fc1 = nn.Linear(32*13*13, 100) # this is a pure linear transform
        self.fc2 = nn.Linear(100, num_experts) # this is a pure linear transform
        self.bn = nn.BatchNorm1d(100)   
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32*13*13)
        x = F.relu(self.bn(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=1)
        
        return x


# create a set of experts
def experts(num_experts):
    models = []
    for i in range(num_experts):
        models.append(ExpertModel())
    return nn.ModuleList(models)

# compute
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# experiment with models with different number of experts
models = {}
total_experts = 10 
history = []
for num_experts in range(1, total_experts+1):
    print('Number of experts ', num_experts)
    expert_models = experts(num_experts)
    gate_model = GateModel(num_experts)
    moe_model = moe_stochastic_model(num_experts, expert_models, gate_model).to(device)
    optimizer = optim.RMSprop(moe_model.parameters(),
                              lr=0.001, momentum=0.9)
    hist = moe_model.train(trainloader, testloader, optimizer, moe_stochastic_loss, accuracy, epochs=10)
    history.append(hist)
    models[num_experts] = moe_model


labels = []
for num_experts, hist in enumerate(history,0):
    plt.plot(range(len(hist['loss'])), hist['loss'])
    labels.append('loss - ' +str(num_experts+1)+' expert')
plt.legend(labels)
plt.savefig('figures/moe_stochastic/loss_moe_stochastic.png')
plt.show()
plt.clf()

labels = []
for num_experts, hist in enumerate(history,0):
    plt.plot(range(len(hist['accuracy'])), hist['accuracy'])
    labels.append('accuracy - ' +str(num_experts+1)+' expert')
plt.legend(labels)
plt.savefig('figures/moe_stochastic/train_accuracy_moe_stochastic.png')
plt.show()
plt.clf()

labels = []
for num_experts, hist in enumerate(history,0):
    plt.plot(range(len(hist['val_accuracy'])), hist['val_accuracy'])
    labels.append('accuracy - ' +str(num_experts+1)+' expert')
plt.legend(labels)
plt.savefig('figures/moe_stochastic/val_accuracy_moe_stochastic.png')
plt.clf()
plt.show()

