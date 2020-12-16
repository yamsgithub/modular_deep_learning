import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil, floor, modf, sqrt

from collections import OrderedDict

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

class single_model_shallow(nn.Module):
    def __init__(self, parameters, num_experts, num_classes, bias=None):
        super(single_model_shallow, self).__init__()
        output = float(parameters)/(2*(num_experts+1)*2)
        if modf(output)[0] < 0.5:
            output = floor(output)
        else:
            output = ceil(output)
        if output <= 0.0:
            output = 1
        print('parameters', parameters, 'num_experts', num_experts+1, 'output', output)
        self.model = nn.Sequential(OrderedDict({
            'linear_1':nn.Linear(2, (num_experts+1)*output),
            'relu':nn.ReLU(),
            'linear_2':nn.Linear((num_experts+1)*output, num_classes),
            'softmax':nn.Softmax(dim=1)
            })
        )
        if not bias is None:
            layers = dict(self.model.named_children())
            with torch.no_grad():
                layers['linear_1'].bias.fill_(bias)
                layers['linear_2'].bias.fill_(bias)
            
        self.model = self.model.to(device)
        
        
    def forward(self, input):
        return self.model(input)

    def train(self, trainloader, testloader, optimizer, loss_criterion, accuracy, epochs):    

        history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}
        for epoch in range(0, epochs):
            running_loss = 0.0
            training_accuracy = 0.0
            test_accuracy = 0.0
            i = 0
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = self(inputs)
                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
                training_accuracy += accuracy(outputs, labels)

                i += 1

            j = 0
            for test_input, test_labels in testloader:
                test_input, test_labels = test_input.to(device, non_blocking=True ), test_labels.to(device, non_blocking=True)
                test_outputs = self(test_input)
                test_accuracy += accuracy(test_outputs, test_labels)
                j+=1
            history['loss'].append(running_loss/(i+1))
            history['accuracy'].append(training_accuracy/(i+1))
            history['val_accuracy'].append(test_accuracy/(j+1))
            print('epoch: %d loss: %.3f training accuracy: %.3f val accuracy: %.3f' %
                  (epoch + 1, running_loss / (i+1), training_accuracy/(i+1), test_accuracy/(j+1)))
        return history

class single_model_deep(nn.Module):
    def __init__(self, parameters, num_experts, num_classes, bias=None):
        super(single_model_deep, self).__init__()
        output = float(parameters)/(4*(num_experts+1)*8)
        output = (sqrt(6*parameters + 9)/3 - 1)/(num_experts + 1)
        output = ceil(output)
            
        if output <= 0.0:
            output = 1
        print('parameters', parameters, 'num_experts', num_experts+1, 'output', output)
        self.model = nn.Sequential(OrderedDict({
            'linear_1':nn.Linear(2, (num_experts+1)*output),
            'relu_1':nn.ReLU(),
            'linear_2':nn.Linear((num_experts+1)*output, (num_experts+1)*output),
            'relu_2':nn.ReLU(),
            'linear_3':nn.Linear((num_experts+1)*output, (num_experts+1)*int(output/2)),
            'relu_3':nn.ReLU(),
            'linear_4':nn.Linear((num_experts+1)*int(output/2),num_classes),
            'softmax':nn.Softmax(dim=1)
        })
        )
        if not bias is None:
            layers = dict(self.model.named_children())
            with torch.no_grad():
                layers['linear_1'].bias.fill_(bias)
                layers['linear_2'].bias.fill_(bias)
                layers['linear_3'].bias.fill_(bias)
                layers['linear_4'].bias.fill_(bias)

        self.model = self.model.to(device)
        
        
    def forward(self, input):
        return self.model(input)

    def train(self, trainloader, testloader, optimizer, loss_criterion, accuracy, epochs):    

        history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}
        for epoch in range(0, epochs):
            running_loss = 0.0
            training_accuracy = 0.0
            test_accuracy = 0.0
            i = 0
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = self(inputs)
                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
                training_accuracy += accuracy(outputs, labels)

                i+=1

            j = 0
            for test_input, test_labels in testloader:
                test_input, test_labels = test_input.to(device), test_labels.to(device)
                test_outputs = self(test_input)
                test_accuracy += accuracy(test_outputs, test_labels)
                j==1
            history['loss'].append(running_loss/(i+1))
            history['accuracy'].append(training_accuracy/(i+1))
            history['val_accuracy'].append(test_accuracy/(j+1))
            print('epoch: %d loss: %.3f training accuracy: %.3f val accuracy: %.3f' %
                  (epoch + 1, running_loss / (i+1), training_accuracy/(i+1), test_accuracy/(j+1)))
        return history
