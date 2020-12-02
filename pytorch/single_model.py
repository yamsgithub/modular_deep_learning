import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil, floor, modf

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

class single_model(nn.Module):
    def __init__(self, parameters, num_experts, num_classes):
        super(single_model, self).__init__()
        output = float(parameters)/(2*(num_experts+1)*2)
        if modf(output)[0] < 0.5:
            output = floor(output)
        else:
            output = ceil(output)
        if output <= 0.0:
            output = 1
        print('parameters', parameters, 'num_experts', num_experts+1, 'output', output)
        self.model = nn.Sequential(
            nn.Linear(2, output*4),
            nn.ReLU(),
            nn.Linear(output*4, num_classes),
            nn.Softmax(dim=1)
        )
        self.model = self.model.to(device)
        
        
    def forward(self, input):
        return self.model(input).to(device)

    def train(self, trainloader, testloader, optimizer, loss_criterion, accuracy, epochs):    

        history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}
        for epoch in range(0, epochs):
            running_loss = 0.0
            training_accuracy = 0.0
            test_accuracy = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = self(inputs)
                loss = loss_criterion(outputs.to(device), labels).to(device)
                loss.backward()
                optimizer.step()
            
                running_loss += loss.item()
                training_accuracy += accuracy(outputs, labels)
                
            for j, test_data in enumerate(testloader, 0):
                test_input, test_labels = test_data
                test_input, test_labels = test_input.to(device), test_labels.to(device)
                test_outputs = self(test_input)
                test_accuracy += accuracy(test_outputs, test_labels)
            history['loss'].append(running_loss/(i+1))
            history['accuracy'].append(training_accuracy/(i+1))
            history['val_accuracy'].append(test_accuracy/(j+1))
            print('epoch: %d loss: %.3f training accuracy: %.3f val accuracy: %.3f' %
                  (epoch + 1, running_loss / (i+1), training_accuracy/(i+1), test_accuracy/(j+1)))
        return history
