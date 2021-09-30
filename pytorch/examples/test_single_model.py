import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt, ceil
from collections import OrderedDict

import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim


if cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

def plot_data(X, y, num_classes, save_as):
    f, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,8))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple'][0:num_classes]
    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y,palette=colors, ax=ax)
    ax.set_title("2D 3 classes Generated Data")
    plt.ylabel('Dim 2')
    plt.xlabel('Dim 1')
    plt.savefig(save_as)
    #plt.show()
    plt.clf()
    plt.close()
    

def generate_data(dataset, size):
    num_classes = 2
    X = y = None        
    X = 2 * np.random.random((size,2)) - 1
    def classifier2(X): # a 4x4 checkerboard pattern -- you can use the same method to make up your own checkerboard patterns
        return (np.sum( np.ceil( 2 * X).astype(int), axis=1 ) % 2).astype(float)
    
    y = classifier2( X )

    plot_data(X, y, num_classes, 'figures/all/'+dataset+'_'+str(num_classes)+'_.png')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(len(y_train))
    print(sum(y_train))
    print(len(y_test))
    print(sum(y_test))

    # Create trainloader
    batchsize = 32
    trainset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), 
                                              torch.tensor(y_train, dtype=torch.long))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=0, pin_memory=True)
    testset = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                             torch.tensor(y_test, dtype=torch.long))
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             shuffle=True, num_workers=0, pin_memory=True)


    return X, y, trainset, trainloader, testset, testloader, num_classes


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
                j+=1
            history['loss'].append(running_loss/i)
            history['accuracy'].append(training_accuracy/i)
            history['val_accuracy'].append(test_accuracy/j)
            print('epoch: %d loss: %.3f training accuracy: %.3f val accuracy: %.3f' %
                  (epoch + 1, running_loss / i, training_accuracy/i, test_accuracy/j))
        return history

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

X, y, trainset, trainloader, testset, testloader, num_classes = generate_data('checker_board-2', 3000)

num_experts = 2
num_classes = 2

model = single_model_deep(2062, num_experts, num_classes)

optimizer = optim.RMSprop(model.parameters(),
                          lr=0.001, momentum=0.9)

epochs = 40
hist = model.train(trainloader, testloader, optimizer, nn.CrossEntropyLoss(), accuracy, epochs=epochs)
