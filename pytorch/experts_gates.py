import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ### Networks and callbacks

#Expert network
class expert_layers_shallow(nn.Module):
    def __init__(self, output):
        super(expert_layers_shallow, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(2, 4),
                    nn.ReLU(),
                    nn.Linear(4,output),
                    nn.Softmax(dim=1)
                )        
        
    def forward(self, input):
        return self.model(input)

class expert_layers_shallow_presoftmax(nn.Module):
    def __init__(self, output):
        super(expert_layers_shallow_presoftmax, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(2, 4),
                    nn.ReLU(),
                    nn.Linear(4,output)
                )        
        
    def forward(self, input):
        return self.model(input)
    
class expert_layers_deep(nn.Module):
    def __init__(self, output):
        super(expert_layers_deep, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4,output),
            nn.Softmax(dim=1)
        )        
        
    def forward(self, input):
        return self.model(input)

class expert_layers_deep_presoftmax(nn.Module):
    def __init__(self, output):
        super(expert_layers_deep_presoftmax, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4,output)
        )        
        
    def forward(self, input):
        return self.model(input)
    
# Implements escort transform to replace softmax
class escort(nn.Module):
    def __init__(self):
        super(escort, self).__init__()
        
    def forward(self, input):
        batch_size = input.shape[0]
        numerator = torch.square(input)
        denominator = torch.reshape(torch.sum(numerator, dim=1),(batch_size,1))
        output = torch.div(numerator, denominator)
        return output
    
class expert_layers_deep_escort(nn.Module):
    def __init__(self, output):
        super(expert_layers_deep_escort, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4,output),
            escort()
        )        
        
    def forward(self, input):
        return self.model(input)
    
# create a set of experts

def experts(expert_layers_type, num_experts, num_classes):
    models = []
    for i in range(num_experts):
        models.append(expert_layers_type(num_classes))
    return nn.ModuleList(models)


#Gate network (Similar to the expert layer)
class gate_layers_shallow(nn.Module):
    def __init__(self, num_experts):
        super(gate_layers_shallow, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(2, 4),
                    nn.ReLU(),
                    nn.Linear(4,num_experts),
                    nn.Softmax(dim=1)
                )
        
    def forward(self, input):
        return self.model(input)

class gate_layers_deep(nn.Module):
    def __init__(self, num_experts):
        super(gate_layers_deep, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,num_experts),
            nn.Softmax(dim=1)
        )
        
    def forward(self, input):
        return self.model(input)


class gate_layers_deep_escort(nn.Module):
    def __init__(self, num_experts):
        super(gate_layers_deep_escort, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,num_experts),
            escort()
        )
        
    def forward(self, input):
        return self.model(input)


#Expert network
class expert_layers(nn.Module):
    def __init__(self, output):
        super(expert_layers, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=6*12*12, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=120)

        self.out = nn.Linear(in_features=120, out_features=output)


    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1
        t = t.reshape(-1, 6*12*12)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = F.softmax(self.out(t), dim=1)
        
        return t
#Expert network
class expert_layers_presoftmax(nn.Module):
    def __init__(self, output):
        super(expert_layers_presoftmax, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=6*12*12, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=120)

        self.out = nn.Linear(in_features=120, out_features=output)


    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1
        t = t.reshape(-1, 6*12*12)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        
        return t

#Expert network
class gate_layers(nn.Module):
    def __init__(self, num_experts):
        super(gate_layers, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=num_experts)
        self.num_experts = num_experts
        self.count = 0

    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # conv 2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # fc1
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        t = F.softmax(t, dim=1)
        return t
