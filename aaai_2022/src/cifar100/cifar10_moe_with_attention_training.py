# Experiments with CIFAR10 Dataset and Attentive Gate MoE Training

# The experiments in this notebook include training the attentive gate MoE models as follows:
# 
# 1. attentive gate MoE without regularization.
# 2. attentive gate MoE with $L_{importance}$ regularization.
# 3. attentive gate MoE with $L_s$ regularization.

from cifar10_original_moe_training import *

# Convolutional network with one convolutional layer and 2 hidden layers with ReLU activation
class gate_attn_layers(nn.Module):
    def __init__(self, num_experts):
        super(gate_attn_layers, self).__init__()
        # define layers
        filter_size = 3
        self.filters = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.filters*2)
        self.mp = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(in_channels= self.filters*2, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.filters*4, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(self.filters*4)

        self.fc1 = nn.Linear(self.filters*4*2*2, 512)
        self.fc2 = nn.Linear(512, 32)
                
    def forward(self, x, T=1.0, y=None):
        # conv 1
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.bn2(self.conv2(x))))

        x = self.mp(F.relu(self.conv3(x)))
        x = self.mp(F.relu(self.bn4(self.conv4(x))))

        x = x.reshape(-1, self.filters*4*2*2)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    

# Convolutional network with one convultional layer and 2 hidden layers with ReLU activation
class gate_attn_layers_conv_2(nn.Module):
    def __init__(self, num_classes, channels=3):
        super(gate_attn_layers_conv_2, self).__init__()
        filter_size = 3
        self.filters = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(self.filters*2*16*16,64)
                                
    def forward(self, x, T=1.0, y=None):
        # conv 1
        
        x = F.relu(self.conv1(x))
        x = self.mp(F.relu(self.conv2(x)))
            
        # print(x.shape)
        
        x = x.reshape(-1, self.filters*2*16*16)
        
        x = self.fc1(x)
        
        return x


