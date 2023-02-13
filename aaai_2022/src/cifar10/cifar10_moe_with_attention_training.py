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


# Functions to train models

# Function to train attentive gate model with and without regularization
# 
# * w_importance_range is the range of values for the $w_{importance}$ hyperparameter of the $L_{importance}$ regularization.
# * w_sample_sim_same_range is the range of values for $\beta_s$ hyperparameter of the $L_s$ regularization.
# * w_sample_sim_diff_range is the range of values for $\beta_d$ hyperparameter of the $L_s$ regularization.
# def train_with_attention(model_1, trainloader, testloader, runs, T=[1.0]*20, 
#                          w_importance=0.0, w_sample_sim_same=0.0, 
#                          w_sample_sim_diff=0.0, 
#                          num_classes=10, total_experts=5, num_epochs=20):
    
#     hidden = 32                                                  
        
#     print('w_importance','{:.1f}'.format(w_importance))
#     if w_sample_sim_same < 1:
#         print('w_sample_sim_same',str(w_sample_sim_same))
#     else:
#         print('w_sample_sim_same','{:.1f}'.format(w_sample_sim_same))

#     if w_sample_sim_diff < 1:
#         print('w_sample_sim_diff',str(w_sample_sim_diff))
#     else:
#         print('w_sample_sim_diff','{:.1f}'.format(w_sample_sim_diff))

#     for run in range(1, runs+1):

#         print('Run:', run)

#         n_run_models_1 = []

#         models = {'moe_expectation_model':{'model':moe_expectation_model,'loss':cross_entropy_loss().to(device),
#                                            'experts':{}},}
#         for key, val in models.items():

#             expert_models = experts(total_experts, num_classes).to(device)

#             gate_model = gate_attn_layers(total_experts).to(device)

#             moe_model = val['model'](total_experts, num_classes, attention_flag=1, hidden=hidden, 
#                                      experts=expert_models, gate=gate_model, device=device).to(device)

#             optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001, amsgrad=False)
#             optimizer = default_optimizer(optimizer_moe=optimizer_moe)

#             hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer = optimizer,
#                                    T = T, w_importance=w_importance, 
#                                    w_sample_sim_same = w_sample_sim_same, w_sample_sim_diff = w_sample_sim_diff, 
#                                    accuracy=accuracy, epochs=num_epochs)
#             val['experts'][total_experts] = {'model':moe_model, 'history':hist}

#         # Save all the trained models
#         plot_file = generate_plot_file(model_1, T[0], w_importance=w_importance, w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
#                                        specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

#         if os.path.exists(os.path.join(model_path, plot_file)):
#             n_run_models_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'))
#         n_run_models_1.append(models)
#         torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))
#         n_run_models_1 = []
            


# # Function to distill the attentive gate model to the original model
# def train_from_model(m, num_epochs, num_classes, total_experts, w_importance=0.0, 
#                      w_sample_sim_same=0.0, w_sample_sim_diff=0.0,
#                      trainloader=None, testloader=None, expert_no_grad=True, gate_no_grad=False):
    
#     T = [1.0]*num_epochs
#     print('w_importance','{:.1f}'.format(w_importance))

#     if w_sample_sim_same < 1:
#         print('w_sample_sim_same',str(w_sample_sim_same))
#     else:
#         print('w_sample_sim_same','{:.1f}'.format(w_sample_sim_same))

#     if w_sample_sim_diff < 1:
#         print('w_sample_sim_diff',str(w_sample_sim_diff))
#     else:
#         print('w_sample_sim_diff','{:.1f}'.format(w_sample_sim_diff))

#     plot_file = generate_plot_file(m, temp=T[0], w_importance=w_importance,  
#                                    w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
#                                    specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

#     attn_models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

#     n_run_models_1 = []
#     for model in attn_models: 
#         # Initialise the new expert weights to the weights of the experts of the trained attentive gate model.
#         # Fix all the weights of the new experts so they are not trained. 

#         new_expert_models = experts(total_experts, num_classes).to(device)
#         old_expert_models = model['moe_expectation_model']['experts'][total_experts]['model'].experts
#         for i, expert in enumerate(new_expert_models):
#             old_expert = old_expert_models[i]
#             expert.load_state_dict(old_expert.state_dict())
#             if expert_no_grad:
#                 for param in expert.parameters():
#                     param.requires_grad = False

#         new_gate_model = gate_layers(total_experts).to(device)
#         old_gate_model = model['moe_expectation_model']['experts'][total_experts]['model'].gate
#         new_gate_model.load_state_dict(old_gate_model.state_dict(), strict=False)

#         if gate_no_grad:
#             for param in new_gate_model.parameters():
#                 param.requires_grad = False
#             new_gate_model.out = nn.Linear(in_features=32, out_features=num_experts)

#         gate_model = new_gate_model

#         models = {'moe_expectation_model':{'model':moe_expectation_model,'loss':cross_entropy_loss().to(device),
#                                        'experts':{}},}

#         for key, val in models.items():

#             # gate_model = gate_layers(total_experts).to(device)                

#             moe_model = val['model'](total_experts, num_classes,
#                                      experts=new_expert_models, gate= gate_model, device=device).to(device)

#             optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001, amsgrad=False)


#             hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer_moe = optimizer_moe,
#                                    T = T, accuracy=accuracy, epochs=num_epochs)
#             val['experts'][total_experts] = {'model':moe_model, 'history':hist}

#         plot_file = generate_plot_file('new_'+m, T[0], w_importance=w_importance, w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
#                                        specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

#         if os.path.exists(os.path.join(model_path, plot_file)):
#             n_run_models_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'))

#         n_run_models_1.append(models)                                
#         torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))
#         n_run_models_1 = []
#         print(plot_file)        
