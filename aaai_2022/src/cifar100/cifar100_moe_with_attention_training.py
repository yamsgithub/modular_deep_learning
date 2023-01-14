from cifar100_original_moe_training import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('device', device)

# import MoE expectation model. All experiments for this dataset are done with the expectation model as it
# provides the best guarantee of interpretable task decompositions
# from moe_models.moe_expectation_model import moe_expectation_model
# from helper.moe_models import cross_entropy_loss
# from helper.visualise_results import *

# from torchvision.models.resnet import resnet18


# ### NOTE: Pre-trained models are provided to check the results of all the experiments if you do not have the time to train all the models. 


# Paths to where the trained models and figures will be stored. You can change this as you see fit.
working_path = '/nobackup/projects/bdrap03/yamuna/modular_deep_learning/aaai_2022/notebooks'
model_path = os.path.join(working_path, '../models')

if not os.path.exists(model_path):
    os.mkdir(model_path)

# Convolutional network with one convolutional layer and 2 hidden layers with ReLU activation
class gate_attn_layers(nn.Module):
    def __init__(self, num_experts):
        super(gate_attn_layers, self).__init__()
        # define layers
       # define layers
        filter_size = 3
        self.filters = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.filters, kernel_size=filter_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters*2, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.filters*2)
        self.mp = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(in_channels= self.filters*2, out_channels=self.filters*4, kernel_size=filter_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.filters*4, out_channels=self.filters*8, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(self.filters*8)

        self.fc1 = nn.Linear(self.filters*8*2*2, 512)
        self.fc2 = nn.Linear(512, 64)
                 
    def forward(self, x, T=1.0, y=None):
        # conv 1
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.bn2(self.conv2(x))))

        x = self.mp(F.relu(self.conv3(x)))
        x = self.mp(F.relu(self.bn8(self.conv4(x))))
        
        x = x.reshape(-1, self.filters*8*2*2)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# ## Functions to train models

# ### Function to train attentive gate model with and without regularization
# 
# * w_importance_range is the range of values for the $w_{importance}$ hyperparameter of the $L_{importance}$ regularization.
# * w_sample_sim_same_range is the range of values for $\beta_s$ hyperparameter of the $L_s$ regularization.
# * w_sample_sim_diff_range is the range of values for $\beta_d$ hyperparameter of the $L_s$ regularization.

def train_with_attention(model_1, trainloader, testloader, runs, T=[1.0]*20, 
                         w_importance=0.0, w_sample_sim_same=0.0, 
                         w_sample_sim_diff=0.0, 
                         num_classes=10, total_experts=5, num_epochs=20):
    
    hidden = 64
    
    print('w_importance','{:.1f}'.format(w_importance))
    if w_sample_sim_same < 1:
        print('w_sample_sim_same',str(w_sample_sim_same))
    else:
        print('w_sample_sim_same','{:.1f}'.format(w_sample_sim_same))

    if w_sample_sim_diff < 1:
        print('w_sample_sim_diff',str(w_sample_sim_diff))
    else:
        print('w_sample_sim_diff','{:.1f}'.format(w_sample_sim_diff))

    for run in range(1, runs+1):

        print('Run:', run)

        n_run_models_1 = []

        models = {'moe_expectation_model':{'model':moe_expectation_model,'loss':cross_entropy_loss().to(device),
                                           'experts':{}},}
        for key, val in models.items():

            expert_models = experts(total_experts, num_classes).to(device)

            gate_model = gate_attn_layers(total_experts).to(device)

            moe_model = val['model'](total_experts, num_classes, attention_flag=1, hidden=hidden, 
                                     experts=expert_models, gate=gate_model, device=device).to(device)

            optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001, amsgrad=False)

            hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer_moe = optimizer_moe,
                                   T = T, w_importance=w_importance, 
                                   accuracy=accuracy, epochs=num_epochs)
            val['experts'][total_experts] = {'model':moe_model, 'history':hist}

        # Save all the trained models
        plot_file = generate_plot_file(model_1, T[0], w_importance=w_importance, w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
                                       specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

        if os.path.exists(os.path.join(model_path, plot_file)):
            n_run_models_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'))
        n_run_models_1.append(models)
        torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))
        n_run_models_1 = []

# ### Function to distill the attentive gate model to the original model
def train_from_model(m, num_epochs, num_classes, total_experts, w_importance=0.0, 
                     w_sample_sim_same=0.0, w_sample_sim_diff=0.0,
                     trainloader=None, testloader=None, expert_no_grad=True, gate_no_grad=False):
    
    T = [1.0]*num_epochs        
    print('w_importance','{:.1f}'.format(w_importance))

    if w_sample_sim_same < 1:
        print('w_sample_sim_same',str(w_sample_sim_same))
    else:
        print('w_sample_sim_same','{:.1f}'.format(w_sample_sim_same))

    if w_sample_sim_diff < 1:
        print('w_sample_sim_diff',str(w_sample_sim_diff))
    else:
        print('w_sample_sim_diff','{:.1f}'.format(w_sample_sim_diff))

    plot_file = generate_plot_file(m, temp=T[0], w_importance=w_importance,  
                                   w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
                                   specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

    attn_models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

    n_run_models_1 = []
    for i, model in enumerate(attn_models): 
        
        # Initialise the new expert weights to the weights of the experts of the trained attentive gate model.
        # Fix all the weights of the new experts so they are not trained. 
        print('Model ', i)
        new_expert_models = experts(total_experts, num_classes).to(device)
        old_expert_models = model['moe_expectation_model']['experts'][total_experts]['model'].experts
        for i, expert in enumerate(new_expert_models):
            old_expert = old_expert_models[i]
            expert.load_state_dict(old_expert.state_dict())
            if expert_no_grad:
                for param in expert.parameters():
                    param.requires_grad = False

        new_gate_model = gate_layers(total_experts).to(device)
        old_gate_model = model['moe_expectation_model']['experts'][total_experts]['model'].gate
        new_gate_model.load_state_dict(old_gate_model.state_dict(), strict=False)

        if gate_no_grad:
            for param in new_gate_model.parameters():
                param.requires_grad = False
            new_gate_model.out = nn.Linear(in_features=32, out_features=num_experts)

        gate_model = new_gate_model

        models = {'moe_expectation_model':{'model':moe_expectation_model,'loss':cross_entropy_loss().to(device),
                                       'experts':{}},}

        for key, val in models.items():

            # gate_model = gate_layers(total_experts).to(device)                

            moe_model = val['model'](total_experts, num_classes,
                                     experts=new_expert_models, gate= gate_model, device=device).to(device)

            optimizer_moe = optim.Adam(moe_model.parameters(), lr=0.001, amsgrad=False)


            hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer_moe = optimizer_moe,
                                   T = T, accuracy=accuracy, epochs=num_epochs)
            val['experts'][total_experts] = {'model':moe_model, 'history':hist}

        plot_file = generate_plot_file('new_'+m, T[0], w_importance=w_importance, w_sample_sim_same=w_sample_sim_same,w_sample_sim_diff=w_sample_sim_diff,
                                       specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

        if os.path.exists(os.path.join(model_path, plot_file)):
            n_run_models_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'))

        n_run_models_1.append(models)                                
        torch.save(n_run_models_1,open(os.path.join(model_path, plot_file),'wb'))
        n_run_models_1 = []
        print(plot_file)        

