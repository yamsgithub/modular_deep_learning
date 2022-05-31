import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

from helper import moe_models
from moe_models.moe_models_base import moe_models_base

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

# The moe architecture that outputs an expected output of the experts
# based on the gate probabilities
class moe_expectation_model(moe_models_base):
    
    def __init__(self, num_experts=5, num_classes=10, augment=0, attention_flag=0, hidden=None, experts=None, gate=None, task='classification'):
        super(moe_expectation_model,self).__init__(num_experts, num_classes, augment, attention_flag, hidden, experts, gate, task)
        
    def forward(self,inputs, T=1.0):

        y = []
        h = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            y.append(expert_output.view(1,-1,self.num_classes))
            if self.attention:
                hidden_output = expert.hidden
                h.append(hidden_output.view(1,-1,hidden_output.shape[1]))

                
        y = torch.vstack(y).transpose_(0,1).to(device)
        
        self.expert_outputs = y

        if self.augment:
            output_aug = torch.flatten(y, start_dim=1)
            input_aug = torch.cat((inputs, output_aug), dim=1)
            p = self.gate(input_aug)
        elif self.attention:
            h = torch.vstack(h).transpose_(0,1).to(device)
            h_gate = self.gate(inputs)

            attention = self.attn(h, h_gate)

            # attention scores are the gate output
            p = attention 
        else:
            p = self.gate(inputs, T)
        
        self.gate_outputs = p
               
        # reshape gate output so probabilities correspond 
        # to each expert
        p = p.reshape(p.shape[0],p.shape[1], 1)

        # repeat probabilities number of classes times so
        # dimensions correspond
        p = p.repeat(1,1,y.shape[2])

        # expected sum of expert outputs
        output = torch.sum(p*y, 1)

        return output


    def train_on_validation(self, trainloader, testloader,
              loss_c, optimizer_moe=None, optimizer_gate=None, optimizer_experts=None, 
              w_importance = 0.0, w_ortho = 0.0, w_ideal_gate = 0.0,
              T=1.0, T_decay=0, T_decay_start=0,
              accuracy=None, epochs=10):

        loss_criterion = loss_c()
        
        history = {'loss':[], 'loss_importance':[], 'baseline_losses':[],
                   'accuracy':[], 'val_accuracy':[], 'expert_accuracy':[], 'expert_sample_accuracy':[],
                   'expert_accuracy_T':[], 'expert_sample_accuracy_T':[],
                   'expert_val_accuracy':[], 'expert_sample_val_accuracy':[],
                   'expert_loss':[], 'expert_sample_loss':[], 'expert_sample_loss_T':[],
                   'sample_entropy':[], 'entropy':[], 'EY':[],'mutual_EY':[], 'H_EY':[],'H_Y':[], 'H_E':[],
                   'per_exp_class_samples':[], 'exp_samples':[],'exp_samples_T':[], 'exp_samples_val':[], 
                   'mean_gate_log_probability': [],'var_gate_log_probability': [],
                   'mean_gate_probability': [],'var_gate_probability': [],
                   'mean_gate_log_probability_T': [],'var_gate_log_probability_T': [],
                   'mean_gate_probability_T': [],'var_gate_probability_T': [],
                   'sample_entropy_T':[], 'entropy_T':[], 
                   'kl_div_gate':[], 'kl_div_gate_T':[],
                   'per_exp_avg_wts':[], 'gate_avg_wts':[], 'Temp':[],
                   'w_importance':w_importance, 'w_ortho':w_ortho, 'w_ideal_gate':w_ideal_gate,
                   'cv':[], 'cv_T':[], 'gate_probabilities':[], 'gate_probabilities_T':[]}
        
        for epoch in range(epochs):  # loop over the dataset multiple times
            num_batches = 0
            running_loss = 0.0
            train_running_accuracy = 0.0 
            test_running_accuracy = 0.0

            expert_train_running_accuracy = torch.zeros(self.num_experts)
            expert_val_running_accuracy = torch.zeros(self.num_experts)
            expert_sample_train_running_accuracy = torch.zeros(self.num_experts)
            expert_sample_val_running_accuracy = torch.zeros(self.num_experts)

            expert_train_running_loss = torch.zeros(self.num_experts)
            expert_sample_train_running_loss = torch.zeros(self.num_experts)

            per_exp_class_samples = torch.zeros(self.num_experts, self.num_classes)
            
            running_entropy = 0.0

            ey =  np.zeros((self.num_classes, self.num_experts))

            per_exp_avg_wts = torch.zeros(self.num_experts)
            for i in range(0, self.num_experts):
                num_params = 0
                for param in self.experts[i].parameters():
                    per_exp_avg_wts[i] = per_exp_avg_wts[i] + param.mean()
                    num_params += 1
                per_exp_avg_wts[i] = per_exp_avg_wts[i]/num_params

            gate_avg_wts = 0.0
            num_params = 0
            for param in self.gate.parameters():
                gate_avg_wts = gate_avg_wts + param.mean()
                num_params += 1
            gate_avg_wts = gate_avg_wts/num_params

            all_labels = None 

            gate_probabilities = []
            
            num_batches = 0
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                if all_labels is None:
                   all_labels = labels
                else:
                   all_labels = torch.cat((all_labels,labels))
                
                outputs = self(inputs)
                gate_outputs = self.gate_outputs
                gate_probabilities.append(gate_outputs)
                expert_outputs = self.expert_outputs

                # zero the parameter gradients
                optimizer_gate.zero_grad()
                loss = loss_criterion(outputs, labels)
                running_loss += loss.item()

                loss.backward()
                
                optimizer_gate.step()
                

                outputs = self(inputs)
                
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc

                #computing entropy
                running_entropy += moe_models.entropy(gate_outputs)

                num_batches += 1

            running_loss = running_loss/num_batches
            with torch.no_grad():
                train_running_accuracy = train_running_accuracy.cpu().numpy()/num_batches
                running_entropy = running_entropy.cpu().numpy()/num_batches

            with torch.no_grad():
                acc = 0.0
                j = 0
                test_gate_probabilities = []
                for test_inputs, test_labels in testloader:
               	    test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)                
                    test_outputs = self(test_inputs)
                    test_gate_outputs = self.gate_outputs
                    test_expert_outputs = self.expert_outputs
                    test_gate_probabilities.append(test_gate_outputs)
                    test_running_accuracy += accuracy(test_outputs, test_labels)
                    
                    j += 1
                    
                test_running_accuracy = test_running_accuracy.cpu().numpy()/j
                
            gate_probabilities = torch.vstack(gate_probabilities)

            history['loss'].append(running_loss)
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)
            history['sample_entropy'].append(running_entropy)
            history['entropy'].append(moe_models.entropy(torch.mean(gate_outputs, dim=0)))
            
            with torch.no_grad():
                history['per_exp_avg_wts'].append(per_exp_avg_wts.cpu().numpy())
                history['gate_avg_wts'].append(gate_avg_wts.cpu().numpy())
                history['mean_gate_log_probability'].append(torch.mean(torch.log(gate_probabilities), dim = 0).cpu().numpy())
                history['var_gate_log_probability'].append(torch.var(torch.log(gate_probabilities), dim = 0).cpu().numpy())
                history['mean_gate_probability'].append(torch.mean(gate_probabilities, dim = 0).cpu().numpy())
                history['var_gate_probability'].append(torch.var(gate_probabilities, dim = 0).cpu().numpy())
                history['kl_div_gate'].append(moe_models.kl_divergence(gate_probabilities.cpu(), torch.mean(gate_probabilities, dim = 0).cpu().repeat(len(gate_probabilities),1)).item())
                history['cv'].append(moe_models.cv(gate_probabilities))

            print('epoch %d' % epoch,
                  'training loss %.2f' % running_loss,
                  ', training accuracy %.2f' % train_running_accuracy,
                  ', test accuracy %.2f' % test_running_accuracy)
        print('val accuracy', history['val_accuracy'])    
        return history
