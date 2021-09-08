import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

import moe_models

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

# The moe architecture that outputs an expected output of the experts
# based on the gate probabilities
class moe_em_model(nn.Module):
    
    def __init__(self, num_experts, num_classes, augment, attention_flag, experts, gate, task='classification'):
        super(moe_em_model,self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.augment = augment
        self.experts = experts.to(device)
        self.gate = gate.to(device)
        self.expert_outputs = None
        self.gate_outputs = None
        self.attention = attention_flag
        if self.attention:
            self.attn = moe_models.attention(num_experts, num_classes)
        self.task = task
        
    def forward(self,inputs, T=1.0):
        y = []
        for i, expert in enumerate(self.experts):
            y.append(expert(inputs))
        y = torch.stack(y)
        y.transpose_(0,1)
        
        self.expert_outputs = y

        p = self.gate(inputs, T)
        
        self.gate_outputs = p
        
        E = torch.argmax(p, dim=1)

        return y[range(y.shape[0]), E]
    
    def train(self, trainloader, testloader,
              loss_c, optimizer_moe, optimizer_gate=None, optimizer_experts=None, 
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
                   'kl_div_gate':[], 'kl_div_gate_T':[],
                   'per_exp_avg_wts':[], 'gate_avg_wts':[], 'Temp':[],
                   'w_importance':w_importance, 'w_ortho':w_ortho, 'w_ideal_gate':w_ideal_gate,
                   'cv':[], 'cv_T':[]}

        trainset = trainloader.dataset
        testset = testloader.dataset

        a = np.asarray([np.random.randint(self.num_experts) for i in range(len(trainset))])

        gate_probabilities_all_epochs = []

        batchsize = 32
        I_len = 128
        for s in range(epochs):

            running_loss = 0.0
            train_running_accuracy = 0.0 
            test_running_accuracy = 0.0

            expert_train_running_accuracy = torch.zeros(self.num_experts)
            expert_val_running_accuracy = torch.zeros(self.num_experts)
            expert_sample_train_running_accuracy = torch.zeros(self.num_experts)
            expert_sample_val_running_accuracy = torch.zeros(self.num_experts)
            expert_sample_train_running_accuracy_T = torch.zeros(self.num_experts)

            expert_train_running_loss = torch.zeros(self.num_experts)
            expert_sample_train_running_loss = torch.zeros(self.num_experts)
            expert_sample_train_running_loss_T = torch.zeros(self.num_experts)

            per_exp_class_samples = torch.zeros(self.num_experts, self.num_classes)
            
            running_entropy = 0.0

            ey =  np.zeros((self.num_classes, self.num_experts))

            gate_probabilities = []
            gate_probabilities_high_T = []

            expert_outputs_epoch = [] 

            per_exp_avg_wts = torch.zeros(self.num_experts)
            for i in range(0, self.num_experts):
                num_params = 0
                for param in self.experts[i].parameters():
                    per_exp_avg_wts[i] += param.mean()
                    num_params += 1
                per_exp_avg_wts[i] = per_exp_avg_wts[i]/num_params

            gate_avg_wts = 0.0
            num_params = 0
            for param in self.gate.parameters():
                gate_avg_wts += param.mean()
                num_params += 1
            gate_avg_wts = gate_avg_wts/num_params

            all_labels = []

            I = [np.random.randint(len(trainset)) for i in range(I_len)]
            new_batch = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, I), batch_size=len(I),
                                                    shuffle=True, num_workers=1, pin_memory=True)

            for x, y in new_batch:
                gate_output = self.gate(x)
                expected_values = []
                for i, expert in enumerate(self.experts):
                    expert_output = expert(x)
                    if self.task=='classification':
                        expected_values.append(expert_output[range(I_len), y]*gate_output[:,i])
                    else:
                        loss_criterion = loss_c(reduction='none')
                        expected_values.append(torch.mean(loss_criterion(expert_output, y), dim=1)*gate_output[:,i])
                expected_values = torch.stack(expected_values, dim=1)

            if self.task=='classification':
                E = torch.argmax(expected_values, dim=1)
            else:
                E = torch.argmin(expected_values, dim=1)

            a[I] = E

            num_iterations = 100
            for _ in range(num_iterations):
                 
                I = [np.random.randint(len(trainset)) for i in range(batchsize)]
                new_batch = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, I), batch_size=len(I),
                                                        shuffle=True, num_workers=1, pin_memory=True)

                for inputs, labels in new_batch:
                    E = a[I]
                
                    output = self(inputs)
                    gate_outputs = self.gate_outputs
                    expert_outputs = self.expert_outputs
                    gate_probabilities.append(gate_outputs)
                    
                    optimizer_moe.zero_grad()
                    
                    loss = []
                    eps=1e-7
                    for i in range(batchsize):
                        expert_output = expert_outputs[i, E[i],:]
                        if self.task == 'classification':
                            loss.append(torch.log(expert_output.reshape(self.num_classes)[labels[i]]+eps)*gate_outputs[i,E[i]])
                        else:
                            loss.append(loss_criterion(expert_output, labels[i])*gate_outputs[i,E[i]])
                    if self.task == 'classification':
                        loss = -1*torch.mean(torch.stack(loss))
                    else:
                        loss = torch.mean(torch.stack(loss))
                    running_loss += loss.item()
                    loss.backward()
                    
                    optimizer_moe.step()
                    
                    outputs = self(inputs)
                    
                    acc = accuracy(outputs, labels)
                    train_running_accuracy += acc

                    for index in range(0, self.num_experts):
                        acc = accuracy(self.expert_outputs[:,index,:], labels, False)
                        exp_sample_acc =  torch.sum(self.gate_outputs[:, index].flatten()*acc)
                        expert_train_running_accuracy[index] += torch.sum(acc)
                        expert_sample_train_running_accuracy[index] += exp_sample_acc
                        
                        e_loss = loss_c(reduction='none')(expert_outputs[:,index,:], labels)
                        e_sample_loss = torch.sum(torch.matmul(gate_outputs[:, index].flatten(), e_loss))
                        expert_train_running_loss[index] += torch.sum(e_loss)
                        expert_sample_train_running_loss[index] += e_sample_loss
                    
                        for label in range(0, self.num_classes):
                            index_l = torch.where(labels==label)[0]
                            per_exp_class_samples[index][label] += (torch.mean(gate_outputs[index_l, index]))*len(index_l)

                    #computing entropy
                    running_entropy += moe_models.entropy(self.gate_outputs)

            gate_probabilities = torch.stack(gate_probabilities)
            new_shape = gate_probabilities.shape
            gate_probabilities = gate_probabilities.reshape(new_shape[0]*new_shape[1], new_shape[2])
            
            I = [np.random.randint(len(testset)) for i in range(batchsize)]
            new_batch = torch.utils.data.DataLoader(torch.utils.data.Subset(testset, I), batch_size=len(I),
                                                    shuffle=True, num_workers=1, pin_memory=True)
            for test_inputs, test_labels in new_batch:
                test_outputs = self(test_inputs)
                test_expert_outputs = self.expert_outputs
                test_gate_outputs = self.gate_outputs
                test_gate_probabilities = test_gate_outputs
                E = torch.argmax(test_gate_outputs, dim=1)
                
                pred = []
                for i in range(batchsize):
                    expert_output = self.expert_outputs[i,E[i],:]
                    pred.append(expert_output)
                pred = torch.stack(pred)
                test_running_accuracy = accuracy(pred, test_labels)

                for index in range(0, self.num_experts):
                    acc = accuracy(test_expert_outputs[:,index,:], test_labels, False)
                    expert_val_running_accuracy[index] += torch.sum(acc)
                    exp_sample_acc =  torch.sum(test_gate_outputs[:, index].flatten()*acc)
                    expert_sample_val_running_accuracy[index] += exp_sample_acc
            
                running_loss = running_loss / num_iterations
                with torch.no_grad():
                    train_running_accuracy = train_running_accuracy.cpu().numpy() / num_iterations
                    running_entropy = running_entropy.cpu().numpy() / num_iterations

            history['loss'].append(running_loss)
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)
            history['sample_entropy'].append(running_entropy)
            history['entropy'].append(moe_models.entropy(torch.mean(gate_probabilities, dim=0)))

            with torch.no_grad():
                history['per_exp_avg_wts'].append(per_exp_avg_wts.cpu().numpy())
                history['gate_avg_wts'].append(gate_avg_wts.cpu().numpy())
                history['expert_accuracy'].append((expert_train_running_accuracy/len(trainloader.dataset)).cpu().numpy())
                history['expert_sample_accuracy'].append((torch.div(expert_sample_train_running_accuracy,torch.mean(gate_probabilities, dim=0)*len(trainloader.dataset))).cpu().numpy())
                history['expert_val_accuracy'].append((expert_val_running_accuracy/len(testloader.dataset)).cpu().numpy())
                history['expert_sample_val_accuracy'].append((torch.div(expert_sample_val_running_accuracy,torch.sum(test_gate_probabilities, dim=0))).cpu().numpy())
                history['expert_loss'].append((expert_train_running_loss/len(trainloader.dataset)).cpu().numpy())
                history['expert_sample_loss'].append((torch.div(expert_sample_train_running_loss,
                                                                torch.mean(gate_probabilities, dim=0)*len(trainloader.dataset))).cpu().numpy())

                history['per_exp_class_samples'].append(per_exp_class_samples.cpu().numpy()/num_iterations)
                history['exp_samples'].append((torch.mean(gate_probabilities, dim = 0)*len(trainloader.dataset)).cpu().numpy())
                history['exp_samples_val'].append((torch.mean(test_gate_probabilities, dim = 0)*len(testloader.dataset)).cpu().numpy())
                history['mean_gate_log_probability'].append(torch.mean(torch.log(gate_probabilities), dim = 0).cpu().numpy())
                history['var_gate_log_probability'].append(torch.var(torch.log(gate_probabilities), dim = 0).cpu().numpy())
                history['mean_gate_probability'].append(torch.mean(gate_probabilities, dim = 0).cpu().numpy())
                history['var_gate_probability'].append(torch.var(gate_probabilities, dim = 0).cpu().numpy())
                history['kl_div_gate'].append(moe_models.kl_divergence(gate_probabilities, torch.mean(gate_probabilities, dim = 0).repeat(len(gate_probabilities),1)).item())
                history['cv'].append(moe_models.cv(gate_probabilities))

            print('step %d' % s,
                  'training loss %.2f' % running_loss,
                  ', training accuracy %.2f' % train_running_accuracy,
                  ', test accuracy %.2f' % test_running_accuracy)
            

        return history
