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

def two_temp_optim(model, inputs, labels, outputs, T, optimizer_gate, optimizer_experts, loss_criterion):
    outputs_with_T = model(inputs, T)
    gate_outputs_T = model.gate_outputs

    optimizer_experts.zero_grad()
    loss_experts = loss_criterion(outputs_with_T, labels)
    loss_experts.backward()
    
    optimizer_gate.zero_grad()
    loss = loss_criterion(outputs, labels)
    for i, expert in enumerate(model.experts):
        for param in expert.parameters():
            param.requires_grad = False
    loss.backward()
    for i, expert in enumerate(model.experts):
        for param in expert.parameters():
            param.requires_grad = True
    optimizer_gate.step()
    optimizer_experts.step()

    return loss, gate_outputs_T

# The moe architecture that outputs an expected output of the experts
# based on the gate probabilities
class moe_expectation_model(nn.Module):
    
    def __init__(self, num_experts, num_classes, augment, attention_flag, experts, gate, task='classification'):
        super(moe_expectation_model,self).__init__()
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

        if self.augment:
            output_aug = torch.flatten(y, start_dim=1)
            input_aug = torch.cat((inputs, output_aug), dim=1)
            p = self.gate(input_aug)
        elif self.attention:
            context = self.attn(inputs, y)
            input_aug = torch.cat((inputs, context.squeeze(1)), dim=1)
            p = self.gate(input_aug, T)
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
        
        gate_probabilities_all_epochs = []
        gate_probabilities_all_epochs_T = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            num_batches = 0
            running_loss = 0.0
            running_loss_importance = 0.0
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
            
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                all_labels.append(labels)

                outputs = self(inputs)
                gate_outputs = self.gate_outputs
                expert_outputs = self.expert_outputs
                expert_outputs_epoch.append(expert_outputs)
                gate_probabilities.append(gate_outputs)

                
                if T > 1.0:
                    loss, gate_probabilities_batch_high_T = two_temp_optim(self, inputs, labels, outputs, T,
                                                                           optimizer_gate, optimizer_experts, loss_criterion)
                    gate_probabilities_high_T.append(gate_probabilities_batch_high_T)

                else:
                    # zero the parameter gradients
                    optimizer_moe.zero_grad()
                    loss = loss_criterion(outputs, labels)
                    if w_ideal_gate > 0.0 and self.num_experts > 1:
                        l_experts = []
                        for i in range(self.num_experts):
                            l_experts.append(loss_c(reduction='none')(expert_outputs[:,i,:],labels))
                        l_experts = torch.stack(l_experts)
                        l_experts.transpose_(0,1)
                        min_indices = torch.min(l_experts, dim=1)[1]
                        ideal_gate_output = torch.zeros((len(min_indices), self.num_experts))
                        for i, index in enumerate(min_indices):
                            ideal_gate_output[i, index] = 1
                        ideal_gate_loss_criterion = nn.MSELoss()
                        ideal_gate_loss = ideal_gate_loss_criterion(gate_outputs, ideal_gate_output)
                        loss += ideal_gate_loss
                        
                    
                    l_imp = 0.0
                    if w_importance > 0.0:
                        l_imp = moe_models.loss_importance(gate_outputs, w_importance)
                        loss += l_imp
                        running_loss_importance += l_imp

                    if w_ortho > 0.0:
                        l_ortho =  None
                        for i in range(0, self.expert_outputs.shape[1]-1):
                            for j in range(i+1, self.expert_outputs.shape[1]):
                                if l_ortho is None:
                                    l_ortho = torch.abs(torch.matmul(self.expert_outputs[:,i,:].squeeze(1),
                                                           torch.transpose(self.expert_outputs[:,j,:].squeeze(1), 0, 1)))
                                else:
                                    l_ortho = torch.add(l_ortho, torch.abs(torch.matmul(self.expert_outputs[:,i,:].squeeze(1),
                                                                                        torch.transpose(self.expert_outputs[:,j,:].squeeze(1), 0, 1))))
                        if not l_ortho is None:
                            loss += w_ortho * l_ortho.mean()

                    loss.backward()

                    optimizer_moe.step()

                running_loss += loss.item()


                outputs = self(inputs)
                
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc

                for index in range(0, self.num_experts):
                    acc = accuracy(self.expert_outputs[:,index,:], labels, False)
                    exp_sample_acc =  torch.sum(gate_outputs[:, index].flatten()*acc)
                    expert_train_running_accuracy[index] += torch.sum(acc)
                    expert_sample_train_running_accuracy[index] += exp_sample_acc

                    e_loss = loss_c(reduction='none')(expert_outputs[:,index,:], labels)
                    e_sample_loss = torch.sum(torch.matmul(gate_outputs[:, index].flatten(), e_loss))
                    expert_train_running_loss[index] += torch.sum(e_loss)
                    expert_sample_train_running_loss[index] += e_sample_loss
                    
                    for label in range(0, self.num_classes):
                        index_l = torch.where(labels==label)[0]
                        per_exp_class_samples[index][label] += (torch.mean(gate_outputs[index_l, index]))*len(index_l)
                        
                    if T > 1.0:
                        exp_sample_acc =  torch.sum(gate_probabilities_batch_high_T[:, index].flatten()*acc)
                        expert_sample_train_running_accuracy_T[index] += exp_sample_acc
                        
                        e_sample_loss = torch.sum(gate_probabilities_batch_high_T[:, index].flatten()*e_loss)
                        expert_sample_train_running_loss_T[index] += e_sample_loss


                        
                #computing entropy
                running_entropy += moe_models.entropy(self.gate_outputs)
                
                # update the Y vs E table to compute joint distribution of Y and E
                if self.task == 'classification':
                    selected_experts = torch.zeros(len(labels))
                    if self.num_experts > 1:
                        selected_experts = torch.argmax(self.gate_outputs, dim=1)
                    y = labels.numpy()
                    e = selected_experts.numpy()
                    for j in range(y.shape[0]):
                        ey[int(y[j]), int(e[j])] += 1

                    mutual_EY, H_EY, H_E, H_Y = moe_models.mutual_information(ey)

                num_batches+=1

            
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

                    for index in range(0, self.num_experts):
                        acc = accuracy(test_expert_outputs[:,index,:], test_labels, False)
                        expert_val_running_accuracy[index] += torch.sum(acc)
                        exp_sample_acc =  torch.sum(test_gate_outputs[:, index].flatten()*acc)
                        expert_sample_val_running_accuracy[index] += exp_sample_acc

                    j += 1
                #print(confusion_matrix(testloader.dataset[:][1], torch.argmax(test_outputs_all, dim=1)))
                    
                test_running_accuracy = test_running_accuracy/j

            running_loss = running_loss / num_batches
            running_loss_importance = running_loss_importance / num_batches
            with torch.no_grad():
                train_running_accuracy = train_running_accuracy.cpu().numpy() / num_batches
                running_entropy = running_entropy.cpu().numpy() / num_batches

            gate_probabilities = torch.stack(gate_probabilities)
            new_shape = gate_probabilities.shape
            gate_probabilities = gate_probabilities.reshape(new_shape[0]*new_shape[1], new_shape[2])

            gate_probabilities_all_epochs.append(gate_probabilities)

            test_gate_probabilities = torch.stack(test_gate_probabilities)
            new_shape = test_gate_probabilities.shape
            test_gate_probabilities = test_gate_probabilities.reshape(new_shape[0]*new_shape[1], new_shape[2])

            all_labels = torch.stack(all_labels).flatten()
            
            baseline_losses = []
            if self.task == 'classification':
                #loss baseline with avg gate prob
                l = all_labels
                y = torch.stack(expert_outputs_epoch)
                #print('expert outputs', y.shape)
                y = y.reshape(y.shape[0]*y.shape[1], y.shape[2], y.shape[3])
                #print('expert outputs', y.shape)
                p = torch.mean(gate_probabilities, dim=0)
                #print('p', p.shape)
                p = p.reshape(1, p.shape[0], 1)
                #print('p', p.shape)
                p = p.repeat(y.shape[0],1,y.shape[2])
                #print('p', p.shape)
                # expected sum of expert outputs
                output = torch.sum(p*y, 1)
                baseline_losses.append(loss_criterion(output, l))
                
                base_class_freq = torch.unique(l,return_counts=True)[1]/float(l.shape[0])
                y = base_class_freq.repeat(y.shape[0]*self.num_experts).reshape(y.shape)
                p = gate_probabilities
                p = p.reshape(p.shape[0],p.shape[1], 1)
                p = p.repeat(1,1,y.shape[2])
                output = torch.sum(p*y, 1)
                baseline_losses.append(loss_criterion(output, l))
                history['baseline_losses'].append(baseline_losses)

            history['loss'].append(running_loss)
            history['loss_importance'].append(running_loss_importance)            
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)
            history['sample_entropy'].append(running_entropy)
            history['entropy'].append(moe_models.entropy(torch.mean(gate_probabilities, dim=0)))
            
            if self.task == 'classification':
                history['EY'].append(ey)
                history['mutual_EY'].append(mutual_EY)
                history['H_EY'].append(H_EY)
                history['H_E'].append(H_E)
                history['H_Y'].append(H_Y)
                
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

                history['per_exp_class_samples'].append(per_exp_class_samples.cpu().numpy()/num_batches)
                history['exp_samples'].append((torch.mean(gate_probabilities, dim = 0)*len(trainloader.dataset)).cpu().numpy())
                history['exp_samples_val'].append((torch.mean(test_gate_probabilities, dim = 0)*len(testloader.dataset)).cpu().numpy())
                history['mean_gate_log_probability'].append(torch.mean(torch.log(gate_probabilities), dim = 0).cpu().numpy())
                history['var_gate_log_probability'].append(torch.var(torch.log(gate_probabilities), dim = 0).cpu().numpy())
                history['mean_gate_probability'].append(torch.mean(gate_probabilities, dim = 0).cpu().numpy())
                history['var_gate_probability'].append(torch.var(gate_probabilities, dim = 0).cpu().numpy())
                history['kl_div_gate'].append(moe_models.kl_divergence(gate_probabilities, torch.mean(gate_probabilities, dim = 0).repeat(len(gate_probabilities),1)).item())
                history['cv'].append(moe_models.cv(gate_probabilities))
                
            if T> 1.0:
                gate_probabilities_high_T = torch.stack(gate_probabilities_high_T)
                new_shape = gate_probabilities_high_T.shape
                gate_probabilities_high_T = gate_probabilities_high_T.reshape(new_shape[0]*new_shape[1], new_shape[2])
                gate_probabilities_all_epochs_T.append(gate_probabilities_high_T)
                
                with torch.no_grad():
                    history['expert_sample_accuracy_T'].append((torch.div(expert_sample_train_running_accuracy_T,torch.mean(gate_probabilities_high_T, dim=0)*len(trainloader.dataset))).cpu().numpy())
                    history['expert_sample_loss_T'].append((torch.div(expert_sample_train_running_loss_T,
                                                                      torch.mean(gate_probabilities_high_T, dim=0)*len(trainloader.dataset))).cpu().numpy())
                    history['exp_samples_T'].append((torch.mean(gate_probabilities_high_T, dim = 0)*len(trainloader.dataset)).cpu().numpy())
                    history['mean_gate_log_probability_T'].append(torch.mean(torch.log(gate_probabilities_high_T), dim = 0).numpy())
                    history['var_gate_log_probability_T'].append(torch.var(torch.log(gate_probabilities_high_T), dim = 0).numpy())
                    history['mean_gate_probability_T'].append(torch.mean(gate_probabilities_high_T, dim = 0).numpy())
                    history['var_gate_probability_T'].append(torch.var(gate_probabilities_high_T, dim = 0).numpy())
                    history['kl_div_gate_T'].append(moe_models.kl_divergence(gate_probabilities_high_T, torch.mean(gate_probabilities_high_T, dim = 0).repeat(len(gate_probabilities_high_T),1)).item())
                    history['cv_T'].append(moe_models.cv(gate_probabilities_high_T))
            history['Temp'].append(T)
            print('epoch %d' % epoch,
                  'training loss %.2f' % running_loss,
                  ', training accuracy %.2f' % train_running_accuracy,
                  ', test accuracy %.2f' % test_running_accuracy)
            
            if epoch > T_decay_start:
                T *= (1. / (1. + T_decay * epoch))

        return history
