import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

from torch.profiler import profile, record_function, ProfilerActivity

from helper import moe_models

class two_temp_optim:
    def __init__(self, optimizer_moe, optimizer_gate, optimizer_experts):
        self.optimizer_moe = optimizer_moe
        self.optimizer_gate = optimizer_gate
        self.optimizer_experts = optimizer_experts
        
    def optimise(self, model=None, inputs=None, labels=None, 
                 outputs=None, expert_outputs=None, gate_outputs=None, 
                 loss_criterion=None, regularization=0.0, T=1.0):
        expert_outputs_T = model.expert_outputs
        gate_outputs_T = model.gate_outputs

        optimizer_experts.zero_grad(set_to_none=True)
        loss_experts = loss_criterion(outputs_with_T, expert_outputs_T, gate_outputs_T, labels)
        loss_experts.backward()

        for expert in model.experts:
            for param in expert.parameters():
                param.requires_grad = False

        optimizer_gate.zero_grad(set_to_none=True)
        loss = loss_criterion(outputs, expert_outputs, gate_outputs, labels)

        if not regularization == 0.0:
            loss += regularization

        loss.backward()

        for expert in model.experts:
            for param in expert.parameters():
                param.requires_grad = True 

        optimizer_gate.step()
        optimizer_experts.step()

        self.gate_probabilities_batch_high_T = gate_outputs_T
        
        return loss

class default_optimizer:
    def __init__(self, optimizer_moe=None, optimizer_gate=None, optimizer_experts=None):
        self.optimizer_moe = optimizer_moe
        
    def optimise(self, model=None, inputs=None, labels=None, outputs=None, expert_outputs=None, gate_outputs=None, 
                 loss_criterion=None, regularization=0.0, T=1.0):
        # zero the parameter gradients
        self.optimizer_moe.zero_grad(set_to_none=True)
        
        loss = loss_criterion(outputs, expert_outputs, gate_outputs, labels)
        if not regularization == 0: 
            loss += regularization   

        loss.backward()

        self.optimizer_moe.step()
        
        return loss

class expert_loss_gate_optimizer:
    def __init__(self, optimizer_moe=None, optimizer_gate=None, optimizer_experts=None):
        self.optimizer_moe = optimizer_moe
        self.optimizer_gate = optimizer_gate
        self.optimizer_experts = optimizer_experts
        self.prev_epoch = 0
        self.count = 1
        
    def optimise(self, model=None, inputs=None, labels=None, outputs=None, expert_outputs=None, gate_outputs=None, 
                 loss_criterion=None, regularization=0.0, epoch=0):
        
        # zero the parameter gradients
        if not self.optimizer_experts is None:
            self.optimizer_experts.zero_grad(set_to_none=True)                 
            expert_loss = loss_criterion(outputs=outputs, targets=labels)
            expert_loss.backward()
        else:
            print('Expert optimizer missing')

        self.optimizer_gate.zero_grad(set_to_none=True)
        gate_loss_criterion = moe_models.expert_entropy_loss()
        gate_loss = gate_loss_criterion(expert_outputs=expert_outputs.detach(), gate_outputs=gate_outputs, targets=labels)
        gate_loss.backward()
        
        self.optimizer_experts.step()
        self.optimizer_gate.step()
        
        moe_loss_criterion = moe_models.cross_entropy_loss()
        loss = moe_loss_criterion(outputs=outputs, targets=labels)
        
        return loss

def default_distance_funct(inputs):
    batch_size = len(inputs) # batch size
    # sum up the channels to make the images 2D
    imgs = inputs.view(batch_size,-1)
    dist = torch.cdist(imgs, imgs, compute_mode='donot_use_mm_for_euclid_dist')
    # normalize
    dist = dist/torch.max(dist)
    
    return dist

class resnet_distance_funct:
    
    def __init__(self, resnet_model=None):
        self.resnet_model = resnet_model
    
    def distance_funct(self, inputs):
        batch_size = len(inputs)
        features = self.resnet_model(inputs)
        dist = torch.cdist(features, features, compute_mode='donot_use_mm_for_euclid_dist')
        # normalize
        dist = dist/torch.max(dist)
        return dist
    
# The moe architecture that outputs an expected output of the experts
# based on the gate probabilities
class moe_models_base(nn.Module):
    
    def __init__(self, num_experts=5, num_classes=10, augment=0, attention_flag=0, hidden=None, softmax=True, experts=None, gate=None, task='classification', device = torch.device("cpu")):
        super(moe_models_base,self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.augment = augment
        self.experts = experts.to(device)
        self.gate = gate
        if not gate is None:
            self.gate = gate.to(device)
        self.expert_outputs = None
        self.gate_outputs = None
        self.attention = attention_flag
        if self.attention:
            self.attn = moe_models.attention(hidden, softmax).to(device)
        self.task = task
        self.device = device
        
    # For updating learning rate
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def train(self, trainloader, testloader,
              loss_criterion, optimizer=None, 
              w_importance = 0.0, w_ortho = 0.0, w_ideal_gate = 0.0,
              w_sample_sim_same = 0.0, w_sample_sim_diff = 0.0, distance_funct = default_distance_funct,
              T=[1.0]*20, T_decay=0.0, T_decay_start=0, no_gate_T = [1.0]*20,
              accuracy=None, epochs=10, model_name='moe_expectation_model'):

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
                   'cv':[], 'cv_T':[], 'gate_probabilities':[],'gate_probabilities_T':[], 'per_sample_entropy':[] }
        
        gate_probabilities_all_epochs = []
        gate_probabilities_all_epochs_T = []
        curr_lr = 0.001
        for epoch in range(epochs):  # loop over the dataset multiple times
            num_batches = 0
            running_loss = 0.0
            running_loss_importance = 0.0
            train_running_accuracy = 0.0 
            test_running_accuracy = 0.0

            expert_train_running_accuracy = torch.zeros(self.num_experts, device=self.device)
            expert_val_running_accuracy = torch.zeros(self.num_experts, device=self.device)
            expert_sample_train_running_accuracy = torch.zeros(self.num_experts, device=self.device)
            expert_sample_val_running_accuracy = torch.zeros(self.num_experts, device=self.device)
            expert_sample_train_running_accuracy_T = torch.zeros(self.num_experts, device=self.device)

            expert_train_running_loss = torch.zeros(self.num_experts, device=self.device)
            expert_sample_train_running_loss = torch.zeros(self.num_experts, device=self.device)
            expert_sample_train_running_loss_T = torch.zeros(self.num_experts, device=self.device)

            per_exp_class_samples = torch.zeros(self.num_experts, self.num_classes, device=self.device)
            
            running_entropy = 0.0

            if not T[epoch] == 1.0:
               running_entropy_T = 0.0
               gate_probabilities_high_T = []

            ey =  torch.zeros((self.num_classes, self.num_experts)).to(self.device)

            gate_probabilities = []

            expert_outputs_epoch = []

            sample_entropies = []

            per_exp_avg_wts = torch.zeros(self.num_experts)
            for i in range(0, self.num_experts):
                num_params = 0
                for param in self.experts[i].parameters():
                    per_exp_avg_wts[i] = per_exp_avg_wts[i] + param.mean()
                    num_params += 1
                per_exp_avg_wts[i] = per_exp_avg_wts[i]/num_params

            gate_avg_wts = 0.0
            if not self.gate is None:
                num_params = 0
                for param in self.gate.parameters():
                    gate_avg_wts = gate_avg_wts + param.mean()
                    num_params += 1
                gate_avg_wts = gate_avg_wts/num_params

            all_labels = []
            
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                all_labels.append(labels)

                if model_name == 'moe_no_gate_model':
                    outputs = self(inputs, targets=labels, T=no_gate_T[epoch])
                else:
                    outputs = self(inputs)
    
                gate_outputs = self.gate_outputs
                expert_outputs = self.expert_outputs

                expert_outputs_epoch.append(expert_outputs)
                gate_probabilities.append(gate_outputs)
                if model_name == 'moe_no_gate_model':
                    sample_entropies.append(self.per_sample_entropy)

                regularization = 0.0

                if w_sample_sim_same > 0.0 or w_sample_sim_diff > 0.0:
                   batch_size = len(inputs)
                   dist = distance_funct(inputs)
                
                   sample_dist = 0
                   
                   pex_dist_same = torch.zeros(batch_size, batch_size).to(self.device)
                   pex_dist_diff = torch.zeros(batch_size, batch_size).to(self.device)
                   for i in range(self.num_experts):
                       pe_i = gate_outputs[:,i].view(-1, 1)
                       for j in range(self.num_experts):
                           pe_j = gate_outputs[:,j].view(1, -1)
                           pex_j = torch.mul(pe_i, pe_j)
                           pex_j_dist = torch.mul(pex_j, dist)
                           if i == j:
                              pex_dist_same = pex_dist_same + (pex_j_dist/self.num_experts)
                           else:
                              pex_dist_diff = pex_dist_diff + (pex_j_dist/(self.num_experts**2-self.num_experts))
                   
                   reg_similarity = ((w_sample_sim_same * torch.sum(pex_dist_same)) - (w_sample_sim_diff * torch.sum(pex_dist_diff))/(batch_size**2))
                   regularization += reg_similarity

                if w_importance > 0.0:
                   l_imp = moe_models.loss_importance(gate_outputs, w_importance)
#                    print('l_imp_reg', l_imp)
                   running_loss_importance += l_imp
                   regularization += l_imp

                if w_ortho > 0.0:
                   p = gate_outputs
                   y = expert_outputs
                   p = p.reshape(p.shape[0],p.shape[1], 1)
                   p = p.repeat(1,1,self.num_classes)
                   regularization += moe_models.loss_importance(torch.sum(p*y, dim=1), w_ortho)

                if w_ideal_gate > 0.0 and self.num_experts > 1:
                        l_experts = []
                        for i in range(self.num_experts):
                            loss_criterion.reduction('none')
                            l_experts.append(loss_criterion(expert_outputs[:,i,:],labels))
                            loss_criterion.reduction(loss_criterion.default_reduction)
                        l_experts = torch.vstack(l_experts)
                        min_indices = torch.min(l_experts, dim=1)[1]
                        ideal_gate_output = torch.zeros((len(min_indices), self.num_experts))
                        for i, index in enumerate(min_indices):
                            ideal_gate_output[i, index] = 1
                        ideal_gate_loss_criterion = nn.MSELoss()
                        ideal_gate_loss = ideal_gate_loss_criterion(gate_outputs, ideal_gate_output)
                        regularization += ideal_gate_loss
                
 
                loss = optimizer.optimise(self, inputs, labels, outputs, expert_outputs, gate_outputs,  
                                 loss_criterion, regularization, epoch)
                if not T[epoch] == 1.0:
                   loss, gate_probabilities_batch_high_T = optimizer.gate_probabilities_batch_high_T
                   gate_probabilities_high_T.append(gate_probabilities_batch_high_T)
                   running_entropy_T += moe_models.entropy(gate_probabilities_batch_high_T)                   

                running_loss += loss

                with torch.no_grad():
                    if model_name == 'moe_no_gate_model':
                        outputs = self(inputs, targets=labels, T=no_gate_T[epoch])
                    else:
                        outputs = self(inputs)
                
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc

                # for index in range(0, self.num_experts):
                #     acc = accuracy(self.expert_outputs[:,index,:], labels, False)
                #     exp_sample_acc =  torch.sum(gate_outputs[:, index].flatten()*acc)
                #     expert_train_running_accuracy[index] = expert_train_running_accuracy[index] + torch.sum(acc)
                #     expert_sample_train_running_accuracy[index] = expert_sample_train_running_accuracy[index] + exp_sample_acc

                #     if model_name == 'moe_expectation_model':
                #         loss_criterion.reduction('none')
                #         e_loss = loss_criterion(expert_outputs[:,index,:], None, None, labels)
                #         loss_criterion.reduction(loss_criterion.default_reduction)
                #     elif model_name == 'moe_stochastic_model' or model_name == 'moe_no_gate_model':
                #         e_loss = loss_criterion.loss_criterion(expert_outputs[:,index,:], None, None, labels)
                    
                #     e_sample_loss = torch.sum(torch.matmul(gate_outputs[:, index].flatten(), e_loss))
                #     expert_train_running_loss[index] =  expert_train_running_loss[index] + torch.sum(e_loss)
                #     expert_sample_train_running_loss[index] = expert_sample_train_running_loss[index] + e_sample_loss
                    
                #     for label in range(0, self.num_classes):
                #         index_l = torch.where(labels==label)[0]
                #         per_exp_class_samples[index][label] = per_exp_class_samples[index][label] + (torch.mean(gate_outputs[index_l, index]))*len(index_l)
                        
                #     if not T[epoch] == 1.0:
                #         exp_sample_acc =  torch.sum(gate_probabilities_batch_high_T[:, index].flatten()*acc)
                #         expert_sample_train_running_accuracy_T[index] = expert_sample_train_running_accuracy_T[index] + exp_sample_acc
                        
                #         e_sample_loss = torch.sum(gate_probabilities_batch_high_T[:, index].flatten()*e_loss)
                #         expert_sample_train_running_loss_T[index] =  expert_sample_train_running_loss_T[index] + e_sample_loss
                        
                #computing entropy
                running_entropy += moe_models.entropy(self.gate_outputs)
                
                # update the Y vs E table to compute joint distribution of Y and E
                if self.task == 'classification':
                    selected_experts = torch.argmax(self.gate_outputs, dim=1)
                    y = labels
                    e = selected_experts
                    for j in range(y.shape[0]):
                        ey[int(torch.argmax(expert_outputs[j,e[j],:])), int(e[j])] += 1

                num_batches+=1
 
            mutual_EY, H_EY, H_E, H_Y = moe_models.mutual_information(ey.detach())

            with torch.no_grad():
                acc = 0.0
                j = 0
                test_gate_probabilities = []
                for test_inputs, test_labels in testloader:
               	    test_inputs, test_labels = test_inputs.to(self.device, non_blocking=True), test_labels.to(self.device, non_blocking=True)                
                    with torch.no_grad():
                        if model_name == 'moe_no_gate_model':
                            test_outputs = self(test_inputs, targets=test_labels, T=no_gate_T[epoch])
                        else:
                            test_outputs = self(test_inputs)
                    test_gate_outputs = self.gate_outputs
                    test_expert_outputs = self.expert_outputs
                    test_gate_probabilities.append(test_gate_outputs)
                    test_running_accuracy += accuracy(test_outputs, test_labels)

                    # for index in range(0, self.num_experts):
                    #     acc = accuracy(test_expert_outputs[:,index,:], test_labels, False)
                    #     expert_val_running_accuracy[index] = expert_val_running_accuracy[index] + torch.sum(acc)
                    #     exp_sample_acc =  torch.sum(test_gate_outputs[:, index].flatten()*acc)
                    #     expert_sample_val_running_accuracy[index] = expert_sample_val_running_accuracy[index] + exp_sample_acc

                    j += 1
                    
                test_running_accuracy = test_running_accuracy/j

            running_loss = running_loss / num_batches
            running_loss_importance = running_loss_importance / num_batches
            with torch.no_grad():
                train_running_accuracy = train_running_accuracy / num_batches
                running_entropy = running_entropy / num_batches
                if not T[epoch] == 1.0:
                    running_entropy_T = running_entropy_T / num_batches

            gate_probabilities = torch.vstack(gate_probabilities)

            gate_probabilities_all_epochs.append(gate_probabilities)

            test_gate_probabilities = torch.vstack(test_gate_probabilities)

            if model_name == 'moe_no_gate_model':
                sample_entropies = torch.vstack(sample_entropies)
            
            # baseline_losses = []
            # if self.task == 'classification':
            #     #loss baseline with avg gate prob
            #     l = all_labels
            #     y = torch.vstack(expert_outputs_epoch)

            #     p = torch.mean(gate_probabilities, dim=0)
            #     p = p.reshape(1, p.shape[0], 1)
            #     p = p.repeat(y.shape[0],1,y.shape[2])
            #     # expected sum of expert outputs
            #     output = torch.sum(p*y, 1)
            #     if model_name == 'moe_expectation_model':
            #         baseline_losses.append(loss_criterion(output, None,None,l))
            #     elif model_name == 'moe_stochastic_model' or model_name == 'moe_no_gate_model':
            #         baseline_losses.append(loss_criterion.loss_criterion(output,None, None, l))                    
                    
                
            #     base_class_freq = torch.unique(l,return_counts=True)[1]/float(l.shape[0])
            #     y = base_class_freq.repeat(y.shape[0]*self.num_experts).reshape(y.shape)
            #     p = gate_probabilities
            #     p = p.reshape(p.shape[0],p.shape[1], 1)
            #     p = p.repeat(1,1,y.shape[2])
            #     output = torch.sum(p*y, 1)
            #     if model_name == 'moe_expectation_model':
            #         baseline_losses.append(loss_criterion(output,None, None, l))
            #     elif model_name == 'moe_stochastic_model' or model_name == 'moe_no_gate_model':
            #         baseline_losses.append(loss_criterion.loss_criterion(output,None,None, l))
            #     history['baseline_losses'].append(baseline_losses)

            history['loss'].append(running_loss)
            history['loss_importance'].append(running_loss_importance)            
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)
            history['sample_entropy'].append(running_entropy)
            history['entropy'].append(moe_models.entropy(torch.mean(gate_probabilities, dim=0)))
            if model_name == 'moe_no_gate_model':
                history['per_sample_entropy'].append(sample_entropies)
                
            if self.task == 'classification':
                history['EY'].append(ey)
                history['mutual_EY'].append(mutual_EY)
                history['H_EY'].append(H_EY)
                history['H_E'].append(H_E)
                history['H_Y'].append(H_Y)
                
            with torch.no_grad():
                history['per_exp_avg_wts'].append(per_exp_avg_wts)
                history['gate_avg_wts'].append(gate_avg_wts)
                history['expert_accuracy'].append((expert_train_running_accuracy/len(trainloader.dataset)))
                #history['expert_sample_accuracy'].append((torch.div(expert_sample_train_running_accuracy,torch.mean(gate_probabilities, dim=0)*len(trainloader.dataset))))
                history['expert_val_accuracy'].append((expert_val_running_accuracy/len(testloader.dataset)))
                history['expert_sample_val_accuracy'].append((torch.div(expert_sample_val_running_accuracy,torch.sum(test_gate_probabilities, dim=0))))
                history['expert_loss'].append((expert_train_running_loss/len(trainloader.dataset)))
                #history['expert_sample_loss'].append((torch.div(expert_sample_train_running_loss,
                #                                                torch.mean(gate_probabilities, dim=0).cpu()*len(trainloader.dataset))).cpu().numpy())
                history['expert_sample_loss'].append((torch.div(expert_sample_train_running_loss,
                                                                torch.sum(gate_probabilities, dim=0))))

                history['per_exp_class_samples'].append(per_exp_class_samples/num_batches)
                history['exp_samples'].append((torch.mean(gate_probabilities, dim = 0)*len(trainloader.dataset)))
                history['exp_samples_val'].append((torch.mean(test_gate_probabilities, dim = 0)*len(testloader.dataset)))
                history['mean_gate_log_probability'].append(torch.mean(torch.log(gate_probabilities), dim = 0))
                history['var_gate_log_probability'].append(torch.var(torch.log(gate_probabilities), dim = 0))
                history['mean_gate_probability'].append(torch.mean(gate_probabilities, dim = 0))
                history['var_gate_probability'].append(torch.var(gate_probabilities, dim = 0))
                history['kl_div_gate'].append(moe_models.kl_divergence(gate_probabilities, torch.mean(gate_probabilities, dim = 0).repeat(len(gate_probabilities),1)))
                history['cv'].append(moe_models.cv(gate_probabilities))
                
            if not T[epoch] == 1.0:
                gate_probabilities_high_T = torch.vstack(gate_probabilities_high_T)
                gate_probabilities_all_epochs_T.append(gate_probabilities_high_T)

                history['sample_entropy_T'].append(running_entropy_T)
                history['entropy_T'].append(moe_models.entropy(torch.mean(gate_probabilities_high_T, dim=0)))
                
                with torch.no_grad():
                    history['expert_sample_accuracy_T'].append((torch.div(expert_sample_train_running_accuracy_T,torch.mean(gate_probabilities_high_T, dim=0)*len(trainloader.dataset))))
                    history['expert_sample_loss_T'].append((torch.div(expert_sample_train_running_loss_T,
                                                                      torch.mean(gate_probabilities_high_T, dim=0)*len(trainloader.dataset))))
                    history['exp_samples_T'].append((torch.mean(gate_probabilities_high_T, dim = 0)*len(trainloader.dataset)))
                    history['mean_gate_log_probability_T'].append(torch.mean(torch.log(gate_probabilities_high_T), dim = 0))
                    history['var_gate_log_probability_T'].append(torch.var(torch.log(gate_probabilities_high_T), dim = 0))
                    history['mean_gate_probability_T'].append(torch.mean(gate_probabilities_high_T, dim = 0))
                    history['var_gate_probability_T'].append(torch.var(gate_probabilities_high_T, dim = 0))
                    history['kl_div_gate_T'].append(moe_models.kl_divergence(gate_probabilities_high_T, torch.mean(gate_probabilities_high_T, dim = 0).repeat(len(gate_probabilities_high_T),1)))
                    history['cv_T'].append(moe_models.cv(gate_probabilities_high_T))
            history['Temp'].append(T)
            print('epoch %d' % epoch,
                  'training loss %.2f' % running_loss,
                  ', training accuracy %.2f' % train_running_accuracy,
                  ', test accuracy %.2f' % test_running_accuracy)
            # if (epoch+1) % 20 == 0:
            #     if not optimizer.optimizer_moe is None:
            #         curr_lr /= 3
            #         self.update_lr(optimizer.optimizer_moe, curr_lr)
                
            if epoch > T_decay_start and T_decay > 0:
                print('t decay', type(T_decay),T_decay, epoch)
                T *= (1. / (1. + T_decay * epoch))
        history['gate_probabilities'] = gate_probabilities_all_epochs
        if not T[epoch] == 1.0:
           history['gate_probabilities_T'] = gate_probabilities_all_epochs_T 

        return history



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
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                if all_labels is None:
                   all_labels = labels
                else:
                   all_labels = torch.cat((all_labels,labels))
                
                outputs = self(inputs)
                gate_outputs = self.gate_outputs
                gate_probabilities.append(gate_outputs)
                expert_outputs = self.expert_outputs

                # zero the parameter gradients
                optimizer_gate.zero_grad(set_to_none=True)
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
               	    test_inputs, test_labels = test_inputs.to(self.device, non_blocking=True), test_labels.to(self.device, non_blocking=True)                
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
