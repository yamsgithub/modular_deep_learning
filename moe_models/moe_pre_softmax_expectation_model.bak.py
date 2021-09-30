import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import moe_models

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

# The moe architecture that outputs an expected output of the experts
# before softmax based on the gate probabilities
class moe_pre_softmax_expectation_model(nn.Module):
    
    def __init__(self, num_experts, num_classes, augment, attention_flag, experts, gate):
        super(moe_pre_softmax_expectation_model,self).__init__()
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
        
    def forward(self,inputs, T=1.0):   
        y = []
        for i, expert in enumerate(self.experts):
            y.append(expert(inputs))
        y = torch.stack(y)
        y.transpose_(0,1)

        self.expert_outputs = F.softmax(y, dim=1)
        
        if self.augment:
            output_aug = torch.flatten(y, start_dim=1)
            input_aug = torch.cat((inputs, output_aug), dim=1)
            p = self.gate(input_aug)
        elif self.attention:
            context = self.attn(inputs, y)
            input_aug = torch.cat((inputs, context.squeeze(1)), dim=1)
            p = self.gate(input_aug)
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
        output = F.softmax(torch.sum(p*y, 1), dim=1)
        
        return output
    
    def train(self, trainloader, testloader,
              loss_criterion, optimizer_moe, optimizer_gate=None, optimizer_experts=None, 
              w_importance = 0.0, w_ortho = 0.0, w_ideal_gate = 0.0,
              T=1.0, T_decay=0, T_decay_start=0,
              accuracy=None, epochs=10):

        history = {'loss':[], 'loss_importance':[],'accuracy':[], 'val_accuracy':[],
                   'entropy':[], 'EY':[],'mutual_EY':[], 'H_EY':[],'H_Y':[], 'H_E':[],
                   'exp_samples':[],'exp_samples_high_T':[], 'exp_samples_val':[],
                   'gate_probability': [], 'gate_probability_high_T': [],
                   'Temp':[],
                   'w_importance':w_importance, 'w_ortho':w_ortho, 'w_ideal_gate':w_ideal_gate
                   }

        for epoch in range(epochs):  # loop over the dataset multiple times
            num_batches = 0
            running_loss = 0.0
            running_loss_importance = 0.0
            train_running_accuracy = 0.0 
            test_running_accuracy = 0.0
            running_entropy = 0.0
            
            ey =  np.zeros((self.num_classes, self.num_experts))
            per_exp_batch = torch.zeros(self.num_experts)
            per_exp_batch_high_T = torch.zeros(self.num_experts)
            per_exp_batch_val = torch.zeros(self.num_experts)

            running_gate_probabilities = torch.zeros(self.num_experts)
            running_gate_probabilities_high_T = torch.zeros(self.num_experts)

            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer_moe.zero_grad()
                
                outputs = self(inputs)
                gate_outputs = self.gate_outputs

                per_exp_batch = torch.add(torch.sum(torch.where(gate_outputs >= 0.001, torch.ones(gate_outputs.shape),torch.zeros(gate_outputs.shape)),0 ),
                                          per_exp_batch)

                loss = loss_criterion(outputs, labels)

                l_imp = 0.0
                if w_importance > 0.0:
                    l_imp = moe_models.loss_importance(gate_outputs, w_importance)
                    loss += l_imp
                    running_loss_importance += l_imp
                                    
                    
                loss.backward()

                optimizer_moe.step()

                running_loss += loss.item()

                outputs = self(inputs)

                running_gate_probabilities = torch.add(torch.sum(self.gate_outputs,0),
                                                       running_gate_probabilities)
            
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc

                #computing entropy
                running_entropy += moe_models.entropy(self.gate_outputs)

                # update the Y vs E table to compute joint distribution of Y and E
                selected_experts = torch.zeros(len(labels))
                if self.num_experts > 1:
                    selected_experts = torch.argmax(self.gate_outputs, dim=1)

                y = labels.numpy()
                e = selected_experts.numpy()
                for j in range(labels.shape[0]):
                    ey[int(y[j]), int(e[j])] += 1
    
                num_batches+=1

            with torch.no_grad():
                acc = 0.0
                j = 0
                for test_inputs, test_labels in testloader:
                    test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)                
                    test_outputs = self(test_inputs)
                    test_gate_outputs = self.gate_outputs
                    per_exp_batch_val = torch.sum(torch.where(test_gate_outputs >= 0.01, torch.ones(test_gate_outputs.shape),torch.zeros(test_gate_outputs.shape)),0 ).reshape(self.num_experts)
                    acc += accuracy(test_outputs, test_labels)
                    j += 1
                test_running_accuracy = (acc.cpu().numpy()/j)

            running_loss = running_loss / num_batches
            running_loss_importance = running_loss_importance / num_batches
            train_running_accuracy = train_running_accuracy.cpu().numpy() / num_batches
            with torch.no_grad():
                running_entropy = running_entropy.cpu().numpy() / num_batches

            mutual_EY, H_EY, H_E, H_Y = moe_models.mutual_information(ey)
            
            history['loss'].append(running_loss)
            history['loss_importance'].append(running_loss_importance)
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)
            history['entropy'].append(running_entropy)
            history['EY'].append(ey)
            history['mutual_EY'].append(mutual_EY)
            history['H_EY'].append(H_EY)
            history['H_E'].append(H_E)
            history['H_Y'].append(H_Y)
            history['exp_samples'].append(per_exp_batch.reshape(self.num_experts).cpu().numpy())
            history['exp_samples_val'].append(per_exp_batch_val.reshape(self.num_experts).cpu().numpy())
            with torch.no_grad():
                history['gate_probability'].append(running_gate_probabilities.cpu().numpy()/len(trainloader.dataset))

            history['Temp'].append(T)

            print('epoch %d' % epoch,
                  'training loss %.2f' % running_loss,
                  ', training accuracy %.2f' % train_running_accuracy,
                  ', test accuracy %.2f' % test_running_accuracy)
            
            running_loss = 0.0
            train_running_accuracy = 0.0
            test_running_accuracy = 0.0
            running_entropy = 0.0
            
        return history
