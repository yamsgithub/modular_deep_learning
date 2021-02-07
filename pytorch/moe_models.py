import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

def cross_entropy_loss(p, targets, reduction='mean'):
    eps=1e-7
    logp = torch.log(p+eps)
    if reduction == 'none':
        criterion = nn.NLLLoss(reduction=reduction)
    else:
        criterion = nn.NLLLoss()
    crossentropy_loss = criterion(logp, targets)
    return crossentropy_loss

def entropy(p):
    logp = torch.log2(p)
    with torch.no_grad():
        logp = np.nan_to_num(logp.cpu().numpy(), neginf=0)
        entropy_val = (-1*torch.sum(p.to(torch.device('cpu'))*logp,dim=1)).mean()
    return entropy_val

def loss_importance(p, w_importance):
    loss_importance = 0.0
    if p.shape[1] > 1:
        importance = torch.sum(p,dim=0)
        loss_importance = w_importance * torch.square(torch.std(importance)/torch.mean(importance))
    return loss_importance

# The moe architecture that outputs an expected output of the experts
# based on the gate probabilities
class moe_expectation_model(nn.Module):
    
    def __init__(self, num_experts, experts, gate):
        super(moe_expectation_model,self).__init__()
        self.num_experts = num_experts
        self.experts = experts.to(device)
        self.gate = gate.to(device)
        
        
    def forward(self,input):   
        x = []
        for i, expert in enumerate(self.experts):
            x.append(expert(input))
        x = torch.stack(x)
        x.transpose_(0,1)
        
        p = self.gate(input)
        
        # reshape gate output so probabilities correspond 
        # to each expert
        p = p.reshape(p.shape[0],p.shape[1], 1)
        
        # repeat probabilities number of classes times so
        # dimensions correspond
        p = p.repeat(1,1,x.shape[2])
        
        # expected sum of expert outputs
        output = torch.sum(p*x, 1)

        return output
    
    def train(self, trainloader, testloader, optimizer, loss_criterion, w_importance = 1.0, accuracy=None, epochs=10):
        expert_models = self.experts
        gate_model = self.gate
        running_loss = 0.0
        running_loss_importance = 0.0
        train_running_accuracy = 0.0 
        test_running_accuracy = 0.0
        running_entropy = 0.0
        history = {'loss':[], 'loss_importance':[],'accuracy':[], 'val_accuracy':[], 'entropy':[]}        
        for epoch in range(epochs):  # loop over the dataset multiple times
            i = 0
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(inputs)
                gate_outputs = self.gate(inputs)

                loss = loss_criterion(outputs, labels)
                l_imp = 0.0
                if w_importance > 0.0:
                    l_imp = loss_importance(gate_outputs, w_importance)
                    loss += l_imp
                    
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                running_loss_importance += l_imp
                
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc
                #computing entropy
                gate_outputs = self.gate(inputs)
                running_entropy += entropy(gate_outputs)

                i+=1
            
            with torch.no_grad():
                acc = 0.0
                j = 0
                for test_inputs, test_labels in testloader:
               	    test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)                
                    test_outputs = self(test_inputs)
                    acc += accuracy(test_outputs, test_labels)
                    j += 1
                test_running_accuracy = (acc.cpu().numpy()/j)

            running_loss = running_loss / i
            running_loss_importance = running_loss_importance / i
            train_running_accuracy = train_running_accuracy.cpu().numpy() / i
            with torch.no_grad():
                running_entropy = running_entropy.cpu().numpy() / i
            history['loss'].append(running_loss)
            history['loss_importance'].append(running_loss_importance)            
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)
            history['entropy'].append(running_entropy)

            print('epoch %d' % epoch,
                  'training loss %.2f' % running_loss,
                  ', training accuracy %.2f' % train_running_accuracy,
                  ', test accuracy %.2f' % test_running_accuracy)
            
            running_loss = 0.0
            train_running_accuracy = 0.0
            test_running_accuracy = 0.0
            running_entropy = 0.0
        return history


# The moe architecture that outputs an expected output of the experts
# before softmax based on the gate probabilities
class moe_pre_softmax_expectation_model(nn.Module):
    
    def __init__(self, num_experts, experts, gate):
        super(moe_pre_softmax_expectation_model,self).__init__()
        self.num_experts = num_experts
        self.experts = experts.to(device)
        self.gate = gate.to(device)
        
        
    def forward(self,input):   
        x = []
        for i, expert in enumerate(self.experts):
            x.append(expert(input))
        x = torch.stack(x)
        x.transpose_(0,1)
        
        p = self.gate(input)
        
        # reshape gate output so probabilities correspond 
        # to each expert
        p = p.reshape(p.shape[0],p.shape[1], 1)
        
        # repear probabilities number of classes times so
        # dimensions correspond
        p = p.repeat(1,1,x.shape[2])
        
        # expected sum of expert outputs
        output = F.softmax(torch.sum(p*x, 1), dim=1)
        
        return output
    
    def train(self, trainloader, testloader, optimizer, loss_criterion, w_importance=1.0, accuracy=None, epochs=10):
        expert_models = self.experts
        gate_model = self.gate
        running_loss = 0.0
        running_loss_importance = 0.0
        train_running_accuracy = 0.0
        test_running_accuracy = 0.0
        running_entropy = 0.0
        history = {'loss':[], 'loss_importance':[],'accuracy':[], 'val_accuracy':[], 'entropy':[]}        
        for epoch in range(epochs):  # loop over the dataset multiple times
            i = 0
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(inputs)
                gate_outputs = self.gate(inputs)

                loss = loss_criterion(outputs, labels)
                l_imp = 0.0
                if w_importance > 0.0:
                    l_imp = loss_importance(gate_outputs, w_importance)
                    loss += l_imp
                    
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                running_loss_importance += l_imp
              
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc
                
                #computing entropy
                gate_outputs = self.gate(inputs)
                running_entropy += entropy(gate_outputs)
                
                i+=1



            with torch.no_grad():
                acc = 0.0
                j = 0
                for test_inputs, test_labels in testloader:
                    test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)                
                    test_outputs = self(test_inputs)
                    acc += accuracy(test_outputs, test_labels)
                    j += 1
                test_running_accuracy = (acc.cpu().numpy()/j)

            running_loss = running_loss / i
            running_loss_importance = running_loss_importance / i
            train_running_accuracy = train_running_accuracy.cpu().numpy() / i
            with torch.no_grad():
                running_entropy = running_entropy.cpu().numpy() / i

            history['loss'].append(running_loss)
            history['loss_importance'].append(running_loss_importance)
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)
            history['entropy'].append(running_entropy)
            
            print('epoch %d' % epoch,
                  'training loss %.2f' % running_loss,
                  ', training accuracy %.2f' % train_running_accuracy,
                  ', test accuracy %.2f' % test_running_accuracy)
            
            running_loss = 0.0
            train_running_accuracy = 0.0
            test_running_accuracy = 0.0
            running_entropy = 0.0
        return history

def moe_stochastic_loss(expert_outputs, gate_output, target):
    expert_loss = []
    for i in range(expert_outputs.shape[0]):
        crossentropy_loss = cross_entropy_loss(expert_outputs[i], target, reduction='none')
        expert_loss.append(torch.exp(-0.5*crossentropy_loss))
    expert_loss = torch.stack(expert_loss)
    expert_loss.transpose_(0,1)
    expected_loss = -1*torch.log(torch.sum(gate_output * expert_loss, 1))
    loss = torch.mean(expected_loss)
    return loss.to(device)   

def moe_stochastic_loss_1(expert_outputs, gate_output, target):
    expert_loss = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    for i in range(expert_outputs.shape[0]):
        cross_entropy_loss = criterion(expert_outputs[i], target)
        expert_loss.append(cross_entropy_loss)
    expert_loss = torch.stack(expert_loss)
    expert_loss.transpose_(0,1)
    expected_loss = torch.sum(gate_output * expert_loss, 1)
    loss = torch.mean(expected_loss)
    return loss.to(device)   

# The moe architecture that outputs a stochastic selection of an expert
# based on the gate probabilities. 
class moe_stochastic_model(nn.Module):
    
    def __init__(self, num_experts, experts, gate):
        super(moe_stochastic_model,self).__init__()
        self.num_experts = num_experts
        self.experts = experts.to(device)
        self.gate = gate.to(device)
        
        
    def forward(self,inputs):
        batch_size = inputs.shape[0]
        if batch_size > 1:
            x = []
            for i, expert in enumerate(self.experts):
                x.append(expert(inputs))
            x = torch.stack(x)
            x.transpose_(0,1)

            try:
                p = self.gate(inputs)
                sample = torch.multinomial(p, 1)
            except:
                print('inputs', inputs.shape, torch.sum(inputs))
                print('p',p[0:5])
                raise
            
            output = torch.cat([x[i][sample[i]] for i in range(batch_size)])
        else:
            output = self.experts[0](inputs)

        return output
    
    def train(self, trainloader, testloader, optimizer, loss_criterion, w_importance, accuracy=None, epochs=10):
        expert_models = self.experts
        gate_model = self.gate
        running_loss = 0.0
        running_loss_importance = 0.0
        train_running_accuracy = 0.0
        test_running_accuracy = 0.0
        running_entropy = 0.0
        history = {'loss':[], 'loss_importance':[],'accuracy':[], 'val_accuracy':[], 'entropy':[]}        
        for epoch in range(epochs):  # loop over the dataset multiple times
            i = 0
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                x = []
                for j, expert in enumerate(expert_models):
                    outputs = expert(inputs)
                    x.append(outputs)
                x = torch.stack(x)
                p = gate_model(inputs)

                #computing entropy
                running_entropy += entropy(p)

                loss = loss = loss_criterion(x.to(device, non_blocking=True), p.to(device, non_blocking=True) , labels)
                l_imp = 0.0
                if w_importance > 0.0:
                    l_imp = loss_importance(p, w_importance)
                    loss += l_imp
                    
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                running_loss_importance += l_imp
                
                try:
                    outputs = self(inputs)
                except:
                    for name, param in self.gate.named_parameters():
                        if param.requires_grad:
                            print(name, param.data)
                    
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc

                i+=1
                
            with torch.no_grad():
                acc = 0.0
                j = 0
                for test_inputs, test_labels in testloader:
               	    test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)
                    test_outputs = self(test_inputs)
                    acc += accuracy(test_outputs, test_labels)
                    j+=1
                test_running_accuracy = (acc.cpu().numpy()/j)

            running_loss = running_loss / i
            running_loss_importance = running_loss_importance / i
            train_running_accuracy = train_running_accuracy.cpu().numpy() / i

            with torch.no_grad():
                running_entropy = running_entropy.cpu().numpy() / i
            
            history['loss'].append(running_loss)
            history['loss_importance'].append(running_loss_importance)            
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)
            history['entropy'].append(running_entropy)
            
            print('epoch %d' % epoch,
                  'training loss %.2f' % running_loss,
                  ', training accuracy %.2f' % train_running_accuracy,
                  ', test accuracy %.2f' % test_running_accuracy)
            running_loss = 0.0
            train_running_accuracy = 0.0
            test_running_accuracy = 0.0
            running_entropy = 0.0
        return history


    

