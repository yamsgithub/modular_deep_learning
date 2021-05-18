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

def mutual_information(count_mat):
    #print('\nEY', count_mat)
    (r,c) = count_mat.shape
    joint_EY = np.zeros((r,c))
    N = np.sum(count_mat)
    for i in range(r):
        for j in range(c):
            joint_EY[i,j] = count_mat[i,j]/N
    marginal_Y = np.sum(joint_EY, axis=1)
    marginal_E = np.sum(joint_EY, axis=0)
    
    log2_marginal_Y = np.ma.log2(marginal_Y).filled(fill_value=0.0)
    H_Y = 0.0
    for i in range(r):
        H_Y += marginal_Y[i]*log2_marginal_Y[i]
    H_Y = -1*H_Y    

    log2_marginal_E = np.ma.log2(marginal_E).filled(fill_value=0.0)
    H_E = 0.0
    for i in range(c):
        H_E += marginal_E[i]*log2_marginal_E[i]
    H_E = -1*H_E   

    H_EY = 0.0
    log2_joint_EY = np.ma.log2(joint_EY).filled(fill_value=0.0)
    for j in range(c):
        for i in range(r):
            H_EY += joint_EY[i,j]*log2_joint_EY[i,j]
    H_EY = -1 * H_EY    
    mutual_EY = H_E+H_Y-H_EY

    return mutual_EY, H_EY, H_E, H_Y

class attention(nn.Module):

    def __init__(self, num_experts, num_classes):
        super(attention,self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes

        self.Wx = nn.Linear(2, num_experts, bias=False)
        self.We = nn.Linear(num_classes, num_experts, bias=False)
        self.weight = nn.Parameter(torch.FloatTensor(1, num_experts))

    def forward(self, inputs, expert_outputs):
        # Calculating Alignment Scores
        batch_size = inputs.shape[0]
        x = torch.tanh(self.Wx(inputs).unsqueeze(1).repeat(1,self.num_experts,1)+self.We(expert_outputs))
        alignment_scores = self.weight.repeat(batch_size,1,1).bmm(x)
        
        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores, dim=1)
        
        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.bmm(attn_weights,
                                  expert_outputs)
        return context_vector
    
# The moe architecture that outputs an expected output of the experts
# based on the gate probabilities
class moe_expectation_model(nn.Module):
    
    def __init__(self, num_experts, num_classes, augment, attention_flag, experts, gate):
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
            self.attn = attention(num_experts, num_classes)
        
    def forward(self,inputs, T=1.0):
        y = []
        for i, expert in enumerate(self.experts):
            y.append(expert(inputs))
        y = torch.stack(y)
        y.transpose_(0,1)
        self.expert_outputs = y

        if self.augment:
            output_aug = torch.flatten(y, start_dim=1)
            #output_aug = output_aug.detach()
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
    
    def train(self, trainloader, testloader, loss_criterion, optimizer_moe, optimizer_gate=None, optimizer_experts=None, 
              w_importance = 0.0, w_ortho = 0.0, w_ideal_gate = 0.0, T=1.0, accuracy=None, epochs=10):
        expert_models = self.experts
        gate_model = self.gate
        running_loss = 0.0
        running_loss_importance = 0.0
        train_running_accuracy = 0.0 
        test_running_accuracy = 0.0
        running_entropy = 0.0
        history = {'loss':[], 'loss_importance':[],'accuracy':[], 'val_accuracy':[], 'entropy':[], 'EY':[],'mutual_EY':[], 'H_EY':[],'H_Y':[], 'H_E':[],'exp_batch':[]}        
        for epoch in range(epochs):  # loop over the dataset multiple times
            num_batches = 0
            ey =  np.zeros((self.num_classes, self.num_experts))
            per_exp_batch = [0] * self.num_experts

            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                outputs = self(inputs)
                gate_outputs = self.gate_outputs
                expert_outputs = self.expert_outputs

                # b_indices = torch.argmax(gate_outputs, dim=1)

                # for b_index in b_indices:
                #     per_exp_batch[b_index] += 1

                
                if T > 1.0:
                    outputs_with_T = self(inputs, T)
                    gate_outputs_T = self.gate_outputs
                    for b_index in range(0, gate_outputs_T.shape[0]):
                        for e_index in range(0, gate_outputs_T.shape[1]):
                            if gate_outputs_T[b_index, e_index] >= 0.01:
                                per_exp_batch[e_index] += 1

                    optimizer_experts.zero_grad()
                    loss_experts = loss_criterion(outputs_with_T, labels)
                    loss_experts.backward()

                    optimizer_gate.zero_grad()
                    loss = loss_criterion(outputs, labels)
                    for i, expert in enumerate(self.experts):
                        for param in expert.parameters():
                            param.requires_grad = False
                    loss.backward()
                    for i, expert in enumerate(self.experts):
                        for param in expert.parameters():
                            param.requires_grad = True
                    optimizer_gate.step()
                    optimizer_experts.step()
                    
                else:
                    # zero the parameter gradients
                    for b_index in range(0, gate_outputs.shape[0]):
                        for e_index in range(0, gate_outputs.shape[1]):
                            if gate_outputs[b_index, e_index] >= 0.01:
                                per_exp_batch[e_index] += 1
                                
                    optimizer_moe.zero_grad()
                    loss = loss_criterion(outputs, labels)
                    
                    if w_ideal_gate > 0.0 and self.num_experts > 1:
                        l_experts = []
                        for i in range(self.num_experts):
                            l_experts.append(loss_criterion(self.expert_outputs[:,i,:],labels, reduction='none'))
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
                        l_imp = loss_importance(gate_outputs, w_importance)
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
                #print('LOSS',loss)

                outputs = self(inputs)
                
                acc = accuracy(outputs, labels)
                #print('ACCURACY', acc)
                train_running_accuracy += acc

                #computing entropy
                running_entropy += entropy(self.gate_outputs)
                
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
                    acc += accuracy(test_outputs, test_labels)
                    j += 1
                test_running_accuracy = (acc.cpu().numpy()/j)

            running_loss = running_loss / num_batches
            running_loss_importance = running_loss_importance / num_batches
            train_running_accuracy = train_running_accuracy.cpu().numpy() / num_batches
            with torch.no_grad():
                running_entropy = running_entropy.cpu().numpy() / num_batches

            mutual_EY, H_EY, H_E, H_Y = mutual_information(ey)
            
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
            history['exp_batch'].append(per_exp_batch)
            
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
            self.attn = attention(num_experts, num_classes)
        
    def forward(self,inputs):   
        y = []
        for i, expert in enumerate(self.experts):
            y.append(expert(inputs))
        y = torch.stack(y)
        y.transpose_(0,1)

        self.expert_outputs = y
        
        if self.augment:
            output_aug = torch.flatten(y, start_dim=1)
            #output_aug = output_aug.detach()            
            input_aug = torch.cat((inputs, output_aug), dim=1)
            p = self.gate(input_aug)
        elif self.attention:
            context = self.attn(inputs, y)
            input_aug = torch.cat((inputs, context.squeeze(1)), dim=1)
            p = self.gate(input_aug)
            
        else:
            p = self.gate(inputs)

        self.gate_outputs = p
        
        # reshape gate output so probabilities correspond 
        # to each expert
        p = p.reshape(p.shape[0],p.shape[1], 1)
        
        # repear probabilities number of classes times so
        # dimensions correspond
        p = p.repeat(1,1,y.shape[2])
        
        # expected sum of expert outputs
        output = F.softmax(torch.sum(p*y, 1), dim=1)
        
        return output
    
    def train(self, trainloader, testloader, optimizer, loss_criterion, w_importance=0.0, w_ortho = 0.0, w_ideal_gate = 0.0, accuracy=None, epochs=10):
        expert_models = self.experts
        gate_model = self.gate
        running_loss = 0.0
        running_loss_importance = 0.0
        train_running_accuracy = 0.0
        test_running_accuracy = 0.0
        running_entropy = 0.0
        history = {'loss':[], 'loss_importance':[],'accuracy':[], 'val_accuracy':[], 'entropy':[], 'EY':[],'mutual_EY':[], 'H_EY':[],'H_Y':[], 'H_E':[]}        
        for epoch in range(epochs):  # loop over the dataset multiple times
            i = 0
            ey =  np.zeros((self.num_classes, self.num_experts))
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                self(inputs)

                p = self.gate_outputs
                y = self.expert_outputs
                
                # reshape gate output so probabilities correspond 
                # to each expert
                p = p.reshape(p.shape[0],p.shape[1], 1)
                
                # repear probabilities number of classes times so
                # dimensions correspond
                p = p.repeat(1,1,y.shape[2])

                T = 1
                outputs = F.softmax(torch.sum(p*y, 1)/T, dim=1)

                #outputs = outputs-torch.mean(outputs,1).reshape(outputs.shape[0],1).repeat(1, self.num_classes)

                #print('labels', labels)
                #print('outputs', outputs)
                loss = loss_criterion(outputs, labels)

                l_imp = 0.0
                if w_importance > 0.0:
                    l_imp = loss_importance(self.gate_outputs, w_importance)
                    loss += l_imp
                    
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                running_loss_importance += l_imp

                outputs = self(inputs)

                acc = accuracy(outputs, labels)
                train_running_accuracy += acc

                #computing entropy
                running_entropy += entropy(self.gate_outputs)

                # update the Y vs E table to compute joint distribution of Y and E
                selected_experts = torch.zeros(len(labels))
                if self.num_experts > 1:
                    selected_experts = torch.argmax(self.gate_outputs, dim=1)

                y = labels.numpy()
                e = selected_experts.numpy()
                for j in range(labels.shape[0]):
                    ey[int(y[j]), int(e[j])] += 1
    
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

            mutual_EY, H_EY, H_E, H_Y = mutual_information(ey)
            
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
    
    def __init__(self, num_experts, num_classes, augment, attention_flag, experts, gate):
        super(moe_stochastic_model,self).__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.augment = augment        
        self.experts = experts.to(device)
        self.gate = gate.to(device)
        self.expert_outputs = None
        self.gate_outputs = None
        self.attn = attention(num_experts, num_classes)
        
    def forward(self,inputs):
        batch_size = inputs.shape[0]
        if batch_size > 1:
            y = []
            for i, expert in enumerate(self.experts):
                y.append(expert(inputs))
            y = torch.stack(y)
            y.transpose_(0,1)

            self.expert_outputs = y
            
            try:
                if self.augment:
                    context = self.attn(inputs, y)
                    input_aug = torch.cat((inputs, context.squeeze(1)), dim=1)
                    p = self.gate(input_aug)
                else:
                    p = self.gate(inputs)

                self.gate_outputs = p
                
                sample = torch.multinomial(p, 1)
            except:
                raise
            
            output = torch.cat([y[i][sample[i]] for i in range(batch_size)])
        else:
            output = self.expert_outputs[:,0,:]

        return output
    
    def train(self, trainloader, testloader, optimizer, loss_criterion, w_importance, accuracy=None, epochs=10):
        expert_models = self.experts
        gate_model = self.gate
        running_loss = 0.0
        running_loss_importance = 0.0
        train_running_accuracy = 0.0
        test_running_accuracy = 0.0
        running_entropy = 0.0
        history = {'loss':[], 'loss_importance':[],'accuracy':[], 'val_accuracy':[], 'entropy':[], 'EY':[],'mutual_EY':[], 'H_EY':[],'H_Y':[], 'H_E':[]}        
        for epoch in range(epochs):  # loop over the dataset multiple times
            i = 0
            ey =  np.zeros((self.num_classes, self.num_experts))            
            for inputs, labels in trainloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(inputs)
                
                y = self.expert_outputs.transpose_(1,0)
                p = self.gate_outputs

                #print('y', y.shape)
                #print('p', p.shape)

                loss = loss = loss_criterion(y.to(device, non_blocking=True), p.to(device, non_blocking=True) , labels)
                l_imp = 0.0
                if w_importance > 0.0:
                    l_imp = loss_importance(p, w_importance)
                    loss += l_imp
                    
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                running_loss_importance += l_imp
                
                outputs = self(inputs)

                acc = accuracy(outputs, labels)
                train_running_accuracy += acc

                #computing entropy
                running_entropy += entropy(self.gate_outputs)

                # update the Y vs E table to compute joint distribution of Y and E
                selected_experts = torch.zeros(len(labels))
                if self.num_experts > 1:
                    selected_experts = torch.argmax(self.gate_outputs, dim=1)

                y = labels.numpy()
                e = selected_experts.numpy()
                for j in range(labels.shape[0]):
                    ey[int(y[j]), int(e[j])] += 1
                
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

            mutual_EY, H_EY, H_E, H_Y = mutual_information(ey)
                
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

            
            print('epoch %d' % epoch,
                  'training loss %.2f' % running_loss,
                  ', training accuracy %.2f' % train_running_accuracy,
                  ', test accuracy %.2f' % test_running_accuracy)
            running_loss = 0.0
            train_running_accuracy = 0.0
            test_running_accuracy = 0.0
            running_entropy = 0.0
        return history


    

