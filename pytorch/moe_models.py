import torch
import torch.nn as nn
import torch.nn.functional as F

# The moe architecture that outputs an expected output of the experts
# based on the gate probabilities

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class moe_expectation_model(nn.Module):
    
    def __init__(self, num_experts, experts, gate):
        super(moe_expectation_model,self).__init__()
        self.num_experts = num_experts
        self.experts = experts
        self.gate = gate
        
        
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
        output = torch.sum(p*x, 1)
        
        return output
    
    def train(self, trainloader, testloader, optimizer, loss_criterion, accuracy, epochs):
        expert_models = self.experts
        gate_model = self.gate
        running_loss = 0.0
        train_running_accuracy = 0.0
        test_running_accuracy = 0.0
        history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}
        for epoch in range(epochs):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data


                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = loss_criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
              
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc


            acc = 0.0
            for j, test_data in enumerate(testloader, 0):
                test_input, test_labels = test_data
                test_outputs = self(test_input)
                acc += accuracy(test_outputs, test_labels)
            test_running_accuracy = (acc/(j+1))

            running_loss = running_loss / (i+1)
            train_running_accuracy = train_running_accuracy / (i+1)
            history['loss'].append(running_loss)
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)
            print('epoch', epoch,
                  'training loss',
                  running_loss,
                  ', training accuracy',
                  train_running_accuracy,
                  ', test accuracy',
                  test_running_accuracy)
            running_loss = 0.0
            train_running_accuracy = 0.0
            test_running_accuracy = 0.0
        return history


# The moe architecture that outputs an expected output of the experts
# before softmax based on the gate probabilities
class moe_pre_softmax_expectation_model(nn.Module):
    
    def __init__(self, num_experts, experts, gate):
        super(moe_pre_softmax_expectation_model,self).__init__()
        self.num_experts = num_experts
        self.experts = experts
        self.gate = gate
        
        
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
    
    def train(self, trainloader, testloader, optimizer, loss_criterion, accuracy, epochs):
        expert_models = self.experts
        gate_model = self.gate
        running_loss = 0.0
        train_running_accuracy = 0.0
        test_running_accuracy = 0.0
        history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}        
        for epoch in range(epochs):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data


                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = self(inputs)
                loss = loss_criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
              
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc


            acc = 0.0
            for j, test_data in enumerate(testloader, 0):
                test_input, test_labels = test_data
                test_outputs = self(test_input)
                acc += accuracy(test_outputs, test_labels)
            test_running_accuracy = (acc/(j+1))

            running_loss = running_loss / (i+1)
            train_running_accuracy = train_running_accuracy / (i+1)
            history['loss'].append(running_loss)
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)

            print('epoch', epoch,
                  'training loss',
                  running_loss,
                  ', training accuracy',
                  train_running_accuracy,
                  ', test accuracy',
                  test_running_accuracy)
            running_loss = 0.0
            train_running_accuracy = 0.0
            test_running_accuracy = 0.0
        return history

def moe_stochastic_loss(expert_outputs, gate_output, target):
    expert_loss = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    for i in range(expert_outputs.shape[0]):
        cross_entropy_loss = criterion(expert_outputs[i], target)
        expert_loss.append(cross_entropy_loss)
    expert_loss = torch.stack(expert_loss)
    expert_loss.transpose_(0,1)
    expected_loss = torch.sum(gate_output * expert_loss, 1)
    loss = torch.mean(expected_loss)
    return loss   

# The moe architecture that outputs a stochastic selection of an expert
# based on the gate probabilities. 
class moe_stochastic_model(nn.Module):
    
    def __init__(self, num_experts, experts, gate):
        super(moe_stochastic_model,self).__init__()
        self.num_experts = num_experts
        self.experts = experts.to(device)
        self.gate = gate.to(device)
        
        
    def forward(self,input):
        batch_size = input.shape[0]
        if batch_size > 1:
            x = []
            for i, expert in enumerate(self.experts):
                x.append(expert(input))
            x = torch.stack(x)
            x.transpose_(0,1)
            
            p = self.gate(input)
            
            sample = torch.multinomial(p, 1)   
            
            output = torch.cat([x[i][sample[i]] for i in range(batch_size)])
        else:
            output = self.experts[0](input)

        return output
    
    def train(self, trainloader, testloader, optimizer, loss_criterion, accuracy, epochs):
        expert_models = self.experts
        gate_model = self.gate
        running_loss = 0.0
        train_running_accuracy = 0.0
        test_running_accuracy = 0.0
        history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}                
        for epoch in range(epochs):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labesl = inputs.to(device), labels.to(device)
                #print(inputs)


                # zero the parameter gradients
                optimizer.zero_grad()

                x = []
                for j, expert in enumerate(expert_models):
                    outputs = expert(inputs)
                    #print(outputs)
                    x.append(outputs)
                #print(x, len(x))
                x = torch.stack(x)
                p = gate_model(inputs)

                loss = loss_criterion(x, p , labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                outputs = self(inputs)
                acc = accuracy(outputs, labels)
                train_running_accuracy += acc


            acc = 0.0
            for j, test_data in enumerate(testloader, 0):
                test_input, test_labels = test_data
                test_input, test_labels = test_inputs.to(device), test_labelss.to(device)
                test_outputs = self(test_input)
                acc += accuracy(test_outputs, test_labels)
            test_running_accuracy = (acc/(j+1))

            running_loss = running_loss / (i+1)
            train_running_accuracy = train_running_accuracy / (i+1)
            history['loss'].append(running_loss)
            history['accuracy'].append(train_running_accuracy)
            history['val_accuracy'].append(test_running_accuracy)

            print('epoch', epoch,
                  'training loss',
                  running_loss,
                  ', training accuracy',
                  train_running_accuracy,
                  ', test accuracy',
                  test_running_accuracy)
            running_loss = 0.0
            train_running_accuracy = 0.0
            test_running_accuracy = 0.0
        return history

