import csv
import os
from helper import moe_models
from scipy.stats import entropy
from itertools import product
from helper.visualise_results import *

import torch.nn.functional as F

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda:0")

print('device', device)

# Compute accuracy of the model
def accuracy(out, yb, mean=True):
    preds = torch.argmax(out, dim=1).to(device, non_blocking=True)
    if mean:
        return (preds == yb).float().mean()
    else:
        return (preds == yb).float()
    
def collect_results(m, model_name, k=0, temps=[1.0], T_decay=[0.0], w_importance_range=[0.0], w_ortho_range=[0.0], 
                    w_sample_sim_same_range=[0.0], w_sample_sim_diff_range=[0.0],
                    total_experts=5, num_classes=10, num_epochs=20, 
                    testloader=None, model_path=None, results_path=None, filename='mnist_results.csv'):
    
    filename = os.path.join(results_path, filename)
    
    if os.path.exists(filename):
        p = 'a'
    else:
        p = 'w'
        
    header = ['filename', 'val error', 'test error','mutual information', 'sample entropy', 'experts usage', 'per_task_entropy']
    
    with open(filename, p) as f:
                
        writer = csv.writer(f)        
        
        if p == 'w':            
            writer.writerow(header)
        
        for T, decay, w_importance, w_sample_sim_same, w_sample_sim_diff in product(temps, T_decay, w_importance_range, w_sample_sim_same_range, w_sample_sim_diff_range):
            plot_file = generate_plot_file(m, temp=T, t_decay=decay, w_importance=w_importance, w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff, 
                                           specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)
            for _ in range(len(models)):
                mod = models.pop()
                data = [plot_file]    
                # model
                model = mod[model_name]['experts'][total_experts]['model']
                # history
                history = mod[model_name]['experts'][total_experts]['history']
                # validation error
                data.append(1-history['val_accuracy'][-1].item())
                gate_probabilities = torch.zeros(total_experts).to(device)
                running_test_accuracy = 0.0
                running_entropy = 0.0
                num_batches = 0
                
                # initialise the count matrix C for computing mutual information
                ey =  torch.zeros((num_classes, total_experts)).to(device)
                exp_class_prob = torch.zeros(total_experts, num_classes).to(device)
                all_labels = None
                for test_inputs, test_labels in testloader:
                    test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)                
                    if all_labels is None:
                        all_labels = test_labels
                    else:
                        all_labels = torch.cat((all_labels,test_labels))
                
                    outputs = model(test_inputs)
                    running_test_accuracy += accuracy(outputs, test_labels)
                    
                    selected_experts = torch.argmax(model.gate_outputs, dim=1)
                    y = test_labels
                    e = selected_experts
                    for j in range(y.shape[0]):
                        ey[int(torch.argmax(model.expert_outputs[j,e[j],:])), int(e[j])] += 1

                    if ('top_k' in model_name and k==1) or 'stochastic' in model_name:
                        for j in range(y.shape[0]):
                            exp_class_prob[selected_experts[j], y[j]] += 1
                    else:
                        for e in range(total_experts):
                            for index, l in enumerate(test_labels):
                                exp_class_prob[e,l] += model.gate_outputs[index,e]

                    running_entropy += moe_models.entropy(model.gate_outputs)
                    
                    gate_probabilities += torch.sum(model.gate_outputs, dim=0)
                    
                    num_batches+=1
                
                N = all_labels.shape[0]                
                counts = torch.unique(all_labels, return_counts=True)[1]
                
                mutual_EY,_,_,_ = moe_models.mutual_information(ey.detach())
    
                test_error = 1-(running_test_accuracy/num_batches)
                data.append(test_error.item())
                data.append(mutual_EY.item())                
                
                if ('top_k' in model_name and k==1) or 'stochastic' in model_name:
                    gate_probabilities = torch.sum(ey, dim=0)
                    # Since only one expert is selected for each sample during inference
                    data.append(0.0)
                else:
                    data.append(running_entropy.item()/num_batches)
                    
                gate_probabilities_mean = gate_probabilities/N
                data.append(entropy(gate_probabilities_mean).item())                
                
                norm_exp_class_prob = exp_class_prob/counts
                per_task_entropy = moe_models.entropy(norm_exp_class_prob.transpose(1,0))
                data.append(per_task_entropy.item())

                writer.writerow(data)


def collect_loss_gate_results(m, model_type='moe_expectation_model', temps=[1.0], w_importance_range=[0.0], 
                    w_sample_sim_same_range=[0.0], w_sample_sim_diff_range=[0.0],
                    total_experts=5, num_classes=10, num_epochs=20, 
                    testloader=None, model_path=None, results_path=None, filename ='mnist_results.csv' ):
    
    filename = os.path.join(results_path, filename)
    
    if os.path.exists(filename):
        p = 'a'
    else:
        p = 'w'
        
    header = ['filename', 'val error', 'test error','mutual information', 'sample entropy', 'experts usage', 'per_task_entropy']
    
    with open(filename, p) as f:
                
        writer = csv.writer(f)        
        
        if p == 'w':            
            writer.writerow(header)
        
        for w_importance, w_sample_sim_same, w_sample_sim_diff in product(w_importance_range, w_sample_sim_same_range, w_sample_sim_diff_range):
            plot_file = generate_plot_file(m, w_importance=w_importance, w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff, 
                                               specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)
            for _ in range(len(models)):
                mod = models.pop()
                data = [plot_file]
                # model
                model = mod[model_type]['experts'][total_experts]['model']
                # history
                history = mod[model_type]['experts'][total_experts]['history']
                # validation error
                data.append(1-history['val_accuracy'][-1].item())
                running_top1_accuracy = 0.0
                running_entropy = 0.0
                num_batches = 0
                ey =  torch.zeros((num_classes, total_experts)).to(device)
                exp_class_prob = torch.zeros(total_experts, num_classes).to(device)
                N = 0
                all_labels = None
                for test_inputs, test_labels in testloader:
                    
                    if all_labels is None:
                        all_labels = test_labels
                    else:
                        all_labels = torch.cat((all_labels,test_labels))
                        
                    test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)                
                    outputs = model(test_inputs)
                    expert_outputs = model.expert_outputs
                    gate_outputs = model.gate_outputs
                    running_top1_accuracy += accuracy(outputs, test_labels)
                    
                    selected_experts = torch.argmax(gate_outputs, dim=1)
                    y = test_labels
                    e = selected_experts
                    for j in range(y.shape[0]):
                        ey[int(torch.argmax(expert_outputs[j,e[j],:])), int(e[j])] += 1
                        
                    for j in range(y.shape[0]):
                            exp_class_prob[selected_experts[j], y[j]] += 1
                    
                    num_batches+=1
                    
                N = all_labels.shape[0]
                counts = torch.unique(all_labels, return_counts=True)[1].to(device)
 
                mutual_EY,_,_,_ = moe_models.mutual_information(ey.detach())
    
                top1_error = 1-(running_top1_accuracy/num_batches)
                
                data.append(top1_error.item())
                data.append(mutual_EY.item())
                
                # Since gate outputs are not probabilities
                data.append(0)  
                
                gate_probabilities = torch.sum(ey, dim=0)
                gate_probabilities_mean = gate_probabilities/N
                data.append(entropy(gate_probabilities_mean).item())
                
                norm_exp_class_prob = exp_class_prob/counts
                per_task_entropy = moe_models.entropy(norm_exp_class_prob.transpose(1,0))
                data.append(per_task_entropy.item())
                
                writer.writerow(data)
            
            