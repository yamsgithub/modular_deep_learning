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
    
# Compute accuracy of the model
def accuracy_top5(out, yb):
    preds = torch.argsort(out, dim=1,descending=True).to(device, non_blocking=True)
    N = preds.shape[0]
    correct = 0
    for i in range(N):
        if yb[i] in preds[i,0:5]:
            correct += 1
    return correct/N
    
    
def generate_results(mod, model_name, k, plot_file, testloader, total_experts, num_classes, top_5=False, writer=None):
    data = [plot_file]    
    # model
    model = mod[model_name]['experts'][total_experts]['model']
    model.device = device
    # history
    history = mod[model_name]['experts'][total_experts]['history']

    # del mod

    # validation error
    data.append(1-history['val_accuracy'][-1].item())

    # del history

    gate_probabilities = torch.zeros(total_experts).to(device)
    running_test_accuracy = 0.0
    running_top5_accuracy = 0.0
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
        if top_5:
            running_top5_accuracy += accuracy_top5(outputs, test_labels)

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

    del model

    N = all_labels.shape[0]                
    counts = torch.unique(all_labels, return_counts=True)[1]

    mutual_EY,_,_,_ = moe_models.mutual_information(ey.detach())

    test_error = 1-(running_test_accuracy/num_batches)
    if top_5:
        top5_error = 1-(running_top5_accuracy/num_batches)
    
    data.append(test_error.item())
    if top_5:
        data.append(top5_error)
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

    
def collect_results(m, model_name, k=0, temps=[1.0], T_decay=[0.0], w_importance_range=[0.0], w_ortho_range=[0.0], 
                    w_sample_sim_same_range=[0.0], w_sample_sim_diff_range=[0.0],
                    total_experts=5, num_classes=10, num_epochs=20, top_5=False,
                    testloader=None, model_path=None, results_path=None, filename='mnist_results.csv'):
    
    filename = os.path.join(results_path, filename)
    
    if os.path.exists(filename):
        p = 'a'
    else:
        p = 'w'
        
    if top_5:
        header = ['filename', 'val error', 'top-1 error', 'top-5 error', 'mutual information',  'sample entropy', 'experts usage', 'per_task_entropy']
    else:
        header = ['filename', 'val error', 'test error','mutual information', 'sample entropy', 'experts usage', 'per_task_entropy']
    
    with open(filename, p) as f:
                
        writer = csv.writer(f)        
        
        if p == 'w':            
            writer.writerow(header)
        
        for T, decay, w_importance, w_sample_sim_same, w_sample_sim_diff in product(temps, T_decay, w_importance_range, w_sample_sim_same_range, w_sample_sim_diff_range):
            plot_file = generate_plot_file(m, temp=T, t_decay=decay, w_importance=w_importance, w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff, 
                                           specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')
            print(plot_file)
            models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)
            print(len(models))
            for _ in range(len(models)):
                mod = models.pop()
                generate_results(mod, model_name, k, plot_file, testloader, total_experts, num_classes, top_5, writer)
                del mod


def collect_single_result(m, num_classes=10, num_epochs=20, testloader=None, top_5=False,
                          model_path=None, results_path=None, filename='mnist_results.csv'):
    import csv

    plot_file = generate_plot_file(m, specific=str(num_classes)+'_models.pt')

    # loading model from pre_trained_model_path. Change this to model_path to use models you train
    models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)
    
    filename = os.path.join(results_path, filename)
    
    if os.path.exists(filename):
        p = 'a'
    else:
        p = 'w'

    if top_5:
        header = ['filename', 'val error', 'top-1 error', 'top-5 error', 'mutual information', 'sample entropy', 'experts usage', 'per_task_entropy']
    else:
        header = ['filename', 'val error', 'test error','mutual information', 'sample entropy', 'experts usage', 'per_task_entropy']
    
    with open(filename, p) as f:
        writer = csv.writer(f)        

        if p == 'w':            
            writer.writerow(header)
        for i, model in enumerate(models['models']):
            data = [plot_file] 
            running_test_accuracy = 0.0
            running_top5_accuracy = 0.0
            num_batches = 0
            val_error = 1-models['history'][i]['val_accuracy'][-1]
            data.append(val_error)
            for test_inputs, test_labels in testloader:
                test_inputs, test_labels = test_inputs.to(device, non_blocking=True), test_labels.to(device, non_blocking=True)                
                outputs = model(test_inputs)
                running_test_accuracy += accuracy(outputs, test_labels)
                if top_5:
                    running_top5_accuracy += accuracy_top5(outputs, test_labels)
                num_batches += 1
            test_error = 1-(running_test_accuracy/num_batches)
            
            data.append(test_error.item())
            if top_5:
                top5_error = 1-(running_top5_accuracy/num_batches)
                data.append(top5_error)
            for i in range(4):
                data.append('')
            writer.writerow(data)


def collect_loss_gate_results(m, model_type='moe_expectation_model', temps=[1.0], w_importance_range=[0.0], 
                    w_sample_sim_same_range=[0.0], w_sample_sim_diff_range=[0.0],
                    total_experts=5, num_classes=10, num_epochs=20, 
                    testloader=None, top_5=False, model_path=None, results_path=None, filename ='mnist_results.csv' ):
    
    filename = os.path.join(results_path, filename)
    
    if os.path.exists(filename):
        p = 'a'
    else:
        p = 'w'
        
    if top_5:
        header = ['filename', 'val error', 'top-1 error', 'top-5 error', 'mutual information', 'sample entropy', 'experts usage', 'per_task_entropy']
    else:
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
                running_top5_accuracy = 0.0
                
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
                    if top_5:
                        running_top5_accuracy += accuracy_top5(outputs, test_labels)
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
                if top_5:
                    top5_error = 1-(running_top5_accuracy/num_batches)
            
                data.append(top1_error.item())
                if top_5:
                    data.append(top5_error)
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
            
            