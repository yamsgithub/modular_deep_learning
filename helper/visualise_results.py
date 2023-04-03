import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps 
import seaborn as sns
import decimal

from itertools import product
import decimal
import os

import numpy as np

import pandas as pd

from sklearn.metrics import confusion_matrix

import torch

from helper.moe_models import entropy

# helper function to show an image
def imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

colors = ['y', 'tab:purple', 'tab:green', 'tab:orange','tab:blue', 'tab:red','tab:cyan', 'tab:olive']

# ### Visualise decision boundaries of mixture of expert model, expert model and gate model
def plot_data(X, y, num_classes, save_as):
    f, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,8))
    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y,palette=colors[0:num_classes], ax=ax)
    ax.set_title("2D 3 classes Generated Data")
    plt.ylabel('Dim 2')
    plt.xlabel('Dim 1')
    plt.savefig(save_as)
    #plt.show()
    plt.clf()
    plt.close()
    

def labels(p, palette=['r','c','y','g','b','tab:pink']):
    pred_labels = torch.argmax(p, dim=1)+1
    uniq_y = np.unique(pred_labels.cpu())
    pred_color = [palette[i-1] for i in uniq_y]
    return pred_color, pred_labels


def predict(dataloader, model):
        
        pred_labels = []
        true_labels = []
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            true_labels.append(labels)
            pred_labels.append(torch.argmax(model(inputs), dim=1))
            
        return torch.stack(true_labels), torch.stack(pred_labels)

def plot_results(X, y, generated_data, num_classes, trainset, trainloader, testset, testloader, models, dataset, total_experts, filename):

    keys = models.keys()
    ncols = len(keys) 
    for e in range(total_experts, total_experts+1):
        nrows = (e*1)+3
        thefigsize = (ncols*5,nrows*5)
        
        print('Number of Experts:', e)
        
        fig,ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=thefigsize)
        ax = ax.flatten()
    
        print(keys)

        index = 0
        for m_key, m_val in models.items():

            if m_key == 'single_model':
                continue
        
            moe_model = m_val['experts'][e]['model']
            
            pred = moe_model(generated_data.to(device))
            pred_color,pred_labels = labels(pred)
            pred_labels_order = np.unique(pred_labels.cpu())
            sns.scatterplot(x=generated_data[:,0].cpu(),y=generated_data[:,1].cpu(),
                            hue=pred_labels.cpu(), hue_order=pred_labels_order, palette=pred_color, legend=False, ax=ax[index])
            sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, hue_order=list(range(0,num_classes)),palette=colors[0:num_classes], ax=ax[index])
            ax[index].set_title(' '.join(m_key.split('_')))
            ax[index].set_ylabel('Dim 2')
            ax[index].set_xlabel('Dim 1')
            
        
            experts = moe_model.experts

            for i in range(0, e):
                pred = experts[i](generated_data.to(device))
                pred_color,pred_labels = labels(pred)
                pred_labels_order = np.unique(pred_labels.cpu())
                sns.scatterplot(x=generated_data[:,0].cpu(),y=generated_data[:,1].cpu(),
                                hue=pred_labels.cpu(), hue_order=pred_labels_order, palette=pred_color, legend=False, ax=ax[((i+1)*ncols)+index])
                sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, hue_order=list(range(0,num_classes)), palette=colors[0:num_classes],  ax=ax[((i+1)*ncols)+index])
                
                ax[((i+1)*ncols)+index].set_title('Expert '+str(i+1)+' Model')
                ax[((i+1)*ncols)+index].set_ylabel('Dim 2')
                ax[((i+1)*ncols)+index].set_xlabel('Dim 1')

            #palette = sns.husl_palette(total_experts)
            palette = sns.color_palette("Paired")+sns.color_palette('Set2')
            pred_gate = moe_model.gate(generated_data.to(device))
            pred_gate_color, pred_gate_labels = labels(pred_gate, palette)
            pred_gate_labels_order = np.unique(pred_gate_labels.cpu())
            sns.scatterplot(x=generated_data[:,0].cpu(),y=generated_data[:,1].cpu(),
                            hue=pred_gate_labels.cpu(), hue_order=pred_gate_labels_order, palette=pred_gate_color, legend=False, ax=ax[((e+1)*ncols)+index])
            sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, hue_order=list(range(0,num_classes)), palette=colors[0:num_classes], ax=ax[((e+1)*ncols)+index])
            ax[((e+1)*ncols)+index].set_title('Gate Model')
            ax[((e+1)*ncols)+index].set_ylabel('Dim 2')
            ax[((e+1)*ncols)+index].set_xlabel('Dim 1')
        
        
            pred_gate = moe_model.gate(trainset[:][0].to(device))
            pred_gate_color, pred_gate_labels = labels(pred_gate, palette)
            
            sns.scatterplot(x=trainset[:][0][:,0].cpu(),y=trainset[:][0][:,1].cpu(),
                            hue=pred_gate_labels.cpu(),palette=pred_gate_color, ax=ax[((e+2)*ncols)+index])       
            
            
            index += 1
        fig.savefig(filename +'_'+m_key+'_'+str(e)+'_experts.png')
        #plt.clf()
        #plt.close()
        plt.show()

def plot_error_rate(models, total_experts, save_as):
    labels = []
    plt.figure(figsize=(20,10))
    for m_key, m_val in models.items():
        labels.append(m_key)
        accuracies = []
        for i in range(1, total_experts+1):                
            history = m_val['experts'][i]['history']
            accuracies.append(history['accuracy'][-1])
        # Plot error rate (1-accuracy) 
        plt.plot(range(1,len(accuracies)+1), 1-np.asarray(accuracies))
    plt.legend(labels)
    plt.ylim(0, 1)
    plt.xticks(range(1, total_experts+1), [str(i) for i in range(1, total_experts+1)])
    plt.xlabel('Number of Experts')
    plt.ylabel('Error rate')
    plt.savefig(save_as)
    plt.clf()
    plt.close()


def plot_accuracy_by_experts(models, total_experts, save_as):
    for m_key, m_val in models.items():
        plt.figure(figsize=(20,10))
        accuracies = []
        for i in range(1, total_experts+1):                
            history = m_val['experts'][i]['history']
            accuracies.append(history['accuracy'][-1])
        plt.plot(range(1,len(accuracies)+1), accuracies)
        plt.plot(range(1,len(accuracies)+1), accuracies)
        plt.title(model_name)
        plt.legend(['Simple Expert', 'Deep Expert'])
        plt.xticks(range(1, total_experts+1), [str(i) for i in range(1, total_experts+1)])
        plt.xlabel('Number of Experts')
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.savefig('figures/all/accuracy_'+model_name+'_'+dataset+'_'+ str(num_classes)+'_experts_.png')
        #plt.show()
        plt.clf()
        plt.close()


def generate_plot_file(dataset, temp=1.0, t_decay=0.0, no_gate_T=1.0, w_importance=0.0, w_sample_sim_same=0.0, w_sample_sim_diff=0.0, specific=''):
    plot_file = dataset
    if w_importance > 0:
        plot_file += '_importance_'+'{:.1f}'.format(w_importance)
    if not temp == 1.0:
        plot_file += '_temp_'+'{:.1f}'.format(temp)
    if t_decay > 0:
        if t_decay < 1:
            plot_file += '_t_decay_'+str(t_decay)
        else:
            plot_file += '_t_decay_'+'{:.1f}'.format(t_decay)
    if not no_gate_T == 1.0:
        if w_sample_sim_diff < 1:
            plot_file += '_no_gate_T_'+str(no_gate_T)
        else: 
            plot_file += '_no_gate_T_'+'{:.1f}'.format(no_gate_T)
    if w_sample_sim_same > 0:
        if w_sample_sim_same < 1:
            plot_file += '_sample_sim_same_'+str(w_sample_sim_same)
        else:
            plot_file += '_sample_sim_same_'+'{:.1f}'.format(w_sample_sim_same)
    if w_sample_sim_diff > 0:
        if w_sample_sim_diff < 1:
            plot_file += '_sample_sim_diff_'+str(w_sample_sim_diff)
        else:
            plot_file += '_sample_sim_diff_'+'{:.1f}'.format(w_sample_sim_diff)
    plot_file += '_'+specific
    
    return plot_file


def find_best_model(m, model_type='moe_expectation_model', temps=[[1.0]*20], T_decay=[0.0], w_importance_range=[],  
            w_sample_sim_same_range=[], w_sample_sim_diff_range=[], 
                    total_experts=5, num_classes=10, num_epochs=20, model_path=None, device='cpu'):

    train_error = 0.0
    min_val_error = float('inf')
    mutual_info = 0.0
    sample_entropy = 0.0
    expert_usage = 0.0
    best_model = None
    best_model_file = None
    best_model_index = 0
    if w_importance_range:
         for T, decay, w_importance in product(temps, T_decay, w_importance_range):
        
              plot_file = generate_plot_file(m, temp=T[0], t_decay=decay, w_importance=w_importance,
                               specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

              models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

              for i, model in enumerate(models):
                  # for e_key, e_val in model.items():
                  history = model[model_type]['experts'][total_experts]['history']
                  val_error = 1-history['val_accuracy'][-1]
                  if min_val_error > val_error:
                     min_val_error = val_error
                     train_error = 1-history['accuracy'][-1]
                     mutual_info = history['mutual_EY'][-1]
                     sample_entropy = history['sample_entropy'][-1]
                     expert_usage = expert_usage_entropy(history,total_experts,num_epochs)
                     best_model = model
                     best_model_file = plot_file
                     best_model_index = i

    for w_sample_sim_same, w_sample_sim_diff in product(w_sample_sim_same_range, w_sample_sim_diff_range):
        
        plot_file = generate_plot_file(m, w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff,                                
                               specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

        models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

        for i, model in enumerate(models):
            # for e_key, e_val in model.items():
            history = model[model_type]['experts'][total_experts]['history']
            val_error = 1-history['val_accuracy'][-1]

            if min_val_error > val_error:
                min_val_error = val_error
                train_error = history['accuracy'][-1]
                mutual_info = history['mutual_EY'][-1]
                sample_entropy = history['sample_entropy'][-1]
                expert_usage = expert_usage_entropy(history,total_experts,num_epochs)
                best_model = model
                best_model_file = plot_file
                best_model_index = i

    print('Train Accuracy','{:.3f}'.format(train_error))
    print('Max Validation Accuracy','{:.3f}'.format(1-min_val_error))
    print('Min Validation Error','{:.3f}'.format(min_val_error))
    print('Mutual Information', '{:.3f}'.format(mutual_info))
    print('Sample Entropy', '{:.3f}'.format(sample_entropy))
    print('Expert Usage', '{:.3f}'.format(expert_usage))
    return best_model, best_model_file, best_model_index



def plot_expert_usage(m, model_type='moe_expectation_model',test_loader=None, temps=[[1.0]*20], T_decay=[0.0], 
                      w_importance_range=[],w_sample_sim_same_range=[], w_sample_sim_diff_range=[], 
                      total_experts=5, num_classes=10, classes=list(range(10)),num_epochs=20, 
                      fig_path=None, model_path=None, dataset='MNIST', 
                      annot=True, best=True, index=0, device=torch.device("cpu")):
    
    plt.tight_layout()
    
    fontsize = 20
    fontsize_label = 12
    
    if best:
        model, model_file, best_model_index = find_best_model(m, model_type=model_type, temps=temps, T_decay=T_decay, 
                                                              w_importance_range=w_importance_range,
                                                              w_sample_sim_same_range=w_sample_sim_same_range, 
                                                              w_sample_sim_diff_range=w_sample_sim_diff_range, 
                                                              num_classes=num_classes, total_experts=total_experts, 
                                                          num_epochs=num_epochs, model_path=model_path, device=device)
        print('Best model index', best_model_index)
    else:
        w_importance = w_importance_range[0] if w_importance_range else 0.0
        w_sample_sim_same = w_sample_sim_same_range[0] if w_sample_sim_same_range else 0.0
        w_sample_sim_diff = w_sample_sim_diff_range[0] if w_sample_sim_diff_range else 0.0
        
        plot_file = generate_plot_file(m, temp=temps[0][0], w_importance=w_importance,
                                       w_sample_sim_same=w_sample_sim_same, 
                                       w_sample_sim_diff=w_sample_sim_diff, 
                               specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')
        model_file = plot_file
        models = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)
        print('Num  models', len(models))
        model = models[index]
        
    
    print(model_file)
    # for e_key, e_val in model.items():

    history = model[model_type]['experts'][total_experts]['history']
    gate_probabilities = torch.vstack(history['gate_probabilities']).view(num_epochs,-1,total_experts)

    gate_probabilities_sum = torch.sum(gate_probabilities, dim =1).cpu().detach().numpy()        

    fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8, 6))

    palette = sns.color_palette("Set2", total_experts )
    for i in range(total_experts):
        sns.lineplot(x=range(num_epochs), y=gate_probabilities_sum[:,i], 
                     hue=[i]*num_epochs, palette=palette[i:i+1], ax=ax)

    ax.set_ylim(bottom=0)
    plt.xlabel('Epochs')
    plt.ylabel('Number of Samples')
    plt.legend(['E'+str(i+1) for i in range(total_experts)])
    plt.title('Number of samples sent to each expert during \n training with '+str(gate_probabilities.shape[1])+' samples of ' +dataset+ ' train dataset and '+str(total_experts)+' experts', fontsize=14)
    plot_file = model_file.replace('models.pt', 'sample_distribution.png')
    plt.savefig(os.path.join(fig_path, plot_file))
    plt.show()

    cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    with torch.no_grad():
        gate_outputs_all = []
        pred_labels_all = []
        labels_all = []
        exp_class_prob = torch.zeros(total_experts, num_classes).to(device)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            moe_model = model[model_type]['experts'][total_experts]['model']
            moe_model.device = device

            # predict the classes for test data
            if model_type == 'moe_no_gate_self_information_model':
                pred = moe_model(images, targets=labels)
            else:
                pred = moe_model(images)

            pred_labels = torch.argmax(pred, dim=1)

            labels_all.append(labels)
            pred_labels_all.append(pred_labels)

            expert_outputs = moe_model.expert_outputs
            gate_outputs = moe_model.gate_outputs

            gate_outputs_all.append(gate_outputs)

        gate_outputs = torch.vstack(gate_outputs_all)
        labels = torch.hstack(labels_all)
        pred_labels = torch.hstack(pred_labels_all)
        
        # get the experts selected by the gate for each sample
        selected_experts = torch.argmax(gate_outputs, dim=1)

        # plot the expert selection table
        class_expert_table = np.asarray([[0] * num_classes]*total_experts)
        for label, expert in zip(labels, selected_experts):
            class_expert_table[expert,label] += 1

        fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8, 6))
        x = ['E '+str(i+1) for i in range(total_experts)]
        
        if model_type == 'moe_expert_loss_model':
            y = np.sum(class_expert_table, axis=1)
        else:
            y = torch.sum(gate_outputs, dim=0).cpu().numpy()
        

        sns.barplot(x=x, y=y, palette=palette, ax=ax)
        plt.xlabel('Experts')
        plt.ylabel('Number of samples', fontsize=fontsize_label)
        ax.tick_params(axis='both', labelsize=10)

        plt.title('Number of samples sent to each expert during inference \n with '+str(len(test_loader.dataset))+ ' samples of '+dataset+ ' test dataset and '+str(total_experts)+' experts', fontsize=14)
        plot_file = model_file.replace('models.pt', 'expert_usage.png')
        plt.savefig(os.path.join(fig_path, plot_file))
        plt.show()

        for e in range(total_experts):
            for index, l in enumerate(labels):
                exp_class_prob[e,l] += gate_outputs[index,e].to(device)

        exp_total_prob = torch.sum(exp_class_prob, dim=1).view(-1,1).to(device)
        fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(18,6))

        sns.heatmap(exp_class_prob.cpu().numpy().astype(int), yticklabels=['E'+str(i) for i in range(1,total_experts+1)], 
                    xticklabels=[classes[i] for i in range(0, num_classes)],
                    cmap=cmap, annot=annot, fmt='d', ax=ax)
        plt.show()

        fig1,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(18, 6))
        sns.heatmap(class_expert_table, yticklabels=['E'+str(i) for i in range(1,total_experts+1)], 
                    xticklabels=[classes[i] for i in range(0, num_classes)],
                    annot=annot, cmap=cmap, fmt='d', ax=ax)

        plt.title('Experts selected per class for '+str(len(test_loader.dataset))+' samples of '+dataset+' test data', 
                  fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plot_file = model_file.replace('models.pt', 'class_expert_table.png')
        plt.savefig(os.path.join(fig_path, plot_file),bbox_inches='tight')

        fig1,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(12, 8))
        sns.heatmap(confusion_matrix(labels.cpu(), pred_labels.cpu()), 
                    xticklabels=[classes[i] for i in range(0, num_classes)],
                    yticklabels=[classes[i] for i in range(0, num_classes)], 
                    annot=annot, cmap=cmap, fmt='d', ax=ax)

        plt.show()


def expert_usage_entropy(history, total_experts=5, num_epochs=20):
    gate_probability = torch.vstack(history['gate_probabilities']).view(num_epochs, -1, total_experts)
    gate_probabilities_sum = torch.mean(gate_probability[-1,:,:].view(-1, total_experts), dim=0)
    return entropy(gate_probabilities_sum).item()


def boxplot(model_single=None, model_with_temp=None,model_with_temp_decay=None, 
            model_with_reg=None, model_without_reg=None, model_with_reg_temp=None, 
            model_with_attention=None, model_with_attn_reg=None, model_with_attn_sample_sim_reg=None,
            model_dual_temp_with_attention = None, 
            model_output_reg =None, model_output_imp_reg=None,model_temp_output_reg=None, 
            mnist_attn_output_reg=None, model_sample_sim_reg=None,model_with_exp_reg=None,
            temps=[1.0], T_decay=[], w_importance_range=[0.0], w_ortho_range=[0.0], 
            w_sample_sim_same_range=[0.0], w_sample_sim_diff_range=[0.0],
            total_experts=5, num_classes=10, num_epochs=20, classes=None, testloader=None, figname=None,fig_path=None, model_path=None, device=torch.device("cpu")):

    x = []
    hues = []
    x_temp = []
    y_error = []
    y_val_error = []
    y_mi = []
    y_H_EY = []
    y_sample_H = []
    y_sample_H_T = []
    y_sample_hue = []
    y_expert_usage = []

    def collect_data(models, x_label, label):
        
        for model in models:
            for e_key, e_val in model.items():
                history = model[e_key]['experts'][total_experts]['history']
                error = 1-torch.vstack(history['accuracy'])
                val_error = 1-torch.vstack(history['val_accuracy'])
                y_error.append(error[-1])
                y_val_error.append(val_error[-1])
                y_mi.append(history['mutual_EY'][-1])
                y_H_EY.append(history['H_EY'][-1])
                if 'top_1' in e_key or 'stochastic' in e_key:
                    y_sample_H.append(torch.tensor(0.0))
                else:
                    y_sample_H.append(history['sample_entropy'][-1])
                y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))

                x.append(x_label)
                hues.append(label)
        
    w_importance = 0.0
    
    if not model_single is None:
        m = model_single

        plot_file = generate_plot_file(m, specific=str(num_classes)+'_models.pt')

        model_0 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)
        for history in model_0['history']:
             error = 1-torch.tensor(history['accuracy']).to(device)
             val_error = 1-torch.tensor(history['val_accuracy']).to(device)
             y_error.append(error[-1])
             y_val_error.append(val_error[-1])
             x.append('SM')
             hues.append('Single Model')

    if not model_without_reg is None:

        for name, m in model_without_reg.items():

            if name == 'ignore':
                x_label = 'I 0.0'
                label = 'MoE'
            else:
                x_label = 'I ' + name +' 0.0' 
                label = name + ' MoE '
                
            plot_file = generate_plot_file(m, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            model_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

            collect_data(model_1, x_label, label)

      
    if not model_with_reg is None:

        for name, m in model_with_reg.items():

            for w_importance in w_importance_range:

                if name == 'ignore':
                    x_label = 'I '+"{:.1f}".format(w_importance)
                    label = 'MoE with importance regularization'
                else:
                    x_label = 'I '+name+" {:.1f}".format(w_importance)
                    label = name + ' MoE with importance regularization'
            
                plot_file = generate_plot_file(m, w_importance=w_importance, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

                model_2 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

                collect_data(model_2, x_label, label)


    w_importance = 0.0
    
    if not model_with_temp is None:
        
        for name, m in model_with_temp.items():

            for T in temps:   
            
                plot_file = generate_plot_file(m, temp=T, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

                model_3 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

                for model in model_3:
                    for e_key, e_val in model.items():
                        history = model[e_key]['experts'][total_experts]['history']
                        error = 1-torch.vstack(history['accuracy'])
                        val_error = 1-torch.vstack(history['val_accuracy'])
                        y_error.append(error[-1])
                        y_val_error.append(val_error[-1])
                        y_mi.append(history['mutual_EY'][-1])
                        y_H_EY.append(history['H_EY'][-1])
                        y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                        
                        if 'top_1' in e_key or 'stochastic' in e_key:
                            y_sample_H.append(torch.tensor(0.0))
                            y_sample_H_T.append(torch.tensor(0.0))
                            y_sample_H_T.append(torch.tensor(0.0))
                        else:
                            y_sample_H.append(history['sample_entropy'][-1])
                            y_sample_H_T.append(history['sample_entropy'][-1])
                            y_sample_H_T.append(history['sample_entropy_T'][-1])
                        
                        if name == 'ignore':
                            x_label = 'T '+"{:.1f}".format(T)
                            label = 'MoE'
                            hues.append('MoE with dual temp')
                        else:
                            x_label = 'T ' + name + " {:.1f}".format(T)
                            label = 'MoE ' + name
                            hues.append(name+' MoE with dual temp')
                        x.append(x_label)
                        
                        x_temp.append('T '+"{:.1f}".format(T))
                        x_temp.append('T '+"{:.1f}".format(T))
                        y_sample_hue.append('Low Temp')
                        y_sample_hue.append('High Temp')               
                        

    if not model_with_temp_decay is None:
        
        for name, m in model_with_temp_decay.items():
            
            for T in temps:   

                for decay in T_decay:

                    if decay < 1:
                        D = str(decay)
                    else:
                        D = "{:.1f}".format(decay)

                    plot_file = generate_plot_file(m, temp=T, t_decay=decay, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

                    # model you build above
                    model_3 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

                    for model in model_3:
                        for e_key, e_val in model.items():
                            history = model[e_key]['experts'][total_experts]['history']
                            error = 1-torch.vstack(history['accuracy'])
                            val_error = 1-torch.vstack(history['val_accuracy'])
                            y_error.append(error[-1])
                            y_val_error.append(val_error[-1])
                            y_mi.append(history['mutual_EY'][-1])
                            y_H_EY.append(history['H_EY'][-1])
                            if 'top_1' in e_key or 'stochastic' in e_key:
                                y_sample_H.append(torch.tensor(0.0))
                            else:
                                y_sample_H.append(history['sample_entropy'][-1])
                            y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                            
                            if name == 'ignore':
                                x_label = 'T '+"{:.1f}".format(T) +' D '+ D
                                label = 'MoE'
                                hues.append('MoE with dual temp on decay')
                            else:
                                x_label = 'T ' + name + " {:.1f}".format(T) +' D '+ D
                                label = 'MoE ' + name
                                hues.append(name + ' MoE with dual temp on decay')

                            x.append(x_label)
                       
                            

    
    if not model_with_reg_temp is None:
        m = model_with_reg_temp

        for T, w_importance in product(temps, w_importance_range):   

            plot_file = generate_plot_file(m, temp=T, w_importance=w_importance, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            # Note: Here we are loading the pre-trained model from 'pre_trained_model_path. Change this to 'model_path' to load the 
            # model you build above
            model_4 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

            for model in model_4:
                for e_key, e_val in model.items():
                    history = model[e_key]['experts'][total_experts]['history']
                    error = 1-torch.vstack(history['accuracy'])
                    val_error = 1-torch.vstack(history['val_accuracy'])
                    y_error.append(error[-1])
                    y_val_error.append(val_error[-1])
                    y_mi.append(history['mutual_EY'][-1])
                    y_H_EY.append(history['H_EY'][-1])
                    y_sample_H.append(history['sample_entropy'][-1])
                    y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                    x.append('T '+"{:.1f}".format(T)+'+ I '+"{:.1f}".format(w_importance))
                    hues.append('Moe with reg and dual temp')
                
    if not model_output_reg is None:
        m = model_output_reg

        for w_ortho in w_ortho_range:

            plot_file = generate_plot_file(m, w_ortho=w_ortho, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            model_5 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

            for model in model_5:
                for e_key, e_val in model.items():
                    history = model[e_key]['experts'][total_experts]['history']
                    error = 1-torch.vstack(history['accuracy'])
                    val_error = 1-torch.vstack(history['val_accuracy'])
                    y_error.append(error[-1])
                    y_val_error.append(val_error[-1])
                    y_mi.append(history['mutual_EY'][-1])
                    y_H_EY.append(history['H_EY'][-1])
                    y_sample_H.append(history['sample_entropy'][-1])
                    y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                    x.append('O '+"{:.1f}".format(w_ortho))
                    hues.append('MoE with output regularization')
    
    if not model_output_imp_reg is None:
        m = model_output_imp_reg

        for w_ortho, w_importance in product(w_ortho_range, w_importance_range):

            plot_file = generate_plot_file(m, w_importance=w_importance, w_ortho=w_ortho, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            model_5 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

            for model in model_5:
                for e_key, e_val in model.items():
                    history = model[e_key]['experts'][total_experts]['history']
                    error = 1-torch.vstack(history['accuracy'])
                    val_error = 1-torch.vstack(history['val_accuracy'])
                    y_error.append(error[-1])
                    y_val_error.append(val_error[-1])
                    y_mi.append(history['mutual_EY'][-1])
                    y_H_EY.append(history['H_EY'][-1])
                    y_sample_H.append(history['sample_entropy'][-1])
                    y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                    x.append('I '+"{:.1f}".format(w_importance) + ' + O '+"{:.1f}".format(w_ortho))
                    hues.append('MoE with output and gate regularization')
                    

                
    if not model_temp_output_reg is None:
        m = model_temp_output_reg

        for T, w_ortho in product(temps, w_ortho_range):   

            plot_file = generate_plot_file(m, temp=T,w_ortho=w_ortho, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            # Note: Here we are loading the pre-trained model from 'pre_trained_model_path. Change this to 'model_path' to load the 
            # model you build above
            model_4 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

            for model in model_4:
                for e_key, e_val in model.items():
                    history = model[e_key]['experts'][total_experts]['history']
                    error = 1-torch.vstack(history['accuracy'])
                    val_error = 1-torch.vstack(history['val_accuracy'])
                    y_error.append(error[-1])
                    y_val_error.append(val_error[-1])
                    y_mi.append(history['mutual_EY'][-1])
                    y_H_EY.append(history['H_EY'][-1])
                    y_sample_H.append(history['sample_entropy'][-1])
                    y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                    x.append('T '+"{:.1f}".format(T)+'+ O '+"{:.1f}".format(w_ortho))
                    hues.append('Moe with outputreg and dual temp')
                    
    if not model_sample_sim_reg is None:

        for name, m in model_sample_sim_reg.items():        

            for w_sample_sim_same, w_sample_sim_diff in product(w_sample_sim_same_range, w_sample_sim_diff_range):

                plot_file = generate_plot_file(m, w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')
                
                model_2 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)
                
                w_sample_sim = ''
                if w_sample_sim_same < 0.1:
                    w_sample_sim += 'S%.0e' % decimal.Decimal(w_sample_sim_same)
                else:
                    w_sample_sim += 'S'+"{:.1f}".format(w_sample_sim_same)

                if w_sample_sim_diff < 0.1:
                    w_sample_sim += 'D%.0e' % decimal.Decimal(w_sample_sim_diff)
                else:
                    w_sample_sim += 'D'+"{:.1f}".format(w_sample_sim_diff)

                error_values = []
                for model in model_2:
                    for e_key, e_val in model.items():
                        history = model[e_key]['experts'][total_experts]['history']
                        error = 1-torch.vstack(history['accuracy'])
                        val_error = 1-torch.vstack(history['val_accuracy'])
                        y_error.append(error[-1])
                        y_val_error.append(val_error[-1])
                        y_mi.append(history['mutual_EY'][-1])
                        y_H_EY.append(history['H_EY'][-1])
                        y_sample_H.append(history['sample_entropy'][-1])
                        y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                        if name == 'ignore':
                            x.append('SS ' + w_sample_sim)
                            hues.append('MoE with sample similarity regularization')
                        else:
                            x.append('SS ' +name +' '+ w_sample_sim)
                            hues.append('MoE '+name+' with sample similarity regularization')
                        
    if not model_with_attention is None:
        
        for name, m in model_with_attention.items():

            if name == 'ignore':
                x_label = 'Attention'
                label = 'MoE with attention'
            else:
                x_label = 'Attention ' + name 
                label = 'MoE ' + name + ' with attention'

            plot_file = generate_plot_file(m, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            model_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

            collect_data(model_1, x_label, label)

    if not model_with_exp_reg is None:
        
        m = model_with_exp_reg

        for w_exp_gamma in w_exp_gamma_range:
            
            plot_file = generate_plot_file(m, w_exp_gamma=w_exp_gamma, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            model_2 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

            error_values = []
            for model in model_2:
                for e_key, e_val in model.items():
                    history = model[e_key]['experts'][total_experts]['history']
                    error = 1-torch.vstack(history['accuracy'])
                    val_error = 1-torch.vstack(history['val_accuracy'])
                    y_error.append(error[-1])
                    y_val_error.append(val_error[-1])
                    y_mi.append(history['mutual_EY'][-1])
                    y_H_EY.append(history['H_EY'][-1])
                    y_sample_H.append(history['sample_entropy'][-1])
                    y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                    x.append('ER '+"{:.3f}".format(w_exp_gamma).strip('0'))
                    hues.append('MoE with exponential reg') 
            
    if not model_with_attn_reg is None:
        
        for name, m in model_with_attn_reg.items():
 
            for w_importance in w_importance_range:

                if name == 'ignore':
                    x_label = 'Attn + I '+"{:.1f}".format(w_importance)
                    label = 'MoE with attention and regularization'
                else:
                    x_label = 'Attn + I '+name+" {:.1f}".format(w_importance)
                    label = 'MoE '+name+' with attention and regularization'

                plot_file = generate_plot_file(m, w_importance=w_importance, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

                model_2 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

                collect_data(model_2, x_label, label)

    if not model_with_attn_sample_sim_reg is None:
        
        for name, m in model_with_attn_sample_sim_reg.items():
                        
            for w_sample_sim_same, w_sample_sim_diff in product(w_sample_sim_same_range, w_sample_sim_diff_range):

                w_sample_sim = ''
                if w_sample_sim_same < 0.1:
                    w_sample_sim += 'S%.0e' % decimal.Decimal(w_sample_sim_same)
                else:
                    w_sample_sim += 'S'+"{:.1f}".format(w_sample_sim_same)

                if w_sample_sim_diff < 0.1:
                    w_sample_sim += 'D%.0e' % decimal.Decimal(w_sample_sim_diff)
                else:
                    w_sample_sim += 'D'+"{:.1f}".format(w_sample_sim_diff)

                if name == 'ignore':
                    x_label = 'Attn + SS ' + w_sample_sim
                    label = 'MoE with attention and sample similarity regularization'
                else:
                    x_label = 'Attn + SS ' +name +' '+ w_sample_sim
                    label = 'MoE '+name+' with attention and sample similarity regularization'

                plot_file = generate_plot_file(m, w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff,
                                               specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

                model_2 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

                collect_data(model_2, x_label, label)
    
    if not  model_dual_temp_with_attention is None:

        m = model_dual_temp_with_attention

        for T in temps:   

            plot_file = generate_plot_file(m, temp=T, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')


            # Note: Here we are loading the pre-trained model from 'pre_trained_model_path. Change this to 'model_path' to load the 
            # model you build above
            model_3 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

            for model in model_3:
                for e_key, e_val in model.items():
                    history = model[e_key]['experts'][total_experts]['history']
                    error = 1-torch.vstack(history['accuracy'])
                    val_error = 1-torch.vstack(history['val_accuracy'])
                    y_error.append(error[-1])
                    y_val_error.append(val_error[-1])
                    y_mi.append(history['mutual_EY'][-1])
                    y_H_EY.append(history['H_EY'][-1])
                    y_sample_H.append(history['sample_entropy'][-1])
                    y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                    y_sample_H_T.append(history['sample_entropy'][-1])
                    y_sample_H_T.append(history['sample_entropy_T'][-1])
                    x.append('Attn + T '+"{:.1f}".format(T))
                    x_temp.append('Attn + T '+"{:.1f}".format(T))
                    x_temp.append('Attn + T '+"{:.1f}".format(T))
                    y_sample_hue.append('Low Temp')
                    y_sample_hue.append('High Temp')               
                    hues.append('Moe with dual temp and attention')
                    
    if not mnist_attn_output_reg is None:
        m = mnist_attn_output_reg

        for w_ortho in w_ortho_range:
            
            plot_file = generate_plot_file(m, w_ortho=w_ortho, specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')

            model_2 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

            for model in model_2:
                for e_key, e_val in model.items():
                    history = model[e_key]['experts'][total_experts]['history']
                    error = 1-torch.vstack(history['accuracy'])
                    val_error = 1-torch.vstack(history['val_accuracy'])
                    y_error.append(error[-1])
                    y_val_error.append(val_error[-1])
                    y_mi.append(history['mutual_EY'][-1])
                    y_H_EY.append(history['H_EY'][-1])
                    y_sample_H.append(history['sample_entropy'][-1])
                    y_expert_usage.append(expert_usage_entropy(history,total_experts,num_epochs))
                    x.append('Attn + O '+"{:.1f}".format(w_ortho))
                    hues.append('MoE with attention and output reg')
                
            
    palette = sns.color_palette("Set2")
    fontsize = 20
    labelsize = 15
    plt.tight_layout()
    
    
    _, indices = np.unique(np.asarray(x), return_index=True)
    xlabels = np.asarray(x)[sorted(indices)]
    
    fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(20, 8))
    sns.boxplot(x=x, y=torch.hstack(y_error).cpu().numpy(), hue=hues, palette=palette, dodge=False, ax=ax)
    ax.set_title('Comparing training errors for different MoE training methods', fontsize=fontsize)
    ax.set_xlabel('MoE training methods', fontsize=labelsize)
    ax.set_ylabel('training error', fontsize=labelsize)      
    ax.set_xticklabels(xlabels, rotation='vertical')
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=labelsize)

    plot_file = generate_plot_file(figname, specific='training_error_boxplot.png')             
    plt.savefig(os.path.join(fig_path, plot_file),bbox_inches='tight')
    
    plt.show()
    
    fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(20, 8))
    sns.boxplot(x=x, y=torch.hstack(y_val_error).cpu().numpy(), hue=hues,palette=palette, dodge=False, ax=ax)
    ax.set_title('Comparing validation errors for different MoE training methods', fontsize=fontsize)
    ax.set_ylabel('validation error', fontsize=labelsize)
    ax.set_xlabel('MoE training methods', fontsize=labelsize)
    ax.set_xticklabels(xlabels, rotation='vertical')
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=labelsize)
    
    plot_file = generate_plot_file(figname, specific='val_error_boxplot.png')             
    plt.savefig(os.path.join(fig_path, plot_file),bbox_inches='tight')
    plt.show()
    
    if not model_single is None:
        x = x[10:]
        hues = hues[10:]
        palette = palette[1:]
        xlabels = xlabels[1:]
        
    fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(20, 8))
    sns.boxplot(x=x, y=torch.hstack(y_mi).cpu().numpy(), hue=hues, palette=palette, dodge=False,ax=ax)
    ax.set_title('Comparing joint mutual information of experts $E$ and MoE model output $Y$ for different MoE training methods', fontsize=fontsize)
    ax.set_ylabel('EY mutual information', fontsize=labelsize)  
    ax.set_xlabel('MoE training methods', fontsize=labelsize)
    ax.set_xticklabels(xlabels, rotation='vertical')
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=labelsize)

    plot_file = generate_plot_file(figname, specific='mutual_info_boxplot.png')             
    plt.savefig(os.path.join(fig_path, plot_file),bbox_inches='tight')
    
    plt.show()
    
#     fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(12, 8))
#     sns.boxplot(x=x, y=y_H_EY, hue=hues, palette=palette, dodge=False,ax=ax)
#     ax.set_ylabel('EY entropy')
#     ax.set_title('EY entropy')
#     ax.set_xticklabels(xlabels, rotation='vertical')
#     plt.savefig(os.path.join(fig_path,'mnist_boxplot_ey_entropy.png'))
#     plt.show()
    
    fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(20, 8))
    sns.boxplot(x=x, y=torch.hstack(y_sample_H).cpu().numpy(), hue=hues, palette=palette, dodge=False, ax=ax)
    ax.set_title('Comparing per sample entropy for different MoE training methods', fontsize=fontsize)
    ax.set_ylabel('Per sample entropy', fontsize=labelsize)
    ax.set_xlabel('MoE training methods', fontsize=labelsize)
    ax.set_xticklabels(xlabels, rotation='vertical')
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=labelsize)

    plot_file = generate_plot_file(figname, specific='sample_entropy_boxplot.png')             
    plt.savefig(os.path.join(fig_path, plot_file),bbox_inches='tight')
    plt.show()
    
    fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(20, 8))
    sns.boxplot(x=x, y=y_expert_usage, hue=hues, palette=palette, dodge=False, ax=ax)
    ax.set_title('Comparing expert usage entropy for different MoE training methods', fontsize=fontsize)
    ax.set_ylabel('expert usage entropy', fontsize=labelsize)
    ax.set_xlabel('MoE training methods', fontsize=labelsize)
    ax.set_xticklabels(xlabels, rotation='vertical', fontsize=labelsize)
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=labelsize)

    plot_file = generate_plot_file(figname, specific='expert_usage_entropy_boxplot.png')             
    plt.savefig(os.path.join(fig_path, plot_file),bbox_inches='tight')
    plt.show()


    if y_sample_H_T:
        palette2 = sns.color_palette("hls", 8)

        fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(12, 8))
        sns.boxplot(x=x_temp, y=torch.hstack(y_sample_H_T).cpu().numpy(), hue=y_sample_hue, palette=[palette2[3], palette2[7]], ax=ax)
        ax.set_ylabel('Per sample entropy')
        ax.set_title('per sample entropy for high T')
        plt.show()

from sklearn.metrics import confusion_matrix

def plot_result_table(model_with_temp, model_with_reg, model_without_reg, temps, w_importance_range,
                 total_experts, num_classes, classes, testloader):
    
    num_epochs = 20

    min_values = []
    max_values = []
    mean_values = []
    std_values = []
    mutual_info = []
    models = []

    w_importance = 0.0

    for T in temps:    

        m = model_with_temp

        plot_file = generate_plot_file(m, w_importance,'temp_'+"{:.1f}".format(T)+'_'+str(num_classes)+'_'+str(total_experts)+'_models.pt')


        # Note: Here we are loading the pre-trained model from 'pre_trained_model_path. Change this to 'model_path' to load the 
        # model you build above
        model_1 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)

        print('Model:', plot_file)

        error_values = []
        for model in model_1:
            for e_key, e_val in model.items():
                history = model[e_key]['experts'][total_experts]['history']
                error = 1-np.asarray(history['accuracy'])
                error_values.append(error[-1])

        models.append(model_1[np.argmin(error_values)])

        min_values.append("{:.3f}".format(min(error_values)))
        max_values.append("{:.3f}".format(max(error_values)))
        mean_values.append("{:.3f}".format(mean(error_values)))
        std_values.append("{:.3f}".format(np.std(error_values)))
        mutual_info.append("{:.3f}".format(history['mutual_EY'][-1]))

    T = [ 'T '+"{:.1f}".format(t) for t in temps]
    N_T = len(T)

    for w_importance in w_importance_range:

        m = model_with_reg

        plot_file = generate_plot_file(m, w_importance, str(num_classes)+'_'+str(total_experts)+'_models.pt')


        # Note: Here we are loading the pre-trained model from 'pre_trained_model_path. Change this to 'model_path' to load the 
        # model you build above
        model_2 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)
        print('Model:', plot_file)

        error_values = []
        for model in model_2:
            for e_key, e_val in model.items():
                history = model[e_key]['experts'][total_experts]['history']
                error = 1-np.asarray(history['accuracy'])
                error_values.append(error[-1])

        models.append(model_2[np.argmin(error_values)])
#         models.append(model_2[-1])

        min_values.append("{:.3f}".format(min(error_values)))
        max_values.append("{:.3f}".format(max(error_values)))
        mean_values.append("{:.3f}".format(mean(error_values)))
        std_values.append("{:.3f}".format(np.std(error_values)))
        mutual_info.append("{:.3f}".format(history['mutual_EY'][-1]))

    N_I = len(w_importance_range)
    I = [ 'I '+"{:.1f}".format(i) for i in w_importance_range]

    m = model_without_reg

    plot_file = generate_plot_file(m, 0.0, str(num_classes)+'_'+str(total_experts)+'_models.pt')


    # Note: Here we are loading the pre-trained model from 'pre_trained_model_path. Change this to 'model_path' to load the 
    # model you build above
    model_3 = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)
    print('Model:', plot_file)

    error_values = []
    for model in model_3:
        history = model[e_key]['experts'][total_experts]['history']
        for e_key, e_val in model.items():
            error = 1-np.asarray(history['accuracy'])
            error_values.append(error[-1])

    models.append(model_3[np.argmin(error_values)])
#     models.append(model_3[-1])
    
    min_values.append("{:.3f}".format(min(error_values)))
    max_values.append("{:.3f}".format(max(error_values)))
    mean_values.append("{:.3f}".format(mean(error_values)))
    std_values.append("{:.3f}".format(np.std(error_values)))
    mutual_info.append("{:.3f}".format(history['mutual_EY'][-1]))

    method = T + I + ['I 0.0']
    N = N_T + N_I + 1

    print('N',N)
    data = np.hstack((np.asarray(method).reshape(N,1), np.asarray(min_values).reshape(N,1), np.asarray(max_values).reshape(N,1), 
                      np.asarray(mean_values).reshape(N,1), np.asarray(std_values).reshape(N,1), 
                      np.asarray(mutual_info).reshape(N,1)))

    print(data.shape)

    columns = ['Method', 'Min', 'Max', 'Mean', 'Std', 'Mutual Info']
    colors = np.array([['w']*len(columns)]*N)
    colors[np.argmin(data[:,1]), 1] = 'y'
    colors[np.argmax(data[:,2]), 2] = 'y'
    colors[np.argmin(data[:,3]), 3] = 'y'
    colors[np.argmin(data[:,4]), 4] = 'y'
    colors[np.argmax(data[:,5]), 5] = 'y'

    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=data, colLabels=columns, cellColours=colors, loc='center')

    fig.tight_layout()

    plt.show()


    model = models[np.argmin(data[0:N_T,1])]

    fig1,ax1 = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 4))
    ax1.flatten()

    # Plotting for the model with reg
    for e_key, e_val in model.items():


        cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                moe_model = e_val['experts'][total_experts]['model']

                # predict the classes for test data
                pred = moe_model(images)
                pred_labels = torch.argmax(pred, dim=1)

                expert_outputs = moe_model.expert_outputs
                gate_outputs = moe_model.gate_outputs

                # get the experts selected by the gate for each sample
                pred_gate_labels = torch.argmax(gate_outputs, dim=1)

                # plot the expert selection table
                print('\nExperts used by the gate for classification of each digit')
                class_expert_table = np.asarray([[0] * num_classes]*total_experts)
                for label, expert in zip(labels, pred_gate_labels):
                    class_expert_table[expert,label] += 1
                sns.heatmap(class_expert_table, yticklabels=['E'+str(i) for i in range(1,total_experts+1)], 
                            xticklabels=[classes[i] for i in range(0, num_classes)],
                            annot=True, cmap=cmap, fmt='d', ax=ax1[0])

                sns.heatmap(confusion_matrix(labels.cpu(), pred_labels.cpu()), annot=True, ax=ax1[1], cmap=cmap, fmt='d')

                plt.show()


    model = models[N_T+np.argmin(data[N_T:N-1,1])]
    
    fig1,ax1 = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 4))
    ax1.flatten()

    # Plotting for the model without reg
    for e_key, e_val in model.items():
        cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                moe_model = e_val['experts'][total_experts]['model']

                # predict the classes for test data
                pred = moe_model(images)
                pred_labels = torch.argmax(pred, dim=1)

                expert_outputs = moe_model.expert_outputs
                gate_outputs = moe_model.gate_outputs

                # get the experts selected by the gate for each sample
                pred_gate_labels = torch.argmax(gate_outputs, dim=1)

                # plot the expert selection table
                print('\nExperts used by the gate for classification of each digit')
                class_expert_table = np.asarray([[0] * num_classes]*total_experts)
                for label, expert in zip(labels, pred_gate_labels):
                    class_expert_table[expert,label] += 1
                sns.heatmap(class_expert_table, yticklabels=['E'+str(i) for i in range(1,total_experts+1)], 
                            xticklabels=[classes[i] for i in range(0, num_classes)],
                            annot=True, cmap=cmap, fmt='d', ax=ax1[0])

                sns.heatmap(confusion_matrix(labels.cpu(), pred_labels.cpu()), annot=True, ax=ax1[1], cmap=cmap, fmt='d')

                plt.show()

    model = models[-1]

    fig1,ax1 = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(12, 4))
    ax1.flatten()

    # Plotting for the model without reg
    for e_key, e_val in model.items():
        cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                moe_model = e_val['experts'][total_experts]['model']

                # predict the classes for test data
                pred = moe_model(images)
                pred_labels = torch.argmax(pred, dim=1)

                expert_outputs = moe_model.expert_outputs
                gate_outputs = moe_model.gate_outputs

                # get the experts selected by the gate for each sample
                pred_gate_labels = torch.argmax(gate_outputs, dim=1)

                # plot the expert selection table
                print('\nExperts used by the gate for classification of each digit')
                class_expert_table = np.asarray([[0] * num_classes]*total_experts)
                for label, expert in zip(labels, pred_gate_labels):
                    class_expert_table[expert,label] += 1
                sns.heatmap(class_expert_table, yticklabels=['E'+str(i) for i in range(1,total_experts+1)], 
                            xticklabels=[classes[i] for i in range(0, num_classes)],
                            annot=True, cmap=cmap, fmt='d', ax=ax1[0])

                sns.heatmap(confusion_matrix(labels.cpu(), pred_labels.cpu()), annot=True, ax=ax1[1], cmap=cmap, fmt='d')

                plt.show()

    # plot error rates
    fig2,ax2 = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(16,4))
    ax2 = ax2.flatten()

    fig3,ax3 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8,4))

    fig4,ax4 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8,4))
    
    fig5,ax5 = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8,4))

    for i, model in enumerate(models):
        for e_key, e_val in model.items():

            # plot training and validation error rates
            sns.lineplot(x=range(num_epochs), y=1-np.asarray(model[e_key]['experts'][total_experts]['history']['accuracy']), ax=ax2[0])
            sns.lineplot(x=range(num_epochs), y=1-np.asarray(model[e_key]['experts'][total_experts]['history']['val_accuracy']), ax=ax2[1])

            # plot training loss
            sns.lineplot(x=range(num_epochs), y=np.asarray(model[e_key]['experts'][total_experts]['history']['loss']), ax=ax3)

            # plot mutual information
            sns.lineplot(x=range(num_epochs), y=np.asarray(model[e_key]['experts'][total_experts]['history']['mutual_EY']), ax=ax4)

            # plot mutual information
            sns.lineplot(x=range(num_epochs), y=np.asarray(model[e_key]['experts'][total_experts]['history']['H_EY']), ax=ax5)

    legend = data[:,0]

    ax2[0].legend(legend)

    ax2[0].set_xlabel('epochs')
    ax2[0].set_xticks(range(num_epochs+1))
    ax2[0].set_ylabel('train error rate')
    ax2[0].set_ylim(ymin=0)


    ax2[1].legend(legend)

    ax2[1].set_xlabel('epochs')
    ax2[1].set_xticks(range(num_epochs+1))
    ax2[1].set_ylabel('validation error rate')
    ax2[1].set_ylim(ymin=0)


    ax3.legend(legend)

    ax3.set_xlabel('epochs')
    ax3.set_xticks(range(num_epochs+1))
    ax3.set_ylabel('training loss')
    ax3.set_ylim(ymin=0)

    ax4.legend(legend)

    ax4.set_xlabel('epochs')
    ax4.set_xticks(range(num_epochs+1))
    ax4.set_ylabel('mutual information')
    ax4.set_ylim(ymin=0)
    
    ax5.legend(legend)

    ax5.set_xlabel('epochs')
    ax5.set_xticks(range(num_epochs+1))
    ax5.set_ylabel('Entropy EY')
    ax5.set_ylim(ymin=0)

    plt.show()

import pandas as pd

def plot_gate_prob(model_name, temps=[1.0], w_importance_range=[0], w_sample_sim_same_range=[0], w_sample_sim_diff_range=[0], 
                   total_experts=5, num_classes=10, classes=None, num_epochs=20, 
                   testloader=None, caption=None, index=0, fig_path=None, model_path=None, device=torch.device("cpu")):
    
    m = model_name
    
    for T, w_importance, w_sample_sim_same, w_sample_sim_diff in product(temps, w_importance_range, w_sample_sim_same_range, w_sample_sim_same_range):
                    
        y_gate_prob = {} 
        y_gate_prob_T = {} 
        
        print('Temperature','{:.1f}'.format(T))
        print('Importance','{:.1f}'.format(w_importance))
        print('Sample sim same','{:.1f}'.format(w_sample_sim_same))
        print('Sample sim diff','{:.1f}'.format(w_sample_sim_diff))

        plot_file = generate_plot_file(m, temp=T, w_importance=w_importance, 
                                       w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff, 
                                       specific=str(num_classes)+'_'+str(total_experts)+'_models.pt')
        
        print('plot_file', plot_file)

        model = torch.load(open(os.path.join(model_path, plot_file),'rb'), map_location=device)[index]
        palette = sns.color_palette("Set2")
        fig,ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(18, 6))
        fontsize = 20
        label_fontsize = 16

        for e_key, e_val in model.items():
            history = model[e_key]['experts'][total_experts]['history']
            gate_probability = torch.vstack(history['gate_probabilities']).view(num_epochs, -1, total_experts)
            gate_probabilities_sum = torch.sum(gate_probability, dim=1).detach().cpu().numpy()
            
            if T > 1:
                gate_probability_T = torch.vstack((history['gate_probabilities_T'])).view(num_epochs, -1, total_experts)
                gate_probabilities_T_sum = torch.sum(gate_probability_T, dim=1).detach().cpu().numpy()
            
            labels = []
            for epoch in range(num_epochs):
                for e in range(total_experts):
                    y_gate_prob['Expert '+str(e+1)] = gate_probabilities_sum[:,e]
                    
                    if T > 1:
                        y_gate_prob_T['Expert'+str(e+1)] = gate_probabilities_T_sum[:,e]
                    
                    labels.append('E'+str(e))
            df = pd.DataFrame(y_gate_prob, index=list(range(1,num_epochs+1)))
            df.plot(kind='bar', stacked=True, color=palette, ax=ax[0])   
    
            ax[0].legend(loc='upper right') 
            ax[0].set_ylabel('Sum of gate probability \n distribution per expert\n  for all samples', fontsize=label_fontsize)
            ax[0].set_xlabel('Epochs', fontsize=label_fontsize)
            ax[0].set_title(caption, 
                            loc='center', wrap=True, fontsize=fontsize)
            ax[0].set_xticks(range(9,num_epochs+1,10))
            ax[0].tick_params(axis='both', which='major', labelsize=14)
            if T > 1:
                df_T = pd.DataFrame(y_gate_prob_T, index=list(range(1,num_epochs+1)))            
                df_T.plot(kind='bar', stacked=True, color=palette, ax=ax[1])
                ax[1].legend(loc='upper right') 
                ax[1].set_ylabel('Sum of gate probability \n distribution per expert \n for all samples', fontsize=label_fontsize)
                ax[1].set_xlabel('Epochs', fontsize=label_fontsize)
                ax[1].set_title('Gate probability distribution, during training, \n with high temperature (T='+'{:.1f}'.format(T)+')\n', loc='center', wrap=True, fontsize=fontsize)
                ax[1].set_xticks(range(9,num_epochs+1,10))
                ax[1].tick_params(axis='both', which='major', labelsize=14)
            else:
                ax[1].axis('off')
                
            plt.tight_layout()
            plot_file = generate_plot_file(m, temp=T, w_importance=w_importance, 
                                           w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff, 
                                           specific=str(num_classes)+'_'+str(total_experts)+'_barplot.png')
            if T>1:
                plt.savefig(os.path.join(fig_path, plot_file))
            else:
                extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                plt.savefig(os.path.join(fig_path, plot_file), bbox_inches=extent.expanded(1.3, 1.5))
            plt.show()


    
def plot_expert_predictions(m, test_loader, temps=[1.0], w_importance_range = [0.0], w_sample_sim_same_range = [0.0], w_sample_sim_diff_range = [0.0], total_experts = 5, num_classes = 10, 
                            classes=range(10), num_epochs = 20, index=0, fig_path=None, model_path=None):


    model, model_file = find_best_model(m, temps=temps, w_importance_range=w_importance_range,
                                        w_sample_sim_same_range=w_sample_sim_same_range, w_sample_sim_diff_range=w_sample_sim_diff_range, 
                                        num_classes=num_classes, total_experts=total_experts, num_epochs=num_epochs, model_path=model_path)
    print(model_file)


    for e_key, e_val in model.items():
        history = model[e_key]['experts'][total_experts]['history']
        gate_probabilities = history['gate_probabilities']

        cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

        expert_label = {i:[] for i in range(total_experts)}
        with torch.no_grad(): 
            for images, labels in test_loader:            

                images, labels = images.to(device), labels.to(device)
                moe_model = e_val['experts'][total_experts]['model']

                # predict the classes for test data
                pred = [p.item() for p in torch.argmax(moe_model(images), dim=1)]

                gate_outputs = moe_model.gate_outputs
                
                experts = [e.item() for e in torch.argmax(gate_outputs, dim=1)]
                
                sample_predictions = [['y_e'+str(i) for i in range(total_experts)]]
                sample_predictions[0] = sample_predictions[0]+['y\'','y','gate prob','E']
                
                for i in range(len(labels)):
                    entry = []
                    probs = []
                    for e in range(total_experts):
                        entry.append(torch.argmax(moe_model.expert_outputs[i,e,:]).item())
                        probs.append('{:.1f}'.format(moe_model.gate_outputs[i,e]))
                    entry.append(pred[i])
                    entry.append(labels[i].item())
                    entry.append(probs)
                    entry.append(experts[i])
                    sample_predictions.append(entry)
                    
                exp_class_prob = torch.zeros(total_experts, num_classes).to(device)
                for e in range(total_experts):
                    for index, l in enumerate(labels):
                        exp_class_prob[e,l] += gate_outputs[index,e]

                fig,ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(8,4))

                sns.heatmap(exp_class_prob.cpu().numpy().astype(int), yticklabels=['E'+str(i) for i in range(1,total_experts+1)], 
                                xticklabels=classes,
                                cmap=cmap, annot=True, fmt='d', ax=ax)
                plt.show()
                df = pd.DataFrame(sample_predictions[1:50], columns=sample_predictions[0])
                print(df)

