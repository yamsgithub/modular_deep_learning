import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps 
import seaborn as sns

import numpy as np

import torch

# ### Visualise decision boundaries of mixture of expert model, expert model and gate model

def plot_data(X, y, num_classes, save_as):
    f, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,8))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple'][0:num_classes]
    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y,palette=colors, ax=ax)
    ax.set_title("2D 3 classes Generated Data")
    plt.ylabel('Dim 2')
    plt.xlabel('Dim 1')
    plt.savefig(save_as)
    #plt.show()
    plt.clf()
    
def create_meshgrid(X):
    #create meshgrid
    resolution = 100 # 100x100 background pixels
    a2d_min, a2d_max = np.min(X[:,0]), np.max(X[:,0])
    b2d_min, b2d_max = np.min(X[:,1]), np.max(X[:,1])
    a, b = np.meshgrid(np.linspace(a2d_min, a2d_max, resolution), 
                       np.linspace(b2d_min, b2d_max, resolution))
    generated_data = torch.tensor(np.c_[a.ravel(), b.ravel()], dtype=torch.float32)

    return generated_data

def labels(p, palette=['r','c','y','g']):
    pred_labels = torch.argmax(p, dim=1)+1
    uniq_y = np.unique(pred_labels)
    pred_color = [palette[i-1] for i in uniq_y]
    return pred_color, pred_labels


def predict(dataloader, model):
        
        pred_labels = []
        true_labels = []
        for i, data in enumerate(dataloader):
            inputs, labels = data
            true_labels.append(labels)
            pred_labels.append(torch.argmax(model(inputs), dim=1))
            
        return torch.stack(true_labels), torch.stack(pred_labels)

def plot_results(X, y, num_classes, trainset, trainloader, testset, testloader, models, dataset, total_experts):

    generated_data = create_meshgrid(X)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
    
    for e in range(1, total_experts+1):
        nrows = (e*1)+3
        ncols = 3
        thefigsize = (ncols*5,nrows*5)
        
        print('Number of Experts:', e)
        
        fig,ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=thefigsize)
        ax = ax.flatten()
    
        keys = models.keys()
        print(keys)

        index = 0
        for m_key, m_val in models.items():

            if m_key == 'single_model':
                continue
        
            moe_model = m_val['experts'][e]['model']
            
            pred = moe_model(generated_data)
            pred_color,pred_labels = labels(pred)
            sns.scatterplot(x=generated_data[:,0],y=generated_data[:,1],
                            hue=pred_labels,palette=pred_color, legend=False, ax=ax[index])
            sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette=colors[0:num_classes], ax=ax[index])
            ax[index].set_title('Mixture of Experts')
            ax[index].set_ylabel('Dim 2')
            ax[index].set_xlabel('Dim 1')
            
        
            experts = moe_model.experts

            for i in range(0, e):
                pred = experts[i](generated_data)
                pred_color,pred_labels = labels(pred)
                sns.scatterplot(x=generated_data[:,0],y=generated_data[:,1],
                                hue=pred_labels,palette=pred_color, legend=False, ax=ax[((i+1)*3)+index])
                sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette=colors[0:num_classes],  ax=ax[((i+1)*3)+index])
                
                ax[((i+1)*3)+index].set_title('Expert '+str(i+1)+' Model')
                ax[((i+1)*3)+index].set_ylabel('Dim 2')
                ax[((i+1)*3)+index].set_xlabel('Dim 1')

            palette = sns.husl_palette(total_experts)
            pred_gate = moe_model.gate(generated_data)
            pred_gate_color, pred_gate_labels = labels(pred_gate, palette)
            
            sns.scatterplot(x=generated_data[:,0],y=generated_data[:,1],
                            hue=pred_gate_labels,palette=pred_gate_color, legend=False, ax=ax[((e+1)*3)+index])
            sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette=colors[0:num_classes], ax=ax[((e+1)*3)+index])
            ax[((e+1)*3)+index].set_title('Gate Model')
            ax[((e+1)*3)+index].set_ylabel('Dim 2')
            ax[((e+1)*3)+index].set_xlabel('Dim 1')
        
        
            pred_gate = moe_model.gate(trainset[:][0])
            pred_gate_color, pred_gate_labels = labels(pred_gate, palette)
            
            sns.scatterplot(x=trainset[:][0][:,0],y=trainset[:][0][:,1],
                            hue=pred_gate_labels,palette=pred_gate_color, ax=ax[((e+2)*3)+index])       
            
            
            index += 1
        plt.savefig('figures/all/'+dataset+'_'+str(num_classes)+'_'+str(e)+'_experts.png')
        #plt.show()
        plt.clf()

def plot_accuracy(models, total_experts, save_as):
    labels = []
    plt.figure(figsize=(20,10))
    for m_key, m_val in models.items():
        labels.append(m_key)
        accuracies = []
        for i in range(1, total_experts+1):                
            history = m_val['experts'][i]['history']
            accuracies.append(history['accuracy'][-1])
        plt.plot(range(1,len(accuracies)+1), accuracies)
    plt.legend(labels)
    plt.ylim(0, 1)
    plt.xticks(range(1, total_experts+1), [str(i) for i in range(1, total_experts+1)])
    plt.xlabel('Number of Experts')
    plt.ylabel('Accuracy')
    plt.savefig(save_as)
    #plt.show()
    plt.clf()


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


