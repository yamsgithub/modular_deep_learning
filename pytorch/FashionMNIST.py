import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps 
                                        
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

import numpy as np
import seaborn as sns
import os

import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from moe_models import moe_stochastic_model, moe_stochastic_loss, cross_entropy_loss, moe_expectation_model, moe_pre_softmax_expectation_model

dataset = 'fashion_mnist_all_10000'

trainset = tv.datasets.FashionMNIST('data', transform=tv.transforms.ToTensor(), train=True)
trainset

testset = tv.datasets.FashionMNIST('data',transform=tv.transforms.ToTensor(), train=False)
testset

batchsize=128

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers=1, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=True, num_workers=1, pin_memory=True)


#Function to display the images
def plot_colour_images(images_to_plot, titles=None, nrows=None, ncols=6, thefigsize=(18,18), classes=10):
    # images_to_plot: list of images to be displayed
    # titles: list of titles corresponding to the images
    # ncols: The number of images per row to display. The number of rows 
    #        is computed from the number of images to display and the ncols
    # theFigsize: The size of the layour of all the displayed images
    
    n_images = images_to_plot.shape[0]
    
    # Compute the number of rows
    if nrows is None:
        nrows = np.ceil(n_images/ncols).astype(int)
    
    fig,ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=thefigsize)
    ax = ax.flatten()

    for i in range(n_images):
        ax[i].imshow( images_to_plot[i,:,:,:]) 
            # cmap=cm.Greys plots in Grey scale so the image looks as if it were written
        ax[i].axis('off')  
        if titles is not None and i<classes:
            ax[i].set_title(titles[i%classes])


# + jupyter={"outputs_hidden": true}
# Display 10 samples from each of the 10 classes in Fashion Mnist dataset
# classes = ['t-shirt', 'Trouser', 'Pullover','Dress','Coat','Sandal',
#            'Shirt','Sneaker','Bag','Ankle boot']
# images_to_plot = None
# i = 0
# for data , labels in trainloader:
#     #print(data.shape)
#     index = np.where(labels==i)[0]
#     if len(index) >= 10:
#         if i == 0:
#             images_to_plot = data[index[0:10],:,:]
#         else:
#             images_to_plot = np.vstack((images_to_plot, data[index[0:10],:,:]))
#         i += 1
# images_to_plot = images_to_plot.reshape(10,10,28, 28, 1).transpose(1,0,2,3,4).reshape(100,28,28,1)
# plot_colour_images(images_to_plot, nrows=10, ncols=10,thefigsize=(15,15), titles=classes)
# -

train_size = 10000
trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,list(range(0,train_size))), 
                                          batch_size=train_size,
                                          shuffle=False, num_workers=1)
test_size = 3000
testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset,list(range(0,test_size))),
                                         batch_size=test_size,
                                         shuffle=False, num_workers=1)


def colors(p,  palette=['y','g','r','c','b','tab:pink','tab:brown','tab:cyan','tab:olive','tab:purple']):
    uniq_y = np.unique(p)
    pred_color = [palette[i] for i in uniq_y]
    return pred_color


def pca_tsne_plot(X, y, classes, dataset, size, filter_class=False):
    transformer = IncrementalPCA(n_components=50, batch_size=128)
    palette = sns.color_palette('Set2') + sns.color_palette("Paired") 
    if filter_class:
        index = filter_classes(classes_sub, size)
    else:
        index = list(range(0,size))
    X_pca = transformer.fit_transform(X.reshape(X.shape[0],X.shape[-1]*X.shape[-1]))
    X_pca = X_pca[index]
    print('X:',X.shape)
    y = y[index]
    print(np.unique(y, return_counts=True))
    print('PCA:',X_pca.shape)
    print('y:', y.shape)
    X_embedded = TSNE(n_components=2,perplexity=50.0,).fit_transform(X_pca)
    print('TSNE:',X_embedded.shape)    
    fig,ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10,10))
    true_label_names = [classes[i] for i in y]
    true_color = colors(y, palette[0:len(classes)]) 
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=true_label_names, hue_order=classes, palette=true_color, ax=ax)
    plt.savefig('figures/fashion_mnist/'+dataset+'.png')
    return X_pca, X_embedded
    


classes = ['t-shirt', 'Trouser', 'Pullover','Dress','Coat','Sandal',
           'Shirt','Sneaker','Bag','Ankle boot']

size = 10000
for X, y in trainloader:
    X_pca, X_tsne = pca_tsne_plot(X, y, classes, dataset, size)


#Expert network
class expert_layers(nn.Module):
    def __init__(self, output):
        super(expert_layers, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)
        self.fc1 = nn.Linear(in_features=1*12*12, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=10)

        self.out = nn.Linear(in_features=10, out_features=output)


    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        #print('CONV T SHAPE', t.shape)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        #print('MAX POOL T SHAPE', t.shape)
        # fc1
        t = t.reshape(-1, 1*12*12)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = F.softmax(self.out(t), dim=1)
        
        return t


#Expert network
class expert_layers_presoftmax(nn.Module):
    def __init__(self, output):
        super(expert_layers_presoftmax, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=1*12*12, out_features=15)
        self.fc2 = nn.Linear(in_features=15, out_features=30)

        self.out = nn.Linear(in_features=30, out_features=output)


    def forward(self, t):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        print(t.shape)
        # fc1
        t = t.reshape(-1, 1*12*12)
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        
        return t


# create a set of experts
def experts(num_experts, expert_layers_type=expert_layers):
    models = []
    for i in range(num_experts):
        models.append(expert_layers_type(num_classes))
    return nn.ModuleList(models)


#Expert network
# class gate_layers(nn.Module):
#     def __init__(self, num_experts, num_classes, T=1.0):
#         super(gate_layers, self).__init__()
#         # define layers
#         self.fc1 = nn.Linear(in_features=28*28, out_features=15)
#         self.fc2 = nn.Linear(in_features=15, out_features=30)
#         self.out = nn.Linear(in_features=30, out_features=num_experts)
#         self.num_experts = num_experts
#         self.num_classes = num_classes
#         self.count = 0
#         self.T = T


#     def forward(self, t):
#         t = torch.flatten(t, start_dim=2).reshape(t.shape[0], t.shape[2]*t.shape[3])

#         # fc1      
#         t = self.fc1(t)
#         t = F.relu(t)

#         # fc2
#         t = self.fc2(t)
#         t = F.relu(t)

#         # output
#         t = self.out(t)
#         t = F.softmax(t/self.T, dim=1)
#         return t


#Expert network
class gate_layers(nn.Module):
    def __init__(self, num_experts, num_classes):
        super(gate_layers, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)

        self.fc1 = nn.Linear(in_features=1*12*12, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=num_experts)
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.count = 0

    def forward(self, t, T=1.0, y=None):
        # conv 1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # fc1
        t = t.reshape(-1, 1*12*12)
        if not y is None:
            t = torch.cat((t, torch.flatten(y, start_dim=1)), dim=1)
            self.fc1 = nn.Linear(in_features=1*12*12+(self.num_experts* self.num_classes), out_features=15)
            
        t = self.fc1(t)
        t = F.relu(t)

        # fc2
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)
        t = F.softmax(t/T, dim=1)
        return t



# compute
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


batchsize = 128

total_experts = 10

num_classes = 10

num_epochs = 20

train_size = 10000
trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,list(range(0,train_size))), 
                                          batch_size=batchsize,
                                          shuffle=True, num_workers=1)
test_size = 3000
testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset,list(range(0,test_size))),
                                         batch_size=test_size,
                                         shuffle=True, num_workers=1)

# +
# experiment with models with different number of experts
w_importance = 0.0
augment = False
attention = False
w_ortho = 0.0
w_ideal_gate = 0.0
T_max = 20
models = {#'moe_stochastic_model':{'model':moe_stochastic_model, 'loss':moe_stochastic_loss,'experts':{}}, 
          'moe_expectation_model':{'model':moe_expectation_model,'loss':cross_entropy_loss,'experts':{}}, 
          #'moe_pre_softmax_expectation_model':{'model':moe_pre_softmax_expectation_model,'loss':cross_entropy_loss,'experts':{}}
         }
for T in range(1, T_max+1):
    for key, val in models.items():
        print('Model:', key)
        for num_experts in range(total_experts, total_experts+1):
            print('Number of experts ', num_experts)
            if 'pre_softmax' in key:
                expert_models = experts(num_experts, expert_layers_presoftmax)
            else:
                expert_models = experts(num_experts)
            gate_model = gate_layers(num_experts, num_classes)
            moe_model = val['model'](num_experts, num_classes, augment, attention, expert_models, gate_model)
            params = [p.numel() for p in moe_model.parameters() if p.requires_grad]
            print('model params:', sum(params))
            optimizer_moe = optim.RMSprop(moe_model.parameters(),
                                      lr=0.001, momentum=0.9)
            optimizer_gate = optim.RMSprop(gate_model.parameters(),
                                      lr=0.001, momentum=0.9)
            params = []
            for i, expert in enumerate(expert_models):
                params.append({'params':expert.parameters()})
            optimizer_experts = optim.RMSprop(params, lr=0.001, momentum=0.9)
            hist = moe_model.train(trainloader, testloader,  val['loss'], optimizer_moe, optimizer_gate, optimizer_experts,  
                                   w_importance, w_ortho, w_ideal_gate, T=T, accuracy=accuracy, epochs=num_epochs)
            val['experts'][num_experts] = {'model':moe_model, 'history':hist}
            torch.save(models,open(os.path.join('results/', dataset+'_T_'+str(T)+'_'+str(num_classes)+'_model.pt'),'wb'))

    models = {'moe_expectation_model':{'model':moe_expectation_model,'loss':cross_entropy_loss,'experts':{}}}

for T in range(1, T_max+1):
    models = torch.load(os.path.join('results/', dataset+'_T_'+str(T)+'_'+str(num_classes)+'_model.pt'))
    for m_key, m_val in models.items():
        for i in range(total_experts, total_experts+1):
            history = m_val['experts'][i]['history']
            plt.plot(range(len(history['exp_batch'])), history['exp_batch'], marker='o')

        legend_labels = ['E'+str(i) for i in range(1, num_experts+1)]
        plt.title('Samples per expert  for ' + str(i) + ' Experts and T = '+str(T))
        plt.legend(legend_labels)
        plt.xlabel('Number of epochs')
        plt.xlim(0, num_epochs)
        plt.ylabel('Samples')
        plt.savefig(os.path.join('figures', 'samples_'+dataset+'_T_'+str(T)+'_'+str(num_classes)+'_'+str(i)+'_experts.png'))
        plt.show()
    

val_accuracies = []

legend_labels = []
colors = ['b','g','r','c', 'm', 'k','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
          'tab:pink','tab:gray','tab:olive','tab:cyan']
start = 0

for T in range(1, T_max+1):
    models = torch.load(os.path.join('results/', dataset+'_T_'+str(T)+'_'+str(num_classes)+'_model.pt'))
    for m_key, m_val in models.items():
        for i in range(total_experts, total_experts+1):
            history = m_val['experts'][i]['history']
            val_accuracies.append((1-np.asarray(history['val_accuracy']))[start:])

    legend_labels.append('T'+str(T))
val_accuracies = np.asarray(val_accuracies)
print(val_accuracies.shape)

plt.rcParams["figure.figsize"] = (30,20)

for i in range(len(val_accuracies)):
    if i > 5:
        marker = '*'
        markersize = 20
    else:
        marker = 'o'
        markersize = 12
    plt.plot(range(start, num_epochs), val_accuracies[i], color=colors[i], marker=marker, markersize=markersize)

plt.title('Validation accuracy  for ' + str(i) + ' Experts and T = 1:20', fontsize=30)
plt.legend(legend_labels, prop={'size': 20})
plt.xlabel('Number of epochs', fontsize=25)
plt.xlim(start, num_epochs)
plt.ylabel('Error rate',  fontsize=25)
plt.tick_params(labelsize=20)
plt.ion()
plt.isinteractive
plt.savefig(os.path.join('figures', 'val_accuracy_'+dataset+'_'+str(num_classes)+'_'+str(total_experts)+'_part_experts.png'))
plt.show()
    

# -

def get_labels(p):
    pred_labels = torch.argmax(p, dim=1)
    return pred_labels


from sklearn.metrics import confusion_matrix
import seaborn as sns
def predict(dataloader, model):
        
        pred_labels = []
        true_labels = []
        for i, data in enumerate(dataloader):
            inputs, labels = data
            true_labels.append(labels)
            pred_labels.append(torch.argmax(model(inputs), dim=1))
            
        return torch.stack(true_labels), torch.stack(pred_labels)


train_size = 10000
trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,list(range(0,train_size))), 
                                          batch_size=train_size,
                                          shuffle=False, num_workers=1)
test_size = 3000
testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset,list(range(0,test_size))),
                                         batch_size=test_size,
                                         shuffle=False, num_workers=1)

for T in range(1, T_max+1):
    models = torch.load(os.path.join('results/', dataset+'_T_'+str(T)+'_'+str(num_classes)+'_model.pt'))

    keys = models.keys()
    print(keys)
    N = len(keys)

    palette = sns.color_palette('Set2') + sns.color_palette("Paired")

    fig,ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10,10))

    for X, y in trainloader:
        true_label_names = [classes[i] for i in y]
        true_color = colors(y, palette[0:num_classes]) 
        sns.scatterplot(x=X_tsne[:,0],y=X_tsne[:,1], 
                        hue=true_label_names, hue_order=classes, palette=true_color,ax=ax)

    for e in range(total_experts, total_experts+1):
        nrows = 2+e
        ncols = N
        thefigsize = (ncols*10,nrows*10)

        print('Number of Experts:', e)

        fig,ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=thefigsize)
        ax = ax.flatten()


        index = 0
        for X, y in trainloader:
            for m_key, m_val in models.items():

                moe_model = m_val['experts'][e]['model']

                print(len(palette))
                pred = moe_model(X)
                pred_labels = get_labels(pred)
                pred_label_names = [classes[i] for i in pred_labels]
                pred_color = colors(pred_labels, palette[0:num_classes])
                pred_classes = [classes[i] for i in np.unique(pred_labels)]

                true_label_names = [classes[i] for i in y]
                true_color = colors(y, palette[0:num_classes]) 
                #sns.scatterplot(x=X_tsne[:,0],y=X_tsne[:,1], 
                #                hue=true_label_names, hue_order=classes, palette=true_color,legend=False, s=50, alpha=0.6,ax=ax[index])

                sns.scatterplot(x=X_tsne[:,0],y=X_tsne[:,1],
                                hue=pred_label_names,hue_order=pred_classes,palette=pred_color, ax=ax[index])
                indices = np.where((pred_labels == y) == False)[0]
                sns.scatterplot(x=X_tsne[indices,0],y=X_tsne[indices,1],
                                hue=['misclassified']*len(indices), palette=['r'], marker='X', ax=ax[index])

                ax[index].set_title('Mixture of Experts')
                ax[index].set_ylabel('Dim 2')
                ax[index].set_xlabel('Dim 1')


                print(moe_model.expert_outputs.shape)
                for i in range(e):
                    pred_expert_labels = get_labels(moe_model.expert_outputs[:,i,:])

                    indices = np.where((pred_expert_labels == y) == True)[0]

                    pred_expert_label_names = [classes[i] for i in pred_expert_labels[indices]]
                    pred_expert_color = colors(pred_expert_labels[indices], palette[0:num_classes])
                    pred_expert_classes = [classes[i] for i in np.unique(pred_expert_labels[indices])]
                    sns.scatterplot(x=X_tsne[indices,0],y=X_tsne[indices,1],
                                hue=pred_expert_label_names,hue_order=pred_expert_classes,palette=pred_expert_color,marker='X', s=20,alpha=1.0, ax=ax[N*(i+1)+index])
                    indices = np.where((pred_expert_labels == y) == False)[0]
                    sns.scatterplot(x=X_tsne[indices,0],y=X_tsne[indices,1],
                                hue=['misclassified']*len(indices), palette=['r'], marker='X', ax=ax[N*(i+1)+index])

                    ax[N*(i+1)+index].set_title('Expert '+str(i+1))
                    ax[N*(i+1)+index].set_ylabel('Dim 2')
                    ax[N*(i+1)+index].set_xlabel('Dim 1')

                pred_gate = moe_model.gate_outputs
                pred_gate_labels = get_labels(pred_gate)+1
                pred_gate_color = colors(pred_gate_labels, palette)

                sns.scatterplot(x=X_tsne[:,0],y=X_tsne[:,1],
                                hue=pred_gate_labels,hue_order=np.unique(pred_gate_labels),palette=pred_gate_color, ax=ax[N*(e+1)+index])       

                index += 1
        plt.savefig('figures/'+dataset+'_T_'+str(T)+'_'+str(num_classes)+'_'+str(e)+'_experts.png')
        plt.show()
    

# -

# + jupyter={"outputs_hidden": true}
# for i in range(1, total_experts+1):
#     legend_labels = []
#     for m_key, m_val in models.items():
#         print('Mutual information for ', m_key)
#         history = m_val['experts'][i]['history']
# #         print('EY', history['EY'][-1], '\n\n')
# #         print('Mutual_EY', history['mutual_EY'], '\n\n')
# #         print('H_EY', history['H_EY'], '\n\n')
# #         print('H_E', history['H_E'], '\n\n')
# #         print('H_Y', history['H_Y'], '\n\n')

#         plt.plot(range(len(history['mutual_EY'])), history['mutual_EY'])
#         plt.plot(range(len(history['H_EY'])), history['H_EY'],marker='X')
#         plt.plot(range(len(history['H_Y'])), 
#                  [history['H_E'][i] + history['H_Y'][i] for i in range(len(history['H_Y']))], marker='o')
        
#         legend_labels = ['MI_EY']
#         legend_labels += ['H_EY']
#         legend_labels += ['H_E+H_Y']
                 
#         plt.title('Mutual information of gate and classes: ' + str(i) + ' Experts')
#         plt.legend(legend_labels)
#         plt.xlabel('Number of epochs')
#         plt.xlim(0, num_epochs)
#         plt.ylabel('Mutual Information')
#         #plt.ylim(0, 1)
#         plt.savefig('figures/all/mutual_information_'+str(num_classes)+'_'+str(i)+'_experts.png')
#         plt.show()



# for i in range(1, total_experts+1):
    
#     legend_labels = []
#     for m_key, m_val in models.items():
#         history = m_val['experts'][i]['history']
#         plt.plot(range(len(history['entropy'])), history['entropy'])
#         legend_labels.append(m_key)
#     plt.title('Average entropy: ' + str(i) + ' Experts')
#     plt.legend(legend_labels)
#     plt.xlabel('Number of epochs')
#     plt.xlim(1, num_epochs+1)
#     plt.xticks(np.arange(1, num_epochs+1, step=1))
#     plt.ylabel('Avg Entropy')
#     #plt.ylim(0, 1)
#     plt.savefig('figures/all/entropy_'+dataset+'_'+ str(num_classes)+'_'+str(i)+'_experts.png')
#     plt.show()

# for i in range(1, total_experts+1):
    
#     legend_labels = []
#     for m_key, m_val in models.items():
#         history = m_val['experts'][i]['history']
#         plt.plot(range(len(history['loss_importance'])), history['loss_importance'])
#         legend_labels.append(m_key)
#     plt.title('Loss Importance: ' + str(i) + ' Experts')
#     plt.legend(legend_labels)
#     plt.xlabel('Number of epochs')
#     plt.xlim(1, num_epochs+1)
#     plt.xticks(np.arange(1, num_epochs+1, step=1))
#     plt.ylabel('Loss')
# #     plt.ylim(0, 1)
#     plt.savefig('figures/all/loss_importance_'+dataset+'_'+ str(num_classes)+'_'+str(i)+'_experts.png')
#     plt.show()

# for i in range(1, total_experts+1):
    
#     legend_labels = []
#     for m_key, m_val in models.items():
#         history = m_val['experts'][i]['history']
#         plt.plot(range(len(history['loss'])), history['loss'])
#         legend_labels.append(m_key)
#     plt.title('Loss: ' + str(i) + ' Experts')
#     plt.legend(legend_labels)
#     plt.xlabel('Number of epochs')
#     plt.xlim(1, num_epochs+1)
#     plt.xticks(np.arange(1, num_epochs+1, step=1))
#     plt.ylabel('Loss')
# #     plt.ylim(0, 1)
#     plt.savefig('figures/all/loss_'+dataset+'_'+ str(num_classes)+'_'+str(i)+'_experts.png')
#     plt.show()

# for i in range(1, total_experts+1):
#     legend_labels = []
#     for m_key, m_val in models.items():
#         history = m_val['experts'][i]['history']
#         plt.plot(range(len(history['accuracy'])), 1-np.asarray(history['accuracy']))
#         legend_labels.append(m_key)
#     plt.title('Error rate: ' + str(i) + ' Experts')
#     plt.legend(legend_labels)
#     plt.xlabel('Number of epochs')
#     plt.xlim(1, num_epochs+1)
#     plt.xticks(np.arange(1, num_epochs+1, step=1))
#     plt.ylabel('Training Error Rate')
# #     plt.ylim(0, 1)
#     plt.savefig('figures/all/accuracy_'+dataset+'_'+str(num_classes)+'_'+str(i)+'_experts.png')
#     plt.show()

# for i in range(1, total_experts+1):
#     legend_labels = []
#     for m_key, m_val in models.items():
#         history = m_val['experts'][i]['history']
#         plt.plot(range(len(history['val_accuracy'])), 1-np.asarray(history['val_accuracy']))
#         legend_labels.append(m_key)
#     plt.title('Error rate: ' + str(i) + ' Experts')
#     plt.legend(legend_labels)
#     plt.xlabel('Number of epochs')
#     plt.xlim(1, num_epochs+1)
#     plt.xticks(np.arange(1, num_epochs+1, step=1))
#     plt.savefig('figures/all/val_accuracy_'+dataset+'_'+str(num_classes)+'_'+str(i)+'_experts.png')
#     plt.show()

# num_classes


# #Expert network
# class single_model_layers(nn.Module):
#     def __init__(self, output):
#         super(single_model_layers, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

#         self.fc1 = nn.Linear(in_features=1*13*13, out_features=5)
#         self.fc2 = nn.Linear(in_features=5, out_features=10)
#         self.out = nn.Linear(in_features=10, out_features=num_classes)
#         self.count = 0

#     def forward(self, t):
#         # conv 1
#         t = self.conv1(t)
#         t = F.relu(t)
#         t = F.max_pool2d(t, kernel_size=2, stride=2)

#         # fc1
#         t = t.reshape(-1, 1*13*13)
#         t = self.fc1(t)
#         t = F.relu(t)

#         # fc2
#         t = self.fc2(t)
#         t = F.relu(t)

#         # output
#         t = F.softmax(self.out(t), dim=1)
#         return t



# #Expert network
# class single_model_layers(nn.Module):
#     def __init__(self, output):
#         super(single_model_layers, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5)

#         self.fc1 = nn.Linear(in_features=1*12*12, out_features=15)
#         self.fc2 = nn.Linear(in_features=15, out_features=30)
#         self.out = nn.Linear(in_features=30, out_features=num_classes)
#         self.count = 0

#     def forward(self, t):
#         # conv 1
#         t = self.conv1(t)
#         t = F.relu(t)
#         t = F.max_pool2d(t, kernel_size=2, stride=2)

#         # fc1
#         t = t.reshape(-1, 1*12*12)
#         t = self.fc1(t)
#         t = F.relu(t)

#         # fc2
#         t = self.fc2(t)
#         t = F.relu(t)

#         # output
#         t = F.softmax(self.out(t), dim=1)
#         return t



# single_model = single_model_layers(num_classes)

# params = [p.numel() for p in single_model.parameters() if p.requires_grad]
# print('model params:', sum(params))

# single_model_optimizer = optim.RMSprop(single_model.parameters(),
#                                       lr=0.001, momentum=0.9)

# train_size = 10000
# trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,list(range(0,train_size))), 
#                                           batch_size=batchsize,
#                                           shuffle=False, num_workers=1)
# test_size = 3000
# testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset,list(range(0,test_size))),
#                                          batch_size=test_size,
#                                          shuffle=False, num_workers=1)

# # +
# single_model_history = {'loss':[], 'accuracy':[], 'val_accuracy':[]}
# for epoch in range(0, num_epochs):
#     running_loss = 0.0
#     training_accuracy = 0.0
#     test_accuracy = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         X, y = data

#         # zero the parameter gradients
#         single_model_optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = single_model(X)
#         loss = cross_entropy_loss(outputs, y)
#         loss.backward()
#         single_model_optimizer.step()
        
#         running_loss += loss.item()
#         training_accuracy += accuracy(outputs, y)
        
#     for j, test_data in enumerate(testloader, 0):
#         test_input, test_labels = test_data
#         test_outputs = single_model(test_input)
#         test_accuracy += accuracy(test_outputs, test_labels)
#     single_model_history['loss'].append(running_loss/(i+1))
#     single_model_history['accuracy'].append(training_accuracy/(i+1))
#     single_model_history['val_accuracy'].append(test_accuracy/(j+1))
#     print('epoch: %d loss: %.2f training accuracy: %.2f val accuracy: %.2f' %
#             (epoch + 1, running_loss / (i+1), training_accuracy/(i+1), test_accuracy/(j+1)))

# print('Finished Training')
# # -

# train_size = 10000
# trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,list(range(0,train_size))), 
#                                           batch_size=train_size,
#                                           shuffle=False, num_workers=1)
# test_size = 3000
# testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset,list(range(0,test_size))),
#                                          batch_size=test_size,
#                                          shuffle=False, num_workers=1)

# fig,ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(8,8))
# palette = sns.color_palette('Set2') + sns.color_palette("Paired")
# for X, y in trainloader:
#     pred = single_model(X)
#     pred_labels = get_labels(pred)
#     pred_label_names = [classes[i] for i in pred_labels]
#     pred_color = colors(pred_labels, palette[0:num_classes])
#     pred_classes = [classes[i] for i in np.unique(pred_labels)]
        
#     sns.scatterplot(x=X_tsne[:,0],y=X_tsne[:,1],
#                             hue=pred_label_names,hue_order=pred_classes,palette=pred_color, ax=ax)
#     indices = np.where((pred_labels == y) == False)[0]
#     sns.scatterplot(x=X_tsne[indices,0],y=X_tsne[indices,1],
#                             hue=['misclassified']*len(indices), palette=['r'], marker='X', ax=ax)

#     #sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette=palette, ax=ax)
#     ax.set_title('Single Model')
#     ax.set_ylabel('Dim 2')
#     ax.set_xlabel('Dim 1')
#     ax.legend(classes)


