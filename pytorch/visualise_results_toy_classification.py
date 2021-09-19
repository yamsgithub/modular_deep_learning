import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormaps 

import seaborn as sns
import numpy as np

import numpy as np

import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('device', device)
else:
    device = torch.device("cpu")
    print('device', device)

palette = ['tab:purple', 'tab:green', 'tab:orange','tab:pink', 'tab:olive','tab:brown', 'tab:cyan']

#create meshgrid
def generate_meshgrid(X):
    resolution = 100 # 100x100 background pixels
    a2d_min, a2d_max = np.min(X[:,0]), np.max(X[:,0])
    b2d_min, b2d_max = np.min(X[:,1]), np.max(X[:,1])
    a, b = np.meshgrid(np.linspace(a2d_min, a2d_max, resolution), 
                       np.linspace(b2d_min, b2d_max, resolution))
    generated_data = torch.tensor(np.c_[a.ravel(), b.ravel()], dtype=torch.float32)
    return generated_data

def labels(p, palette=palette):
    pred_labels = torch.argmax(p, dim=1)
    uniq_y = np.unique(pred_labels)
    pred_color = [palette[i] for i in uniq_y]
    return pred_color, pred_labels


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

def lighten_color(colors, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    l_colors = []
    for color in colors:
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
        l_color = colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])
        l_colors.append(l_color)
    return l_colors


def visualize(models, total_experts, num_classes, generated_data, 
              X_orig_data, y_orig_data, filename):
    keys = models.keys()
    print(keys)
    N = len(keys)

    fontsize = 14
    fontsize1 = 8
    labelsize = 11
    legendsize = 11
    
    index = 0
    visible = True
    for m_key, m_val in models.items():
        
        for num_experts in total_experts:
            
            nrows = 1
            ncols = num_experts + 1

            thefigsize = (ncols*6, nrows*6)
            fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=thefigsize)

            print('\n',m_key)
            moe_model = m_val['experts'][num_experts]['model'].to(device)

            pred_orig_data = moe_model(X_orig_data)

            orig_data_gate_outputs = moe_model.gate_outputs
            orig_data_expert_outputs = moe_model.expert_outputs
            if 'stochastic' in m_key:
                orig_data_samples = moe_model.samples

            pred_orig_data_color,pred_orig_data_labels = labels(pred_orig_data.cpu())

            pred_gen_data = moe_model(generated_data)
            pred_gen_data_expert_output = moe_model.expert_outputs

            pred_gen_data_color,pred_gen_data_labels = labels(pred_gen_data.cpu())

            palette_gate = sns.color_palette("Paired")+sns.color_palette('Set2')
            pred_gate_gen_data_output = moe_model.gate_outputs
            pred_gate_gen_data_color, pred_gate_gen_data_labels = labels(pred_gate_gen_data_output.cpu(), palette_gate)

            u_gate_labels = np.unique(pred_gate_gen_data_labels)
            sorted_labels = np.argsort(pred_gate_gen_data_labels)
            legend_labels = ['Expert '+str(l.item()+1) for l in pred_gate_gen_data_labels[sorted_labels]]
            sns.scatterplot(x=generated_data[:,0][sorted_labels].cpu(),y=generated_data[:,1][sorted_labels].cpu(),
                            hue=legend_labels, legend=visible, palette=pred_gate_gen_data_color, s=10, ax=ax[0])
            sns.scatterplot(x=X_orig_data[:,0].cpu() , y=X_orig_data[:,1].cpu() , 
                            hue=pred_orig_data_labels, palette=pred_orig_data_color, legend=visible, s=40,ax=ax[0])
            indices = np.where((pred_orig_data_labels == y_orig_data.cpu() )== False)[0]
            sns.scatterplot(x=X_orig_data[indices,0].cpu(),y=X_orig_data[indices,1].cpu(),
                            hue=['mis-classified']*len(indices), palette=['r'], marker='X', legend=visible, s=40, ax=ax[0])
            # Put a legend below current axis
            ax[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), fancybox=True, shadow=True,
                         ncol=len(u_gate_labels)+num_classes+1, fontsize=legendsize, markerscale=2.)
            ax[0].set_title('Mixture of Experts', fontsize=fontsize)
            ax[0].set_ylabel('Dim 2', fontsize=fontsize1)
            ax[0].set_xlabel('Dim 1', fontsize=fontsize1)
            ax[0].tick_params(labelsize=labelsize)

            for i in range(0, num_experts):
                if 'pre_softmax' in m_key:
                    pred_orig_data_expert_color,pred_orig_data_expert_labels = labels(F.softmax(orig_data_expert_outputs[:,i,:].cpu(), dim=1))
                else:
                    pred_orig_data_expert_color,pred_orig_data_expert_labels = labels(orig_data_expert_outputs[:,i,:].cpu())

                index = np.where(pred_gate_gen_data_labels==i)[0]
                color = np.unique(pred_gate_gen_data_labels[index].cpu())
                if len(color) > 0:
                    sns.scatterplot(x=generated_data[:,0][index].cpu(),y=generated_data[:,1][index].cpu(),
                                    hue=pred_gate_gen_data_labels[index]+1,palette=[palette_gate[0:num_experts][color[0]]],
                                    legend=False, ax=ax[i+1])

                sns.scatterplot(x=X_orig_data[:,0].cpu(),y=X_orig_data[:,1].cpu(),
                                hue=pred_orig_data_expert_labels,palette=pred_orig_data_expert_color, 
                                legend=False, s=40, ax=ax[i+1])

                pred_orig_data_expert_color,pred_orig_data_expert_labels = labels(F.softmax(orig_data_expert_outputs[:,i,:].cpu(), dim=1))
                indices = np.where((pred_orig_data_expert_labels == y_orig_data.cpu()) == False)[0]
                sns.scatterplot(x=X_orig_data[indices,0].cpu(),y=X_orig_data[indices,1].cpu(),
                                hue=['mc']*len(indices), palette=['r'], 
                                legend=False, marker='X', ax=ax[i+1])

                ax[i+1].set_title('Expert '+str(i+1)+' Model', fontsize=fontsize)
                ax[i+1].set_ylabel('Dim 2', fontsize=fontsize1)
                ax[i+1].set_xlabel('Dim 1', fontsize=fontsize1)
                ax[i+1].tick_params(labelsize=labelsize)

    #         captions = '(a)                                                  (b)'+\
    #                    '                                                   (c)' +\
    #                    '                                                   (d)'
    #         plt.figtext(0.2,0.00, captions, fontsize=20, va="top", ha="left")
            fig.savefig(filename +'_'+m_key+'_'+str(num_experts)+'_experts.png')
            plt.show()

def visualize_vertical(models, total_experts, num_classes, generated_data, 
              X_orig_data, y_orig_data, filename):
    keys = models.keys()
    print(keys)
    N = len(keys)
#     fontsize = 25-((N-1)*5)
#     fontsize1 = 20-((N-1)*3)
#     labelsize = 20-((N-1)*3)
#     legendsize = 18-((N-1)*3)

    fontsize = 14
    fontsize1 = 8
    labelsize = 11
    legendsize = 11
    
    nrows = total_experts +1
    ncols = N

    thefigsize = (ncols*6, nrows*6)
    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=thefigsize)
    ax = ax.flatten()

    index = 0
    visible = True
    for m_key, m_val in models.items():
        print('\n',m_key)
        moe_model = m_val['experts'][total_experts]['model']

        pred_orig_data = moe_model(X_orig_data)

        orig_data_gate_outputs = moe_model.gate_outputs
        orig_data_expert_outputs = moe_model.expert_outputs
        if 'stochastic' in m_key:
            orig_data_samples = moe_model.samples

        pred_orig_data_color,pred_orig_data_labels = labels(pred_orig_data.cpu())

        pred_gen_data = moe_model(generated_data)
        pred_gen_data_expert_output = moe_model.expert_outputs

        pred_gen_data_color,pred_gen_data_labels = labels(pred_gen_data.cpu())

        palette_gate = sns.color_palette("Paired")+sns.color_palette('Set2')
        pred_gate_gen_data_output = moe_model.gate_outputs
        pred_gate_gen_data_color, pred_gate_gen_data_labels = labels(pred_gate_gen_data_output.cpu(), palette_gate)
    
        u_gate_labels = np.unique(pred_gate_gen_data_labels)
        sorted_labels = np.argsort(pred_gate_gen_data_labels)
        legend_labels = ['Expert '+str(l.item()+1) for l in pred_gate_gen_data_labels[sorted_labels]]
        sns.scatterplot(x=generated_data[:,0][sorted_labels].cpu(),y=generated_data[:,1][sorted_labels].cpu(),
                        hue=legend_labels, legend=visible, palette=pred_gate_gen_data_color, s=10, ax=ax[index])
        sns.scatterplot(x=X_orig_data[:,0].cpu() , y=X_orig_data[:,1].cpu() , 
                        hue=pred_orig_data_labels, palette=pred_orig_data_color, legend=visible, s=40,ax=ax[index])
        indices = np.where((pred_orig_data_labels == y_orig_data.cpu() )== False)[0]
        sns.scatterplot(x=X_orig_data[indices,0].cpu(),y=X_orig_data[indices,1].cpu(),
                        hue=['mis-classified']*len(indices), palette=['r'], marker='X', legend=visible, s=40, ax=ax[index])
        ax[index].set_title(m_key+'\nMixture of Experts', fontsize=fontsize)
        ax[index].set_ylabel('Dim 2', fontsize=fontsize1)
        ax[index].set_xlabel('Dim 1', fontsize=fontsize1)
        ax[index].tick_params(labelsize=labelsize)
        
        visible = False

        for i in range(0, total_experts):
            if 'pre_softmax' in m_key:
                pred_orig_data_expert_color,pred_orig_data_expert_labels = labels(F.softmax(orig_data_expert_outputs[:,i,:].cpu(), dim=1))
            else:
                pred_orig_data_expert_color,pred_orig_data_expert_labels = labels(orig_data_expert_outputs[:,i,:].cpu())
            
            indicies = np.where(pred_gate_gen_data_labels==i)[0]
            color = np.unique(pred_gate_gen_data_labels[indicies].cpu())
            if len(color) > 0:
                sns.scatterplot(x=generated_data[:,0][indicies].cpu(),y=generated_data[:,1][indicies].cpu(),
                                hue=pred_gate_gen_data_labels[indicies]+1,palette=[palette_gate[0:total_experts][color[0]]],
                                legend=False, ax=ax[index+((i+1)*N)])

            sns.scatterplot(x=X_orig_data[:,0].cpu(),y=X_orig_data[:,1].cpu(),
                            hue=pred_orig_data_expert_labels,palette=pred_orig_data_expert_color, 
                            legend=False, s=40, ax=ax[index+((i+1)*N)])

            pred_orig_data_expert_color,pred_orig_data_expert_labels = labels(F.softmax(orig_data_expert_outputs[:,i,:].cpu(), dim=1))
            indices = np.where((pred_orig_data_expert_labels == y_orig_data.cpu()) == False)[0]
            sns.scatterplot(x=X_orig_data[indices,0].cpu(),y=X_orig_data[indices,1].cpu(),
                            hue=['mc']*len(indices), palette=['r'], 
                            legend=False, marker='X', ax=ax[index+((i+1)*N)])

            ax[index+((i+1)*N)].set_title('Expert '+str(i+1)+' Model', fontsize=fontsize)
            ax[index+((i+1)*N)].set_ylabel('Dim 2', fontsize=fontsize1)
            ax[index+((i+1)*N)].set_xlabel('Dim 1', fontsize=fontsize1)
            ax[index+((i+1)*N)].tick_params(labelsize=labelsize)
            
        index+=1
    #Put a legend below current axis
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.0, -3.7), fancybox=True, shadow=True,
                ncol=total_experts+num_classes+1, fontsize=legendsize, markerscale=2.)

#         captions = '(a)                                                  (b)'+\
#                    '                                                   (c)' +\
#                    '                                                   (d)'
#         plt.figtext(0.2,0.00, captions, fontsize=20, va="top", ha="left")
    fig.savefig(filename +'_'+str(total_experts)+'_experts.png')
    plt.show()

def visualize_decision_boundaries(models, total_experts, num_classes, generated_data, 
              X_orig_data, y_orig_data,filename):
    keys = models.keys()
    print(keys)
    N = len(keys)
    fontsize = 18
    fontsize1 = 11
    labelsize = 15
    legendsize = 15
    for e in range(total_experts, total_experts+1):
        
        print('Number of Experts:', e)
        
        nrows = 1
        ncols = N

        thefigsize = (ncols*6,1*6)
        fig_moe,ax = plt.subplots(1, ncols, sharex=False, sharey=True, figsize=thefigsize)
        
                      
        thefigsize3 = (ncols*4,1*4)
        fig_gate_prob,ax3 = plt.subplots(1, ncols, sharex=False, sharey=True, figsize=thefigsize3)       
             
        if ncols > 1:            
            ax = ax.flatten()
            ax3 = ax3.flatten()
        else:
            ax = [ax]
            ax3 = [ax3]
        
        index = 0
        visible = True
        l_visible = True
        for m_key, m_val in models.items():
            print('\n',m_key)
            moe_model = m_val['experts'][e]['model']
            
            pred_orig_data = moe_model(X_orig_data)
            
            orig_data_gate_outputs = moe_model.gate_outputs
            orig_data_expert_outputs = moe_model.expert_outputs
            if 'stochastic' in m_key:
                orig_data_samples = moe_model.samples
                
            pred_orig_data_color,pred_orig_data_labels = labels(pred_orig_data.cpu())

            pred_gen_data = moe_model(generated_data)
            pred_gen_data_expert_output = moe_model.expert_outputs
                
            pallette_gen_data = lighten_color(palette)
            pred_gen_data_color,pred_gen_data_labels = labels(pred_gen_data.cpu(), palette=pallette_gen_data)
            
            sns.scatterplot(x=generated_data[:,0].cpu(),y=generated_data[:,1].cpu(),
                            hue=pred_gen_data_labels.cpu(),palette=pred_gen_data_color, 
                            legend=False, s=10, ax=ax[index])
            sns.scatterplot(x=X_orig_data[:,0].cpu(), y=X_orig_data[:,1].cpu(), 
                            hue=y_orig_data.cpu(), palette=palette[0:num_classes], 
                            legend=visible, s=40,ax=ax[index])
            indices = np.where((pred_orig_data_labels == y_orig_data.cpu() )== False)[0]
            sns.scatterplot(x=X_orig_data[indices,0].cpu(),y=X_orig_data[indices,1].cpu(),
                            hue=['mis-classified']*len(indices), palette=['r'], 
                            legend=visible, marker='X', s=40, ax=ax[index])
            ax[index].set_title(m_key, fontsize=fontsize)
            ax[index].set_ylabel('Dim 2', fontsize=fontsize1)
            ax[index].set_xlabel('Dim 1', fontsize=fontsize1)
            ax[index].tick_params(labelsize=8)

            nrows = 1
            ncols = e
            
            thefigsize1 = (ncols*6,nrows*6)
            fig_experts,ax1 = plt.subplots(nrows, ncols, sharex=False, sharey=True, figsize=thefigsize1)
            fig_experts.subplots_adjust(hspace=0.25)
            ax1 = ax1.flatten()  
            e_l_visible = True
            for i in range(0, e):
                e_title = 'Expert '+str(i+1)+' Model'
                if 'pre_softmax' in m_key:
                    pred_gen_data_expert_color,pred_gen_data_expert_labels = labels(F.softmax(pred_gen_data_expert_output[:,i,:].cpu(), dim=1),
                                                                                   palette=pallette_gen_data)
                else:
                    pred_gen_data_expert_color,pred_gen_data_expert_labels = labels(pred_gen_data_expert_output[:,i,:].cpu(),
                                                                                   palette=pallette_gen_data)                               
                sorted_index = torch.argsort(pred_gen_data_expert_labels)
                sns.scatterplot(x=generated_data[:,0][sorted_index].cpu(),y=generated_data[:,1][sorted_index].cpu(),
                                hue=pred_gen_data_expert_labels[sorted_index],palette=pred_gen_data_expert_color, 
                                legend=False, s=10, ax=ax1[i])
                
                sns.scatterplot(x=X_orig_data[:,0].cpu(), y=X_orig_data[:,1].cpu(), hue=y_orig_data.cpu(), 
                                palette=palette[0:num_classes], legend=e_l_visible, s=40, ax=ax1[i])

                pred_orig_data_expert_color,pred_orig_data_expert_labels = labels(F.softmax(orig_data_expert_outputs[:,i,:].cpu(), dim=1))
                indices = np.where((pred_orig_data_expert_labels == y_orig_data.cpu()) == False)[0]
                sns.scatterplot(x=X_orig_data[indices,0].cpu(),y=X_orig_data[indices,1].cpu(),
                                hue=['mis-classified']*len(indices),legend=e_l_visible, palette=['r'], 
                                marker='X', s=40, ax=ax1[i])
                ax1[i].set_title(e_title, fontsize=14)
                ax1[i].set_ylabel('Dim 2', fontsize=fontsize1)
                ax1[i].set_xlabel('Dim 1', fontsize=fontsize1)
                ax1[i].tick_params(labelsize=8)
                e_l_visible = False
                
            ax1[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), fancybox=True, shadow=False,
                            ncol=num_classes+1, fontsize=10, markerscale=2.)
            fig_experts.savefig(filename+'_'+m_key+'_expert_outputs_'+str(e)+'_experts.png')
           
            history = m_val['experts'][e]['history']
            gate_prob = np.asarray(history['mean_gate_probability'])
            num_epochs = len(gate_prob)
            for i in range(0, total_experts):
                sns.lineplot(x=np.asarray(list(range(0,num_epochs))), y=gate_prob[:,i], 
                             marker='o', legend=l_visible, ax=ax3[index])
                l_visible = False

            ax3[index].set_title(m_key, fontsize=12)

            ax3[index].set_xlabel('epochs', fontsize=8)
            ax3[index].set_xlim(0,num_epochs)
            ax3[index].set_ylabel('avg gate probabilitis\nper expert', fontsize=8)
            ax3[index].set_ylim(0,1)
            ax3[index].tick_params(labelsize=5)
            
            visible = False
            index += 1

        ax[0].legend(loc='upper left', bbox_to_anchor=(0.0, -0.08), fancybox=True, shadow=False,
                        ncol=num_classes+1, fontsize=fontsize1)

        fig_moe.savefig(filename +'_'+str(e)+'_experts.png')
        fig_gate_prob.savefig(filename +'_gate_prob_'+str(e)+'_experts.png')
        plt.show()
    


