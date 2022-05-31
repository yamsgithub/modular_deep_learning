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

def kl_divergence(p, q):
    return torch.mean(torch.sum(p*torch.log(torch.div(p,q)), dim=1))

class cross_entropy_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(cross_entropy_loss, self).__init__()
        self.default_reduction = 'mean'
        self.criterion = nn.NLLLoss(reduction=reduction).to(device)

    def reduction(self, r='mean'):
        self.criterion.reduction = r
        
    def forward(self, outputs=None, expert_outputs=None, gate_outputs=None, targets=None):
        eps=1e-7
        logp = torch.log(outputs+eps).to(device)
        crossentropy_loss = self.criterion(logp, targets)
        return crossentropy_loss

def entropy(p):
    logp = torch.log2(p)
    with torch.no_grad():
        logp = np.nan_to_num(logp.cpu().numpy(), neginf=0)
        entropy_val = (-1*torch.sum(p.to(torch.device('cpu'))*logp,dim=len(p.shape)-1)).mean()
    return entropy_val

# Coefficient of variation
def cv(p):
    if p.shape[1] > 1:
        importance = torch.sum(p,dim=0)
        return torch.std(importance)/torch.mean(importance)

    return 0
    
def loss_importance(p, w_importance):
    loss_importance = 0.0
    if p.shape[1] > 1:
        importance = torch.sum(p,dim=0)
        loss_importance = w_importance * torch.square(torch.std(importance)/torch.mean(importance))
    return loss_importance

def mutual_information(count_mat):
    (r,c) = count_mat.shape
    joint_EY = np.zeros((r,c))
    N = np.sum(count_mat)
    for i in range(r):
        for j in range(c):
            joint_EY[i,j] = count_mat[i,j]/N
    marginal_Y = np.sum(joint_EY, axis=1)
    marginal_E = np.sum(joint_EY, axis=0)

    #(r,c) = joint_EY.shape
    #marginal_Y = np.sum(joint_EY, axis=1)
    #marginal_E = np.sum(joint_EY, axis=0)
    
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

    def __init__(self, hidden):
        super(attention,self).__init__()

        self.Wq = nn.Linear(hidden, hidden, bias=False)
        self.Wk = nn.Linear(hidden, hidden, bias=False)
        #self.Wv = nn.Linear(hidden, hidden, bias=False)
        
        self.hidden_size = hidden

    def forward(self, hidden_expert, hidden_gate):

        #print('hidden_expert', hidden_expert.shape)
        #print('hidden_gate', hidden_gate.shape)

        # Calculating Alignment Scores
        Q = self.Wq(hidden_gate)
        Q = Q.view(-1,1,Q.shape[1])
        K = self.Wk(hidden_expert)
        #V = self.Wv(hidden_expert)

        #print('Q', Q.shape)
        #print('K',K.shape)
        #print('V', V.shape)

        alignment_scores = Q @ torch.transpose(K,2,1)/self.hidden_size**0.5
        #print('alignment scores', alignment_scores.shape) 

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores.squeeze(), dim=1)
        #print('attn weights', attn_weights.shape)
        
        return attn_weights
        
        # Multiplying the Attention weights with encoder outputs to get the context vector
        #context_vector = torch.bmm(attn_weights, V).squeeze()
        #print('context_vector', context_vector.shape)

        #return context_vector
    

class attention_old(nn.Module):

    def __init__(self, num_experts, num_classes):
        super(attention,self).__init__()
        self.num_classes = num_classes

        self.Wq = nn.Linear(num_classes, num_classes, bias=False)
        self.Wk = nn.Linear(num_classes, num_classes, bias=False)
        self.Wv = nn.Parameter(torch.FloatTensor(num_classes, num_experts))

    def forward(self, expert_outputs):

        # Calculating Alignment Scores
        Q = self.Wq(expert_outputs)
        K = self.Wk(expert_outputs)

        #print('Q', Q.shape)
        #print('K',K.shape)
        #print('experts', expert_outputs.shape)
        #print('Wv',self.Wv.shape)

        V = expert_outputs @ self.Wv
        #print('V', V.shape)

        alignment_scores = Q @ torch.transpose(K,2,1)/ self.num_classes**0.5
        #print('alignment scores', alignment_scores)

        # Softmaxing alignment scores to get Attention weights
        attn_weights = F.softmax(alignment_scores, dim=1)

        # Multiplying the Attention weights with encoder outputs to get the context vector
        context_vector = torch.bmm(attn_weights, V)

        return context_vector




    

