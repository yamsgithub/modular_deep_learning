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

class stochastic_loss(nn.Module):

    def __init__(self, loss_criterion):
        super(stochastic_loss, self).__init__()
        self.default_reduction = 'none'
        self.loss_criterion = loss_criterion(reduction='none')

    def reduction(self, r='none'):
        self.loss_criterion.reduction(r)

    def forward(self, outputs, expert_outputs, gate_outputs, target):
        expert_outputs = torch.transpose(expert_outputs, 0,1)
        expert_loss = []
        for i in range(expert_outputs.shape[0]):
            loss = self.loss_criterion(expert_outputs[i], None, None, target)
            if len(loss.shape) > 1:
                loss = torch.mean(loss, dim=1)
            expert_loss.append(torch.exp(-0.5*loss))
        expert_loss = torch.stack(expert_loss)
        expert_loss.transpose_(0,1)
        expected_loss = -1*torch.log(torch.sum(gate_outputs * expert_loss, 1))
        total_loss = torch.mean(expected_loss)
        return total_loss.to(device)   


def entropy(p, reduction='mean'):
    logp = torch.log2(p)
    with torch.no_grad():
        logp = np.nan_to_num(logp.cpu().numpy(), neginf=0)
        entropy_val = (-1*torch.sum(p.to(torch.device('cpu'))*logp,dim=len(p.shape)-1))
        if reduction == 'mean':
            entropy_val = entropy_val.mean()
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
    joint_EY = torch.zeros((r,c), device=device)
    N = torch.sum(count_mat)
    for i in range(r):
        for j in range(c):
            joint_EY[i,j] = count_mat[i,j]/N
    marginal_Y = torch.sum(joint_EY, dim=1)
    marginal_E = torch.sum(joint_EY, dim=0)

    log2_marginal_Y = torch.log2(marginal_Y)
    mask_nan = torch.isnan(log2_marginal_Y)
    mask_inf = torch.isinf(log2_marginal_Y)
    log2_marginal_Y.masked_fill_(mask_nan, 0.0)
    log2_marginal_Y.masked_fill_(mask_inf, 0.0)

    H_Y = 0.0
    for i in range(r):
        H_Y += marginal_Y[i]*log2_marginal_Y[i]
    H_Y = -1*H_Y    

    log2_marginal_E = torch.log2(marginal_E)
    mask_nan = torch.isnan(log2_marginal_E)
    mask_inf = torch.isinf(log2_marginal_E)
    log2_marginal_E.masked_fill_(mask_nan, 0.0)
    log2_marginal_E.masked_fill_(mask_inf, 0.0)
    
    H_E = 0.0
    for i in range(c):
        H_E += marginal_E[i]*log2_marginal_E[i]
    H_E = -1*H_E   

    H_EY = 0.0
    log2_joint_EY = torch.log2(joint_EY)
    mask_nan = torch.isnan(log2_joint_EY)
    mask_inf = torch.isinf(log2_joint_EY)
    log2_joint_EY.masked_fill_(mask_nan, 0.0)
    log2_joint_EY.masked_fill_(mask_inf, 0.0)

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




    

