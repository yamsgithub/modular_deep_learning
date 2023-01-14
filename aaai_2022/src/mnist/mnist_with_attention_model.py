import argparse 

from mnist_original_moe_training import *
from moe_with_attention_training import *

expert_layers_types = {'expert_layers': expert_layers}

gate_layers_types = {'gate_layers': gate_layers,
                    'gate_attn_layers': gate_attn_layers}

expert_layers_type = expert_layers
gate_layers_type = gate_attn_layers

m = 'mnist_with_attention'
mt = 'moe_expectation_model'
total_experts = 10
k = 0
num_epochs=20
hidden = 32
runs = 1
w_importance_range=[0.0]
w_sample_sim_same_range = [0.0]
w_sample_sim_diff_range = [0.0]

parser = argparse.ArgumentParser()
parser.add_argument('-e', help='expert layers type')
parser.add_argument('-g', help='gate layers type')
parser.add_argument('-k', help='top k')
parser.add_argument('-m', help='model name')
parser.add_argument('-mt', help='model type')
parser.add_argument('-r', help='number of runs')
parser.add_argument('-E', help='number of epochs')
parser.add_argument('-M', help='number of experts')
parser.add_argument('-H', help='Hidden layers')
parser.add_argument('-i', help='Importance factor')
parser.add_argument('-ss', help='sample similarity factor')
parser.add_argument('-sd', help='sample dissimilarity factor')
args = vars(parser.parse_args())
print('args', args)

if not args['e'] is None:
    expert_layers_type = expert_layers_types[args['e']] 
if not args['g'] is None:
    gate_layers_type = gate_layers_types[args['g']]    
if not args['k'] is None:
    k = int(args['k'])
if not args['m'] is None:
    m = args['m']
if not args['mt'] is None:
    mt = args['mt']
if not args['r'] is None:
    runs = int(args['r'])
if not args['M'] is None:
    total_experts = int(args['M'])
if not args['E'] is None:
    num_epochs = int(args['E'])
if not args['i'] is None:
    w_importance_range = [float(args['i'])]
if not args['ss'] is None:
    w_sample_sim_same_range = [float(args['ss'])]
if not args['sd'] is None:
    w_sample_sim_diff_range = [float(args['sd'])]


print('expert layers type:', expert_layers_type)
print('gate layers type:', gate_layers_type)
print('k:', k)
print('model name:', m)
print('model type:', mt)
print('runs:', runs)
print('total experts:', total_experts)
print('Num epochs:', num_epochs)
print('Hidden layers:', hidden)
print('importance factor:', w_importance_range[0])
print('sample similarity factor:', w_sample_sim_same_range[0])
print('sample dissimilarity factor:', w_sample_sim_diff_range[0])

num_classes = 10

# Paths to where the trained models, figures and results will be stored. You can change this as you see fit.
working_path = '/nobackup/projects/bdrap03/yamuna/modular_deep_learning/aaai_2022/src'
model_path = os.path.join(working_path, '../models/mnist')

train_with_attention(m, mt, k, trainloader, testloader, 
                     expert_layers=expert_layers_type, gate_layers=gate_layers_type,
                     w_importance_range=w_importance_range,
                     w_sample_sim_same_range=w_sample_sim_same_range, 
                     w_sample_sim_diff_range=w_sample_sim_diff_range,
                     runs=runs, temps=[[1.0]*num_epochs], hidden=hidden, num_classes=num_classes, 
                     total_experts=total_experts, num_epochs=num_epochs, model_path=model_path)
