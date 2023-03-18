import argparse
import sys
from cifar10_original_moe_training import *
from original_moe_training import *
from helper.collect_results import *
from torchvision.models import resnet18

sys.path.append('WideResNet-pytorch-master')
from wideresnet import WideResNet


expert_layers_types = {'expert_layers': expert_layers,
                      'expert_layers_conv_2': expert_layers_conv_2}

gate_layers_types = {'gate_layers': gate_layers,
                     'gate_layers_conv_2': gate_layers_conv_2,
                     'gate_layers_conv_2_top_k': gate_layers_conv_2_top_k,
                    'gate_layers_top_k': gate_layers_top_k}

expert_layers_type = expert_layers
gate_layers_type = gate_layers

m = 'cifar10_without_reg'
mt = 'moe_expectation_model'
total_experts = 10
k = 0
runs = 1
w_importance_range=[0.0]
w_sample_sim_same_range = [0.0]
w_sample_sim_diff_range = [0.0]
d = 'default_distance_funct'
distance_funct = default_distance_funct
filename = 'cifar10_top_k_results.csv'

parser = argparse.ArgumentParser()
parser.add_argument('-e', help='expert layers type')
parser.add_argument('-g', help='gate layers type')
parser.add_argument('-k', help='top k')
parser.add_argument('-m', help='model name')
parser.add_argument('-mt', help='model type')
parser.add_argument('-r', help='number of runs')
parser.add_argument('-E', help='number of epochs')
parser.add_argument('-M', help='number of experts')
parser.add_argument('-D', help='sample distance function')
parser.add_argument('-i', help='Importance factor')
parser.add_argument('-ss', help='sample similarity factor')
parser.add_argument('-sd', help='sample dissimilarity factor')
parser.add_argument('-f', help='filename')
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
if not args['D'] is None:
    d = args['D']
if not args['i'] is None:
    w_importance_range = [float(args['i'])]
if not args['ss'] is None:
    w_sample_sim_same_range = [float(args['ss'])]
if not args['sd'] is None:
    w_sample_sim_diff_range = [float(args['sd'])]
if not args['f'] is None:
    runs = args['f']

print('expert layers type:', expert_layers_type)
print('gate layers type:', gate_layers_type)
print('k:', k)
print('model name:', m)
print('model type:', mt)
print('runs:', runs)
print('total experts:', total_experts)
print('Num epochs:', num_epochs)
print('importance factor:', w_importance_range[0])
print('sample similarity factor:', w_sample_sim_same_range[0])
print('sample dissimilarity factor:', w_sample_sim_diff_range[0])
print('results filename:', filename)

num_classes = 10

# Paths to where the trained models, figures and results will be stored. You can change this as you see fit.
working_path = '/nobackup/projects/bdrap03/yamuna/modular_deep_learning/aaai_2022/src'
model_path = os.path.join(working_path, '../models/cifar10')
results_path = os.path.join(working_path, '../results/cifar10')

if not os.path.exists(model_path):
    os.mkdir(model_path)

if d == 'resnet_distance_funct':
    print('sample distance function: resnet_distance_funct')
    model_state = torch.load(os.path.join(model_path, 'resnet.ckpt'))
    model = resnet18(pretrained=True).to(device)
    model.load_state_dict(model_state)
    # model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    distance_funct = resnet_distance_funct(model).distance_funct
    
elif d == 'wideres_distance_funct':
    print('sample distance function: wideres_distance_funct') 
    number_of_layers = 40
    model = WideResNet(depth=number_of_layers, num_classes=10, widen_factor=4)
    checkpoint = torch.load('WideResNet-pytorch-master/runs/WideResNet-28-10/cifar10.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    # model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    distance_funct = resnet_distance_funct(model).distance_funct

collect_results(m, 'moe_expectation_model', 
                w_sample_sim_same_range=w_sample_sim_same_range,
                w_sample_sim_diff_range=w_sample_sim_diff_range,
                total_experts=total_experts, num_classes=num_classes, num_epochs=num_epochs, 
                testloader=cifar10_testloader, model_path=model_path, results_path=results_path, filename=filename)