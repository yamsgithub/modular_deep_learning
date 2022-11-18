import argparse
from cifar100_moe_with_attention_training_hidden256 import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='task count')
args = vars(parser.parse_args())
print('args', args)

w_importance = 0.0
w_sample_sim_same = 0.0
w_sample_sim_diff = 0.0

if not args['i'] is None:
    w_importance = 0.2*int(args['i'])

# expert resnet pre-trained
# gate resnet not pre-trained
model = 'cifar100_with_attn_reg'

total_experts = 20

num_classes = 100

num_epochs = 40

runs = 5

train_from_model(model, num_epochs, num_classes, total_experts, w_importance=w_importance, trainloader=cifar100_trainloader, testloader=cifar100_testloader, expert_no_grad=True, gate_no_grad=False)
  
