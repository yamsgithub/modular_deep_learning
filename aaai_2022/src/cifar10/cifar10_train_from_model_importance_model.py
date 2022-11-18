import argparse
from cifar10_moe_with_attention_training import *

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
model = 'cifar10_with_attn_reg'

total_experts = 5

num_classes = 10

num_epochs = 40

train_from_model(model, num_epochs, num_classes, total_experts, w_importance=w_importance, trainloader=cifar10_trainloader, testloader=cifar10_testloader, expert_no_grad=True, gate_no_grad=False)
  
