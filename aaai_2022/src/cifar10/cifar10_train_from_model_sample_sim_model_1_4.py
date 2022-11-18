import argparse
from cifar10_moe_with_attention_training import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='task count')
args = vars(parser.parse_args())
print('args', args)

w_sample_sim_same = 0.0
w_sample_sim_diff = 0.0

w_sample_sim_same_range = [1e-4]
w_sample_sim_diff_range = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
sample_sim_weights = list(product(w_sample_sim_same_range,w_sample_sim_diff_range))

if not args['i'] is None:
    index = int(args['i'])-1
    w_sample_sim_same = sample_sim_weights[index][0]
    w_sample_sim_diff = sample_sim_weights[index][1]

# expert resnet pre-trained
# gate resnet not pre-trained
model = 'cifar10_with_attn_reg'

total_experts = 5

num_classes = 10

num_epochs = 40

train_from_model(model, num_epochs, num_classes, total_experts, w_sample_sim_same=w_sample_sim_same, w_sample_sim_diff=w_sample_sim_diff, trainloader=cifar10_trainloader, testloader=cifar10_testloader, expert_no_grad=True, gate_no_grad=False)
  
