import argparse
from cifar100_moe_with_attention_training import *

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='task count')
args = vars(parser.parse_args())
print('args', args)

w_importance = 0.0
w_sample_sim_same = 0.0
w_sample_sim_diff = 0.0

w_sample_sim_same_range = [1e-5]
w_sample_sim_diff_range = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
sample_sim_weights = list(product(w_sample_sim_same_range,w_sample_sim_diff_range))

if not args['i'] is None:
    index = int(args['i'])-1
    w_sample_sim_same = sample_sim_weights[index][0]
    w_sample_sim_diff = sample_sim_weights[index][1]

# expert resnet pre-trained
# gate resnet not pre-trained
model = 'cifar100_with_reg'

total_experts = 20

num_classes = 100

num_epochs = 40

runs = 5

train_with_attention(model, cifar100_trainloader, cifar100_testloader, runs, T=[1.0]*num_epochs, 
                         w_importance=w_importance, w_sample_sim_same=w_sample_sim_same, 
                         w_sample_sim_diff=w_sample_sim_diff,
                         num_classes=num_classes, total_experts=total_experts, num_epochs=num_epochs)
