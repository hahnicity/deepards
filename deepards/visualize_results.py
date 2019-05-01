import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import torch


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--network', default='cnn_lstm')
parser.add_argument('-bn', '--base-network', default='resnet18')
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('-nb', '--num-sub-batches', default=100)
parser.add_argument('-lc', '--loss-calc', default='all_breaths')
parser.add_argument('-rip', '--initial-planes', default=64)
parser.add_argument('-lvp', '--lstm-vote-percentage', default=70)
parser.add_argument('-o', '--optim', default='sgd')
parser.add_argument('-lr', '--learning-rate', type=float, default=.001)
parser.add_argument('-b', '--batch-size', type=int, default=16)
parser.add_argument('-rdc', '--resnet-double-conv', action='store_true')
args = parser.parse_args()

get_moving_average = lambda x, N: np.convolve(x, np.ones((N,))/N, mode='valid')

glob_search = 'results/loss*_{network}_base{base_network}_e{epochs}_nb{num_sub_batches}_lc{loss_calc}_rip{initial_planes}_lvp{lstm_vote_percentage}_rfptmax_optim{optim}_lr{learning_rate}_bs{batch_size}_rdc{resnet_double_conv}*'
glob_search = glob_search.format(**args.__dict__)
results_files = glob(glob_search)
if len(results_files) == 0:
    raise Exception('No loss results files found')

for i, f in enumerate(sorted(results_files)):
    vals = torch.load(f).values.numpy()
    ma = get_moving_average(vals, 1000)
    plt.plot(ma, label='Loss Fold {}'.format(i+1))
plt.legend()
plt.grid()
plt.ylabel('loss')
plt.show()

all_vals = None
glob_search = 'results/test_auc*_{network}_base{base_network}_e{epochs}_nb{num_sub_batches}_lc{loss_calc}_rip{initial_planes}_lvp{lstm_vote_percentage}_rfptmax_optim{optim}_lr{learning_rate}_bs{batch_size}_rdc{resnet_double_conv}*'
glob_search = glob_search.format(**args.__dict__)
print('searching for results with params:')
pprint(args.__dict__)
results_files = glob(glob_search)
if len(results_files) == 0:
    raise Exception('No AUC results files found')

for i, f in enumerate(sorted(results_files)):
    vals = torch.load(f).values.numpy()
    if all_vals is None:
        all_vals = vals
    else:
        all_vals += vals
    plt.plot(vals, label='AUC Fold {}'.format(i+1))
all_vals = all_vals / len(glob(glob_search))
plt.plot(all_vals, label='mean_auc')
plt.legend(loc=3)
plt.grid()
plt.ylabel('AUC')
plt.xlabel('epochs')
plt.ylim(.45, 1)
plt.yticks(np.arange(.45, 1.01, .05))
plt.xticks(range(0, len(all_vals)), range(1, len(all_vals)+1))
#plt.title(
plt.show()

glob_search = 'results/test_accuracy*_{network}_base{base_network}_e{epochs}_nb{num_sub_batches}_lc{loss_calc}_rip{initial_planes}_lvp{lstm_vote_percentage}_rfptmax_optim{optim}_lr{learning_rate}_bs{batch_size}_rdc{resnet_double_conv}*'
glob_search = glob_search.format(**args.__dict__)
results_files = glob(glob_search)
if len(results_files) == 0:
    raise Exception('No accuracy results files found')

for i, f in enumerate(sorted(results_files)):
    vals = torch.load(f).values.numpy()
    ma = get_moving_average(vals, 100)
    plt.plot(ma, label='Test accuracy Fold {}'.format(i+1))
plt.legend()
plt.grid()
plt.ylabel('Accuracy')
plt.ylim(.5, 1)
plt.yticks(np.arange(.5, 1.01, .05))
#plt.title(
plt.show()
