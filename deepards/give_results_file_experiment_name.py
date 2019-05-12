import argparse
from glob import glob

import torch


parser = argparse.ArgumentParser()
parser.add_argument('start_time')
parser.add_argument('experiment_name')
parser.add_argument('-b', '--batch-size', required=True)
parser.add_argument('-lr', '--learning-rate', default='.001')
parser.add_argument('-n', '--network', default='cnn_lstm')
parser.add_argument('--base-network', default='resnet18')
args = parser.parse_args()


dict = {
    'start_time': args.start_time,
    'batch_size': args.batch_size,
    'learning_rate': args.learning_rate,
    'network': args.network,
    'base_network': args.base_network,
}
torch.save(dict, 'results/{}_{}.pth'.format(args.experiment_name, args.start_time))
