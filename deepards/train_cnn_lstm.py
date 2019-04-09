from __future__ import print_function
import argparse

import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from deepards.models.resnet import resnet18
from deepards.models.torch_cnn_lstm_combo import CNNLSTMNetwork
from deepards.dataset import ARDSRawDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--data-path', default='/fastdata/ardsdetection', help='Path to ARDS detection dataset')
    parser.add_argument('-en', '--experiment-num', type=int, default=1)
    parser.add_argument('-c', '--cohort-file', default='cohort-description.csv')
    parser.add_argument('-n', '--network', choices=['basic'], default='basic')
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('-p', '--train-from-pickle')
    parser.add_argument('--train-to-pickle')
    parser.add_argument('--test-from-pickle')
    parser.add_argument('--test-to-pickle')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--base-network', choices=['resnet18'], default='resnet18')
    args = parser.parse_args()

    cuda_wrapper = lambda x: x.cuda() if args.cuda else x
    network_map = {'basic': CNNLSTMNetwork}
    base_network = {'resnet18': resnet18}[args.base_network]()
    model = cuda_wrapper(network_map[args.network](base_network))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()
    train_dataset = ARDSRawDataset(
        args.data_path,
        args.experiment_num,
        args.cohort_file,
        to_pickle=args.train_to_pickle,
        from_pickle=args.train_from_pickle
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    n_loss = 0
    total_loss = 0

    with torch.enable_grad():
        for ep in range(args.epochs):
            print("\nrun epoch {}\n".format(ep+1))
            for idx, (patient, seq, target) in enumerate(train_loader):
                hidden = model.init_hidden(seq.shape[0])
                model.zero_grad()
                target_shape = target.numpy().shape
                target = cuda_wrapper(target.float())
                inputs = cuda_wrapper(Variable(seq.float()))
                outputs = model(inputs, hidden)
                # Tried doing this and it didn't work well
                #loss = criterion(outputs[:, -1, :], target.squeeze(dim=0))
                # the below doesn't work so well either, but works better
                if args.batch_size > 1:
                    target = target.unsqueeze(1)
                loss = criterion(outputs, target.repeat((1, 20, 1)))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print individual loss and total loss
                total_loss += loss.data
                n_loss += 1
                print("batch num: {}/{}, avg loss: {}\r".format(idx+1, len(train_loader), total_loss/n_loss), end="")

    test_dataset = ARDSRawDataset(
        args.data_path,
        args.experiment_num,
        args.cohort_file,
        to_pickle=args.test_to_pickle,
        from_pickle=args.test_from_pickle,
        train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    preds = None
    gt = None
    with torch.no_grad():
        for idx, (patient, seq, target) in enumerate(test_loader):
            hidden = model.init_hidden(seq.shape[0])
            inputs = cuda_wrapper(Variable(seq.float()))
            outputs = model(inputs, hidden)
            if args.batch_size > 1:
                target = target.unsqueeze(1)
            target = target.repeat((1, 20, 1))
            if preds is None:
                gt = target
                preds = outputs
            else:
                gt = torch.cat([gt, target], dim=0)
                preds = torch.cat([preds, outputs], dim=0)
    import IPython; IPython.embed()
    print(classification_report(gt.argmax(dim=-1).view(-1).cpu().numpy(), preds.argmax(dim=-1).view(-1).cpu().numpy()))

if __name__ == "__main__":
    main()
