from __future__ import print_function
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

from deepards.metrics import DeepARDSResults
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
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--base-network', choices=['resnet18'], default='resnet18')
    parser.add_argument('--loss-calc', choices=['all_breaths', 'last_breath'], required=True)
    parser.add_argument('-nb', '--n-breaths-in-seq', type=int, default=20)
    parser.add_argument('--no-print-progress', action='store_true')
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
        args.n_breaths_in_seq,
        to_pickle=args.train_to_pickle,
        from_pickle=args.train_from_pickle
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    n_loss = 0
    total_loss = 0

    with torch.enable_grad():
        for ep in range(args.epochs):
            print("\nrun epoch {}\n".format(ep+1))
            for idx, (obs_idx, patient, seq, target) in enumerate(train_loader):
                hidden = model.init_hidden(seq.shape[0])
                model.zero_grad()
                target_shape = target.numpy().shape
                target = cuda_wrapper(target.float())
                inputs = cuda_wrapper(Variable(seq.float()))
                outputs = model(inputs, hidden)
                if args.loss_calc == 'all_breaths':
                    if args.batch_size > 1:
                        target = target.unsqueeze(1)
                    loss = criterion(outputs, target.repeat((1, args.n_breaths_in_seq, 1)))
                elif args.loss_calc == 'last_breath':
                    loss = criterion(outputs[:, -1, :], target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # print individual loss and total loss
                total_loss += loss.data
                n_loss += 1
                if not args.no_print_progress:
                    print("batch num: {}/{}, avg loss: {}\r".format(idx+1, len(train_loader), total_loss/n_loss), end="")

    test_dataset = ARDSRawDataset(
        args.data_path,
        args.experiment_num,
        args.cohort_file,
        args.n_breaths_in_seq,
        to_pickle=args.test_to_pickle,
        from_pickle=args.test_from_pickle,
        train=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    preds = []
    pred_idx = []
    y_test = test_dataset.get_ground_truth_df()
    results = DeepARDSResults(y_test)
    with torch.no_grad():
        for idx, (obs_idx, patient, seq, target) in enumerate(test_loader):
            hidden = model.init_hidden(seq.shape[0])
            inputs = cuda_wrapper(Variable(seq.float()))
            outputs = model(inputs, hidden)
            # get the last prediction in the LSTM chain. Just do this for now. Maybe
            # later we can have a slightly more sophisticated voting. Or we can just
            # skip all of that together and only backprop on the last item.
            preds.extend(outputs[:, -1, :].argmax(dim=1).cpu().tolist())
            pred_idx.extend(obs_idx.cpu().tolist())

    preds = pd.Series(preds, index=pred_idx)
    preds = preds.sort_index()
    results.perform_patient_predictions(preds)
    results.aggregate_all_results()
    print(results.aggregate_stats)

if __name__ == "__main__":
    main()
