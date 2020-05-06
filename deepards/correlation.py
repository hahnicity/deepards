"""
correlation
~~~~~~~~~~~~~

Want to look through our data and see if auto-correlation or cross-correlation will
tell us anything about the data involved. From what I could see, cross-correlation
didn't tell us anything.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import seaborn as sns
import statsmodels.api as sm
import torch
from torch import nn
from torch.utils.data import DataLoader

import deepards.dataset


class AutoCorrelation(object):
    def analyze_dataset_only(self, dataset):
        if dataset.oversample:
            raise Exception('oversample is turned on. this will skew results! Turn off oversampling and redo fold indexing!')
        r2 = [self.get_auto_corr_r2(seq[1].ravel()) for seq in dataset]
        actual = [seq[-1].argmax() for seq in dataset]
        arr = np.array([r2, actual]).T
        sns.distplot(arr[arr[:, 1] == 0][:, 0], kde=False, label='non-ards dist', hist_kws={'alpha': 0.5, }, bins=100)
        sns.distplot(arr[arr[:, 1] == 1][:, 0], kde=False, label='ards dist', hist_kws={'alpha': 0.5, }, bins=100)
        plt.legend()
        plt.show()

    def get_auto_corr_r2(self, seq):
        ac = np.correlate(seq, seq, mode='same')[:len(seq)/2]
        # one thing that I've noticed is that theres quite a bit of noise in the ac signal
        # so it'll probably need to go thru a bandpass filter of some kind.
        #
        # What's a good sigma? maybe something a bit higher try 10 for now
        ac = gaussian_filter(ac, 10)
        peak_func = lambda x: [v for i, v in enumerate(x[1:-1]) if x[i] < v > x[i+2] and v > 0]
        filt = [ac[0]] + peak_func(ac) + [ac[-1]]

        x = pd.DataFrame(np.arange(len(filt)).reshape((len(filt), 1)))
        x = sm.add_constant(x)
        est = sm.OLS(pd.Series(filt), x, hasconst=True)
        res = est.fit()
        return res.rsquared

    def autocorrelate_by_pred(self, model_data):
        model_data['r2'] = []
        for i, seq in enumerate(model_data['seqs']):
            model_data['r2'].append(self.get_auto_corr_r2(seq))

        # XXX find way to remove this plotting func into other place
#        for cls, name in {0: "Non-ARDS", 1: 'ARDS'}.items():
#            cls_preds = [model_data['r2'][idx] for idx, pred in enumerate(model_data['pred']) if model_data['pred'][idx] == cls]
#            cls_actuals = [model_data['r2'][idx] for idx, actual in enumerate(model_data['actual']) if model_data['actual'][idx] == cls]
#            sns.distplot(cls_actuals, kde=False, label='{}-true'.format(name), bins=100)
#            sns.distplot(cls_preds, kde=False, label='{}-pred'.format(name), bins=100, hist_kws={'alpha': 0.4})
#            plt.legend()
#            plt.show()

        misclassified = {'all': [], 1: [], 0: []}
        for i, seq in enumerate(model_data['seqs']):
            actual = model_data['actual'][i]
            if actual != model_data['pred'][i]:
                r2 = model_data['r2'][i]
                misclassified['all'].append(r2)
                misclassified[actual].append(r2)

        return misclassified

    def format_r2_for_bar_chart(self, arr):
        arr = np.array(sorted(arr))
        interval = .01
        counts = []
        x = np.arange(0, 1, interval)
        for lower in x:
            upper = lower + interval
            counts.append(len(arr[np.logical_and(lower < arr, arr <= upper)]))
        return np.round(x, 2), np.array(counts)


def run_entire_model(test_dataset_path, model_path, kfold_num):
    test_dataset = pd.read_pickle(test_dataset_path)
    test_dataset.transforms = None
    if kfold_num is not None:
        test_dataset = deepards.dataset.ARDSRawDataset.make_test_dataset_if_kfold(test_dataset)
        test_dataset.set_kfold_indexes_for_fold(kfold_num)
    model = torch.load(model_path).cuda(0)
    model_data = {'patient': [], 'seqs': [], 'pred': [], 'actual': [], 'pred_prob': []}
    softmax = nn.Softmax()
    # use 256 because its a big number or something like that
    bs = 256

    with torch.no_grad():
        for data in DataLoader(test_dataset, batch_size=bs, num_workers=0):
            # add patient data
            for idx in data[0]:
                model_data['patient'].append(test_dataset.all_sequences[idx][0])

            # work on test predictions
            seq = data[1]
            target = data[3].argmax(dim=1)
            cuda_seq = seq.float().cuda()
            output = model(cuda_seq, None)
            batch_predictions = softmax(output).argmax(dim=1)
            model_data['seqs'].extend(seq.view(seq.shape[0], -1).numpy())
            model_data['pred'].extend(batch_predictions.tolist())
            model_data['actual'].extend(target.tolist())
            model_data['pred_prob'].extend([arr[batch_predictions[i]] for i, arr in enumerate(softmax(output))])

    model_data['model_name'] = os.path.splitext(os.path.basename(model_path))[0]
    return model_data


def plot_misclassified_data(model_data):
    auto = AutoCorrelation()
    misclassified = auto.autocorrelate_by_pred(model_data)
    sns.distplot(misclassified[0], kde=True, bins=100, label='other misclassified', hist_kws={'alpha': .5}, hist=False)
    sns.distplot(misclassified[1], kde=True, bins=100, label='ards misclassified', hist=False)
    plt.title(model_data['model_name'])
    plt.legend()
    plt.show()

    every_fifth = lambda x: x % 5 == 0
    x, other = auto.format_r2_for_bar_chart(misclassified[0])
    x, ards = auto.format_r2_for_bar_chart(misclassified[1])
    plt.bar(range(len(other)), other, label='other misclassified')
    plt.bar(range(len(ards)), ards, bottom=other, label='ards misclassified')
    plt.xticks(filter(every_fifth, range(len(ards))), filter(every_fifth, x), rotation=45, fontsize='xx-small')
    plt.title(model_data['model_name'])
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='test set data to load.')
    parser.add_argument('model', help='Path to saved pytorch .pth model file')
    parser.add_argument('--cc-samp-size', type=float, default=.2, help='fraction of sequences to choose for cross correlation')
    parser.add_argument('-sm', '--save-model-data')
    parser.add_argument('-lm', '--load-model-data')
    parser.add_argument('-k', '--kfold-num', type=int, help='kfold number if you are using kfold dataset. Fold nums should start at 0. so 0,1,2,3,4...etc.')
    args = parser.parse_args()

    if not args.load_model_data:
        model_data = run_entire_model(args.file, args.model, args.kfold_num)
    else:
        model_data = pd.read_pickle(args.load_model_data)

    if args.save_model_data:
        pd.to_pickle(model_data, args.save_model_data)

    plot_misclassified_data(model_data)


if __name__ == "__main__":
    main()
