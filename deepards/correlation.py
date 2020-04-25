"""
correlation
~~~~~~~~~~~~~

Want to look through our data and see if auto-correlation or cross-correlation will
tell us anything about the data involved.
"""

import argparse
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import seaborn as sns
import statsmodels.api as sm
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_pred_lower_prob(prob):
    for i in np.arange(.5, 1+.01, .05):
        if i <= prob < i+.05:
            return i


def get_auto_corr_r2(seq):
    ac = np.correlate(seq, seq, mode='same')[:len(seq)/2]
    # one thing that I've noticed is that theres quite a bit of noise in the ac signal
    # so it'll probably need to go thru a bandpass filter of some kind.
    #
    # What's a good sigma? maybe something a bit higher try 10 for now
    ac = gaussian_filter(ac, 10)
    peak_func = lambda x: [v for i, v in enumerate(x[1:-1]) if x[i] < v > x[i+2] and v > 0]
    filt = peak_func(ac)
    x = pd.DataFrame(np.arange(len(filt)).reshape((len(filt), 1)))
    x = sm.add_constant(x)
    est = sm.OLS(pd.Series(filt), x, hasconst=True)
    res = est.fit()
    return res.rsquared


def autocorrelate_by_pred(model_data):
    cls_map = {
        0: {'name': 'Non-ARDS', 'pred': [], 'actual': [], 'pred_pt': [], 'actual_pt': []},
        1: {'name': 'ARDS', 'pred': [], 'actual': [], 'pred_pt': [], 'actual_pt': []},
    }
    true = []
    pred = []
    for i, seq in enumerate(model_data['seqs']):
        # Oh man... I think my initial analysis for this was incorrect. I think I
        # misunderstood what valid mode was doing and misinterpreted the results. So
        # what I think the analysis meant was that the magnitude of the ARDS data was larger.
        #
        # If you want to determine synchonicity then I think periodicity would be a good
        # thing to look at, like how quickly does it cycle from max to min
        #
        # I'm not sure it would be periodicity. I think it would be more about the ratio
        # of maxes between peaks. For instance a very synchronous waveform would see peaks
        # grow linearly by same ratio.
        #
        # I mean if you want a linear function to be better than an exponential one, then
        # you can just do a regression with the peaks and calc MSE. In this case the most
        # linear of MSEs will win.
        #
        # When looking at data, this method works just fine for VC data but there is also
        # information with double-humped periodicities in PC/PS. This would cause the
        # AC to have minor secondary spikes. In effect, you'd have to ignore the secondary
        # spikes and only focus on the primary ones. Well... I dunno I guess I could invest
        # a ton of engineering effort to work out the corner cases but I don't have time
        # and its not useful energy spent. So I will work on my basic algo
        r2 = get_auto_corr_r2(seq)
        #if .98 <= r2 <= 1.0:
        #    import IPython; IPython.embed()
        #elif .00 <= r2 <= .02:
        #    import IPython; IPython.embed()

        pred = model_data['pred'][i]
        actual = model_data['actual'][i]
        patient = model_data['patient'][i]

        cls_map[pred]['pred'].append(r2)
        cls_map[actual]['actual'].append(r2)
        cls_map[pred]['pred_pt'].append(patient)
        cls_map[actual]['actual_pt'].append(patient)

    for cls in cls_map:
        name = cls_map[cls]['name']
        sns.distplot(cls_map[cls]['actual'], kde=False, label='{}-true'.format(name), bins=100)
        sns.distplot(cls_map[cls]['pred'], kde=False, label='{}-pred'.format(name), bins=100, hist_kws={'alpha': 0.4})
        plt.legend()
        plt.show()

    for cls in cls_map:
        data = cls_map[cls]['actual']
        data_pt = [float(i[:4]) for i in cls_map[cls]['actual_pt']]
        act = np.array(data).reshape((len(data), 1))
        act_pt = np.array(data_pt).reshape((len(data_pt), 1))
        arr = np.append(act, act_pt, axis=1)
        arr = arr[arr[:, 0] < .2]
        # XXX TODO I wanted to look at what specific classes were doing with synchronous data
        # and which patients had it versus which didnt, and then to tell if we were
        # misclassifying it

    misclassified = []
    misclassified_ards = []
    misclassified_other = []
    for i, seq in enumerate(model_data['seqs']):
        if model_data['actual'][i] != model_data['pred'][i]:
            r2 = get_auto_corr_r2(seq)
            misclassified.append(r2)

            if model_data['actual'][i] == 1:
                misclassified_ards.append(r2)
            else:
                misclassified_other.append(r2)

    sns.distplot(misclassified, kde=False, bins=100, label='all misclassified')
    sns.distplot(misclassified_other, kde=False, bins=100, label='other misclassified', hist_kws={'alpha': .5})
    sns.distplot(misclassified_ards, kde=False, bins=100, label='ards misclassified')
    plt.legend()
    plt.show()


def cross_correlations(model_data, cc_samp_size):
    prob_range = np.arange(.5, 1+.01, .05)
    probs = {'ards': {i: [] for i in prob_range}, 'non_ards': {i: [] for i in prob_range}}
    samp_size = int(len(model_data['seqs']) * cc_samp_size)
    samp_idx = np.random.choice(np.arange(0, len(model_data['seqs'])), replace=False, size=samp_size)
    model_data = {k: [v[i] for i in samp_idx] for k, v in model_data.items()}

    for i, seq in enumerate(model_data['seqs']):
        pred = model_data['pred'][i]
        pred_name = {0: 'non_ards', 1: 'ards'}[pred]
        actual = model_data['actual'][i]
        prob = model_data['pred_prob'][i]
        lower_prob = get_pred_lower_prob(prob)

        for j, s in enumerate(model_data['seqs']):
            if i == j:
                continue
            pred_j = model_data['pred'][j]

            if pred_j != pred:
                continue

            prob_j = model_data['pred_prob'][j]
            lower_prob_j = get_pred_lower_prob(prob_j)
            if lower_prob == lower_prob_j:
                # mode='valid' isnt terrible for cross-correlation. It might be limited in
                # what it tells you though. It's probably going to mostly tell you which
                # sequences came from the same patient. As for the probabiliy ratios, I'm
                # not that surprised it failed to yield any information. It's likely that
                # false pos/neg work wont tell us anything either.
                probs[pred_name][lower_prob].append(np.correlate(seq, s, mode='valid'))

    pd.to_pickle(probs, 'cc-probs-last-run.pkl')

    import IPython; IPython.embed()


def run_entire_model(test_dataset_path, model_path):
    test_dataset = pd.read_pickle(test_dataset_path)
    test_dataset.transforms = None
    model = torch.load(model_path)
    model_data = {'patient': [], 'seqs': [], 'pred': [], 'actual': [], 'pred_prob': []}
    softmax = nn.Softmax()
    # use 256 because its a big number or something like that
    bs = 256

    with torch.no_grad():
        for data in DataLoader(test_dataset, batch_size=bs):
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

    return model_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='test set data to load.')
    parser.add_argument('model', help='Path to saved pytorch .pth model file')
    parser.add_argument('--cc-samp-size', type=float, default=.2, help='fraction of sequences to choose for cross correlation')
    parser.add_argument('-sm', '--save-model-data')
    parser.add_argument('-lm', '--load-model-data')
    args = parser.parse_args()

    if not args.load_model_data:
        model_data = run_entire_model(args.file, args.model)
    else:
        model_data = pd.read_pickle(args.load_model_data)

    if args.save_model_data:
        pd.to_pickle(model_data, args.save_model_data)
    #cross_correlations(model_data, args.cc_samp_size)
    autocorrelate_by_pred(model_data)


if __name__ == "__main__":
    main()
