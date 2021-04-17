"""
credits for cam code to:

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import csv
from pathlib import Path
import random

import numpy as np
import torch
import argparse
import pickle
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, resample, sosfilt
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch.nn.functional as F

from deepards import dataset


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.softmax = torch.nn.Softmax()

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        x = self.model.breath_block.features(x)
        # the register_hook function is exclusively for examining gradient
        x.register_hook(self.save_gradient)
        conv_output = x
        return conv_output, x

    def forward_pass(self, x):
        """
        Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # they forgot relu when they were doing this initially. damn
        x = F.relu(x)
        # pool and flatten
        try:
            x = self.model.breath_block.avgpool(x).view(-1)
        except AttributeError:
            x = F.avg_pool2d(x, 7).view(-1)
        x = self.model.linear_final(x).unsqueeze(0)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        # dont use eval because it will normalize twice. It does slow things
        # down, but its only 1 item.
        #self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_one_hot_grad_and_output(self, input, target):
        return self._generate_grad_and_output(input, target, self.one_hot_model_output)

    def _generate_grad_and_output(self, input, target, model_out_func):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        conv_output, model_output = self.extractor.forward_pass(input)
        # this line ensures grad cam is done wrt model prediction
        output = model_out_func(model_output, target)
        self.model.zero_grad()
        # Backward pass wrt output
        # retention of the graph is not necessary if we are only running a single
        # backward op. However if we ran cuda on multiple different layers in the
        # same network after the same pass then graph retention would be necessary.
        output.backward(retain_graph=False)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().data.numpy()
        # Get convolution outputs
        conv_output = conv_output.cpu().data.numpy()
        return conv_output, guided_gradients, model_output

    def one_hot_model_output(self, output, target):
        if target is None:
            target = np.argmax(output.cpu().data.numpy())
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        return torch.sum(one_hot.cuda() * output)


class MaxMinNormCam(GradCam):
    """
    normalize cam outputs based on max/min of just the single class output.
    This gives us good insight into what the model is doing with just a single
    class but the downside is that it doesnt tell us the real strength of the
    class association wrt other classes, or even with a single class. For instance
    the predicted prob for a class can be very low but grad-cam can still show
    very brightly, which is a bit misleading because it implies that the network
    focused on an area intensely, whereas the area could have just been the
    strongest area out of many weak ones
    """
    # XXX can probably drop grads one of these days
    grads = []
    preds = []

    def generate_read_cam(self, input, target):
        conv_output, grad, mo = self.generate_one_hot_grad_and_output(input, target)
        # XXX can probably drop grads one of these days
        self.grads.append(grad)
        self.preds.append(mo)
        weights = np.mean(grad, axis=(2,))
        cam = np.zeros((conv_output.shape[0], conv_output.shape[2]), dtype=np.float32)
        for i, b in enumerate(conv_output):
            for j, w in enumerate(weights[i,:]):
                cam[i] += w * conv_output[i, j, :]
            cam[i] = self.normalize(cam[i])
        return cam, mo

    def generate_cam(self, input, target=None):
        conv_output, grad, mo = self.generate_one_hot_grad_and_output(input, target)
        # XXX can probably drop grads one of these days
        self.grads.append(grad)
        self.preds.append(mo)
        weights = np.mean(grad, axis=(0, 2))
        # Take averages across all breaths because of the way we are structuring
        # our model. We can in the future just pick a specific breath if we want
        # to visualize single breaths in a true breath window. currently though
        # we are just using cloned breaths for an entire read. So really, averaging
        # the mapping should do very little to the underlying variable.
        conv_output = np.mean(conv_output, axis=0)
        cam = np.zeros(conv_output.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * conv_output[i, :]

        return self.normalize(cam), mo

    def normalize(self, cam):
        # perform relu
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


class FracTotalNormCam(GradCam):
    """
    Apparently this did not help the situation caused by max min.
    """
    def generate_cam(self, input, target):
        raise NotImplementedError('Havent done this yet')

    def generate_read_cam(self, input, target):
        conv_output, grad_target, mo = self.generate_one_hot_grad_and_output(input, target)
        _, grad_other, __ = self.generate_one_hot_grad_and_output(input, (target + 1) % 2)
        weights_target = np.mean(grad_target, axis=(2,))
        weights_other = np.mean(grad_other, axis=(2,))
        cam_target = np.zeros((conv_output.shape[0], conv_output.shape[2]), dtype=np.float32)
        cam_other = np.zeros((conv_output.shape[0], conv_output.shape[2]), dtype=np.float32)
        cam = np.zeros((conv_output.shape[0], conv_output.shape[2]), dtype=np.float32)
        for i, b in enumerate(conv_output):
            for j, w in enumerate(weights_target[i,:]):
                cam_target[i] += w * conv_output[i, j, :]
                cam_other[i] += weights_other[i, j] * conv_output[i, j, :]
            cam[i] = self.normalize(cam_target[i], cam_other[i])
        return cam, mo

    def normalize(self, cam_target, cam_other):
        cam_target = np.maximum(cam_target, 0)
        cam_other = np.maximum(cam_other, 0)
        cam = cam_target / (cam_other + cam_target)
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


class UnNormalizedCam(GradCam):
    def generate_cam(self, input, target=None):
        conv_output, grad, mo = self.generate_one_hot_grad_and_output(input, target)
        weights = np.mean(grad, axis=(0, 2))
        conv_output = np.mean(conv_output, axis=0)
        cam = np.zeros(conv_output.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_output[i, :]

        # just apply ReLU and be done
        return np.maximum(0, cam), mo


def get_sequence(filename, ards, c, fold):
    data = pickle.load(open(filename, "rb"))
    data.set_kfold_indexes_for_fold(fold)
    data.transforms = None
    first_seq = None
    count = 0
    for i, d in enumerate(data.all_sequences):
        if d[2][1] == ards:
            if count == c:
                first_seq = d
                break
            count = count + 1
            continue
    br = first_seq[1]
    br = torch.FloatTensor(br).cuda()
    return br


def img_process(model_in):
    img = model_in.squeeze().cpu()
    # process img to 0-1
    img -= img.min()
    img = img / img.max()
    if img.shape[0] == 2:
        img = torch.cat([img, torch.ones((1, 224, 224)).double()], dim=0)
    return img.numpy()


def cam_process(cam, seq_shape, normalize):
    cam = cv2.resize(cam, seq_shape)
    if cam.shape[0] == 1:
        cam = cam.ravel()
    # process cam to 0-1
    if normalize:
        cam -= cam.min()
        cam = cam / cam.max()
    return cam


def get_fft(seq):
    real = seq[:, 0, :]
    imag = seq[:, 1, :]
    fft_with_shift = real + (1j*imag)
    return fft_with_shift


def fft_to_ts(seq):
    fft = get_fft(seq)
    fft_noshift = np.fft.ifftshift(fft)
    signal = np.fft.ifft(fft_noshift)
    return signal


def fft_to_ts_with_mask(seq, mask):
    fft = get_fft(seq)
    fft_noshift = np.fft.ifftshift(fft*mask)
    signal = np.fft.ifft(fft_noshift)
    return signal


def kmean_clust_search(X):

    # show mean prototypes
    distortions = []
    inertias = []
    sil = []
    mapping1 = {}
    mapping2 = {}
    max_clusts = 50
    K = range(2, max_clusts)

    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        labels = kmeanModel.labels_

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)

        sil.append(metrics.silhouette_score(X, labels, metric='euclidean'))
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_

    # code from https://anaconda.org/milesgranger/gap-statistic/notebook
    gaps = np.zeros((len(range(2, max_clusts)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    nrefs = 3
    for gap_index, k in enumerate(range(2, max_clusts)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):

            # Create new random reference set
            randomReference = np.random.random_sample(size=X.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(X)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

    # Plus 2 because index of 0 means 2 cluster is optimal
    return distortions, inertias, sil, gaps.argmax()+2, resultsdf


def viz_pca_clustering(X):
    transformed = PCA(2).fit_transform(X)
    for k in range(2, 10):
        km = KMeans(k)
        km.fit(X)
        labels = km.labels_
        for i in range(k):
            mask = labels == i
            plt.scatter(transformed[mask, 0], transformed[mask, 1])
        plt.show()


def viz_prototype_by_clust(n_clust, X, dataset, sequence_map):
    km = KMeans(n_clust)
    km.fit(X)
    labels = km.labels_
    centroids = km.cluster_centers_
    out = np.zeros((X.shape[0], len(centroids), X.shape[1]))
    for i, cam in enumerate(X):
        out[i] = X[i] - centroids
    # so lemme think this out... you're taking the distance between the
    # sequence and the centroid. so then taking the euclidean dist gives
    # you the distance for each row between each centroid. So then you
    # would take an argmin across the final
    row_dist_to_centroid = np.sqrt(np.sum(out ** 2, axis=-1))
    closest_row_to_centroid = np.argmin(row_dist_to_centroid, axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=n_clust)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    for i in range(n_clust):
        mask = labels == i
        # want to find distance each row has to each centroid because you want
        # to find the closest row by centroid
        row_proto_idx = closest_row_to_centroid[i]
        true_idx = sequence_map[row_proto_idx]
        seq = dataset.all_sequences[true_idx][1]
        axes[i].plot(fft_to_ts(seq)[0].ravel())
        axes[i].set_title('cluster {} proto'.format(i))
    plt.show()


def two_d_analytics():
    ards_cam_data = []
    other_cam_data = []
    oned_dat = dataset.ARDSRawDataset.from_pickle('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl', False, 1.0, None, -1, 0.2, 1.0, None, False, False, False)
    oned_dat.butter_filter = None
    dat = dataset.ImgARDSDataset(oned_dat, [], False, True, False, False, False)
    oned_dat.train = False
    dat.train = False
    freqs = np.fft.fftshift(np.fft.fftfreq(224, d=0.02))
    ards_all_img = None
    other_all_img = None
    dev = torch.device('cuda:0')
    target_map = {0: 'non-ards', 1: 'ards'}
    ards_seq_idxs = []
    other_seq_idxs = []
    ards_model_out = []
    other_model_out = []
    ards_kfold_idxs = []
    other_kfold_idxs = []

    file_map = {
        0: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2_only_fft_baseline/model-run-0-epoch2-fold0.pth',
        1: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2_only_fft_baseline/model-run-0-epoch2-fold1.pth',
        2: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2_only_fft_baseline/model-run-0-epoch2-fold2.pth',
        3: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2_only_fft_baseline/model-run-0-epoch2-fold3.pth',
        4: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2_only_fft_baseline/model-run-0-epoch2-fold4.pth',
    }

    for fold in range(5):
        dat.set_kfold_indexes_for_fold(fold)
        model = torch.load(file_map[fold]).to(dev).double()
        # the model is holding onto some kinda state between samples. Its being
        # caused by graph retention. Not retaining the graph seems to cause things
        # to work just fine.
        g = UnNormalizedCam(model)
        #n_samps = len(dat)
        n_samps = len(dat)-500

        for i in range(n_samps):
            kfold_idx = i if n_samps == len(dat) else np.random.randint(0, len(dat))
            idx, seq, _, target = dat[kfold_idx]
            input = torch.tensor(seq).to(dev).unsqueeze(dim=0)
            seq_size = input.shape[2]
            # focus on ground truth
            cam, out = g.generate_cam(input, target=int(target.argmax()))
            cam = cam_process(cam, (seq_size, seq_size), False)
            out_pred = out.argmax()
            # you can do average/median value by frequency. thats a fairly ez thing.
            if out_pred == 1:
                ards_cam_data.append(cam)
                ards_seq_idxs.append(idx)
                ards_model_out.append(out)
                ards_kfold_idxs.append([fold, kfold_idx])
            else:
                other_cam_data.append(cam)
                other_seq_idxs.append(idx)
                other_model_out.append(out)
                other_kfold_idxs.append([fold, kfold_idx])

            if ards_all_img is None and target.argmax() == 1:
                ards_all_img = input
            elif other_all_img is None and target.argmax() == 0:
                other_all_img = input
            elif target.argmax() == 1:
                ards_all_img = torch.cat([ards_all_img, input], dim=0)
            else:
                other_all_img = torch.cat([other_all_img, input], dim=0)

    import seaborn as sns
    sns.set_style('white')
    sns.despine()
    ards_cam_data = np.array(ards_cam_data)
    other_cam_data = np.array(other_cam_data)
    cam_intensities = np.hstack((ards_cam_data.ravel(), other_cam_data.ravel()))
    freq_col = np.hstack([freqs]*(len(ards_cam_data)+len(other_cam_data))*224)
    patho_col = ([1] * (len(ards_cam_data) * 224**2)) + ([0] * (len(other_cam_data) * 224**2))
    cam_data = pd.DataFrame(np.vstack([cam_intensities, freq_col, patho_col]).T, columns=['Cam Intensity', 'Frequency', 'Patho'])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    sns.lineplot(data=cam_data, x='Frequency', y='Cam Intensity', hue='Patho', ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    #axes[0].set_title('Non-ARDS')
    #axes[1].set_title('ARDS')
    plt.xticks(np.arange(-25, 26, 5), fontsize=14)
    #axes[1].set_xticks(np.arange(-25, 26, 5))
    #plt.yticks(np.arange(0, 0.81, 0.1), fontsize=14)
    plt.ylabel('Cam Intensity', fontsize=16)
    plt.xlabel('Frequency', fontsize=16)
    #axes[1].set_yticks(np.arange(0, 0.81, 0.1))
    ax.legend(handles, ['Non-ARDS', 'ARDS'], fontsize=16)
    ax.grid(axis='y')
    plt.xlim((-25.2, 25.2))

    plt.savefig('2d_cam_unnormalized_intensities_ards_non_ards.png', dpi=200)
    plt.close()


def one_d_analytics():
    ards_freq_avgs = np.zeros(224)
    other_freq_avgs = np.zeros(224)
    ards_freq_rows = 0
    other_freq_rows = 0
    ards_cam_data = []
    other_cam_data = []
    dat = dataset.ARDSRawDataset.from_pickle('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl', False, 1.0, None, -1, 0.2, 1.0, None, False, True, False, False)
    dat.butter_filter = None
    dat.train = False
    freqs = np.fft.fftshift(np.fft.fftfreq(224, d=0.02))
    ards_all_img = None
    other_all_img = None
    dev = torch.device('cuda:0')
    target_map = {0: 'non-ards', 1: 'ards'}
    ards_seq_idxs = []
    other_seq_idxs = []
    ards_model_out = []
    other_model_out = []
    ards_kfold_idxs = []
    other_kfold_idxs = []

    file_map = {
        0: 'saved_models/1d_only_fft/1d_only_fft-epoch2-fold0.pth',
        1: 'saved_models/1d_only_fft/1d_only_fft-epoch2-fold1.pth',
        2: 'saved_models/1d_only_fft/1d_only_fft-epoch2-fold2.pth',
        3: 'saved_models/1d_only_fft/1d_only_fft-epoch2-fold3.pth',
        4: 'saved_models/1d_only_fft/1d_only_fft-epoch2-fold4.pth',
    }

    for fold in range(5):
        #dat = dataset.ImgARDSDataset(pkl_dataset, [], False, True, False, False, False)
        dat.set_kfold_indexes_for_fold(fold)
        model = torch.load(file_map[fold]).to(dev).double()
        # the model is holding onto some kinda state between samples. Its being
        # caused by graph retention. Not retaining the graph seems to cause things
        # to work just fine.
        g = UnNormalizedCam(model)
        n_samps = 50

        for i in range(n_samps):
            kfold_idx = i if n_samps == len(dat) else np.random.randint(0, len(dat))
            idx, seq, _, target = dat[kfold_idx]
            input = torch.tensor(seq).to(dev)
            seq_size = input.shape[2]
            # focus on ground truth
            cam, out = g.generate_cam(input, target=int(target.argmax()))
            cam = cam_process(cam, (1, seq_size), True)
            out_pred = out.argmax()
            # you can do average/median value by frequency. thats a fairly ez thing.
            if out_pred == 1:
                ards_cam_data.append(cam)
                ards_seq_idxs.append(idx)
                ards_model_out.append(out)
                ards_kfold_idxs.append([fold, kfold_idx])
            else:
                other_cam_data.append(cam)
                other_seq_idxs.append(idx)
                other_model_out.append(out)
                other_kfold_idxs.append([fold, kfold_idx])

            if ards_all_img is None and target.argmax() == 1:
                ards_all_img = input
            elif other_all_img is None and target.argmax() == 0:
                other_all_img = input
            elif target.argmax() == 1:
                ards_all_img = torch.cat([ards_all_img, input], dim=0)
            else:
                other_all_img = torch.cat([other_all_img, input], dim=0)

    # plot box plots for frequencies.
    #
    # can do plots for frequencies 14 idxs long, so we'll have 16 plots
    # cols should have names like so:
    # val, start freq idx, ards/other
    #
    # Well 14 plots is too squished. maybe 28 will be better
    rows = []
    idx_jump = 14
    for start in range(0, 224, idx_jump):
        end = start + idx_jump
        vals = ards_all_img[:, 0, start:end].cpu().numpy().ravel()
        start_freq = [start] * len(vals)
        y = [1] * len(vals)
        rows.append(np.vstack([vals, start_freq, y]).T)
        vals = other_all_img[:, 0, start:end].cpu().numpy().ravel()
        start_freq = [start] * len(vals)
        y = [0] * len(vals)
        rows.append(np.vstack([vals, start_freq, y]).T)

    import seaborn as sns
    sns.set_style('white')
    sns.despine()
    rows = np.concatenate(rows, axis=0)
    rows = pd.DataFrame(rows, columns=['val', 'freq', 'patho'])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    sns.boxplot(x='freq', y='val', hue='patho', data=rows, palette='Set2', showfliers=False, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Non-ARDS', 'ARDS'], fontsize=16)
    plt.ylabel('')
    plt.xlabel('Frequency Start', fontsize=16)
    plt.xticks(range(0, int(224/idx_jump)), [
        '{}'.format(round(freqs[start],1)) for start in range(0, 224, idx_jump)
    ], fontsize=14)
    plt.yticks(np.arange(-4, 5), fontsize=14)
    ax.grid(axis='y')
    plt.savefig('fft_freq_box_ards_non_ards.png', dpi=200)
    plt.close()

    sns.set_style('white')
    sns.despine()
    cam_intensities = np.hstack(ards_cam_data+other_cam_data).ravel()
    freq_col = np.hstack([freqs]*(len(ards_cam_data)+len(other_cam_data)))
    patho_col = ([1] * len(ards_cam_data) * 224) + ([0] * len(other_cam_data) * 224)
    cam_data = pd.DataFrame(np.vstack([cam_intensities, freq_col, patho_col]).T, columns=['Cam Intensity', 'Frequency', 'Patho'])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    sns.lineplot(data=cam_data, x='Frequency', y='Cam Intensity', hue='Patho', ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    #axes[0].set_title('Non-ARDS')
    #axes[1].set_title('ARDS')
    plt.xticks(np.arange(-25, 26, 5), fontsize=14)
    #axes[1].set_xticks(np.arange(-25, 26, 5))
    plt.yticks(np.arange(0, 0.81, 0.1), fontsize=14)
    plt.ylabel('Cam Intensity', fontsize=16)
    plt.xlabel('Frequency', fontsize=16)
    #axes[1].set_yticks(np.arange(0, 0.81, 0.1))
    ax.legend(handles, ['Non-ARDS', 'ARDS'], fontsize=16)
    ax.grid(axis='y')
    plt.xlim((-25.2, 25.2))
    plt.savefig('1d_cam_intensities_ards_non_ards.png', dpi=200)
    plt.close()
    # show mean prototypes
#    ards_freq_avgs = np.nanmean(np.array(ards_cam_data), axis=0).ravel()
#    other_freq_avgs = np.nanmean(np.array(other_cam_data), axis=0).ravel()
#    ards_min_seq = ards_seq_idxs[np.sum((np.array(ards_cam_data) - ards_freq_avgs) ** 2, axis=0).argmin()]
#    other_min_seq = other_seq_idxs[np.sum((np.array(other_cam_data) - other_freq_avgs) ** 2, axis=0).argmin()]
#    ards_seq = fft_to_ts(dat.all_sequences[ards_min_seq][1])
#    other_seq = fft_to_ts(dat.all_sequences[other_min_seq][1])
#    thresh, seq_idx = .5, 0
#    rand_idx = np.random.randint(0, 20)
#
#    fig, axes = plt.subplots(nrows=2, ncols=2)
#    fig.set_figheight(10)
#    fig.set_figwidth(16)
#    axes[0][0].plot(ards_freq_avgs)
#    axes[0][0].set_title('ARDS mean cam')
#    axes[0][1].plot(other_freq_avgs)
#    axes[0][1].set_title('Non-ARDS mean cam')
#    axes[1][0].plot(ards_seq[rand_idx].ravel())
#    axes[1][0].set_title('ARDS mean prototype')
#    axes[1][1].plot(other_seq[rand_idx].ravel())
#    axes[1][1].set_title('Non-ARDS mean prototype')
#    plt.show()

    # perform kmeans on the prototype
    #
    # Elbow model is inconclusive, silhouette method tells us 2 clusters is best,
    # and gap stat says that infinitely more clusters is better.
#    X = np.array(ards_cam_data)
#    ards_dist, ards_inertias, ards_sil, ards_gap_optim, ards_gapdf = kmean_clust_search(X)
#    X = np.array(other_cam_data)
#    other_dist, other_inertias, other_sil, other_gap_optim, other_gapdf = kmean_clust_search(X)
#    fig, axes = plt.subplots(nrows=1, ncols=6)
#    fig.set_figheight(10)
#    fig.set_figwidth(16)
#    axes[0].plot(ards_dist)
#    axes[1].plot(ards_sil)
#    axes[2].plot(ards_gapdf.gap)
#    axes[2].set_title('ARDS Gap Stat')
#    axes[3].plot(other_dist)
#    axes[4].plot(other_sil)
#    axes[5].plot(other_gapdf.gap)
#    axes[5].set_title('Other Gap Stat')
#    plt.show()

    # lets just try 5 protos
    #X = np.array(ards_cam_data)
    #viz_prototype_by_clust(5, X, dat, ards_seq_idxs)
    #X = np.array(other_cam_data)
    #viz_prototype_by_clust(5, X, dat, other_seq_idxs)

    # I want to try splicing frequencies from one sequence to another. So
    # basically I would take one item that does very well in one area. ARDS
    # for example, and then try to nudge another item into ARDS by splicing the ARDS
    # frequency into the other item. If it works then I guess I look to see
    # what changed in the waveform that made it look special
    #
    # lets try getting all freqs after 15Hz
    #
    # Hmm... so initial results werent that promising. it doesnt work that
    # frequently. But I wonder if we chose frequencies that were more in
    # line with the ARDS average frequencies rather than just any item that
    # was confidently predicted as ARDS if things would be different
    #
    # This is actually interesting. because splicing doesnt seem to work with the
    # 15Hz frequency bands that I tried. That is supposedly where ARDS gradcam is looking
    # tho. Oh... i think I kno why. Its because ARDS is probably looking negatively
    # at the lower frequencies. Whereas for other its looking positively. the neg.
    # association is being filtered by relu tho. The high frequencies are a nice
    # fine tuning point for the network so that they can slightly improve AUC/accuracy
    # but the main thing is that the lower frequencies will never be out-dominated
    # by the higher frequencies.
    #
    #
    # what about a sanity check where you set ARDS pred target  high frequencies
    # to 0 and then see what the cam mask looks like.
    # OK, so setting frequencies to 0 does very little. So just more confirmation for
    # the theory that it is a mix of high+low frequency data, and that the high freq
    # data can serve as a tipping point.
    freq_mask = np.abs(freqs) >= 15
    num_mask = np.argwhere(freq_mask).ravel()
    for i, item in enumerate(ards_model_out):

        if F.softmax(item)[0, 1] > .95:
            fold_n, idx = ards_kfold_idxs[i]
            dat.set_kfold_indexes_for_fold(fold_n)
            _, seq, __, ___ = dat[idx]
            ards_cam, ards_out = g.generate_cam(torch.tensor(seq).to(dev), target=1)
            # select random other item
            other_fold_n, other_idx = random.choice(other_kfold_idxs)
            dat.set_kfold_indexes_for_fold(other_fold_n)
            _, other_seq, __, ___ = dat[other_idx]

            cam_before, out_before = g.generate_cam(torch.tensor(other_seq).to(dev), target=0)
            other_before = other_seq.copy()
            other_seq[:, :, num_mask] = seq[:, :, num_mask]
            other_seq = torch.tensor(other_seq).to(dev)

            cam_after, out_after = g.generate_cam(other_seq, target=0)

            import IPython; IPython.embed()
            # I mean how would I show the high frequency part. One idea is splicing
            # between ARDS and OTHER / OTHER and ARDS. Then visual comparison on
            # how things change for each breath. Another option is talking to Jason
            # about what it possibly means, and what the small fluctuations could be
            # doing.

    binary_thresh = True
    if binary_thresh:
        ards_mask = ards_freq_avgs >= thresh
        other_mask = other_freq_avgs >= thresh
        fourth_plot_title = 'threshold mask val={}'.format(thresh)
    else:
        ards_mask = ards_freq_avgs
        other_mask = other_freq_avgs
        fourth_plot_title = 'avg cam value mask'

    # gotta modify this so that frequency is a value and the columns are <frequency, intensity>
    # so we need an input sequence to visualize. I mean the q is where to get one. I
    # guess we could just select randomly from the ground truth based on patho
    # downside of all this is that it kinda ignores whether the classifier would be
    # looking at these spots for the specific image to begin with
    #
    # So this experiment has pretty much showed me the same thing. That ARDS fft info
    # ends up showing a fairly well re-created sequence of VWD. other fft avg'ing shows
    # us an oscillating waveform. I mean I think this is just one of those things
    # that had some promise in theory, but not so much IRL
    fig, axes = plt.subplots(nrows=1, ncols=4)
    fig.set_figheight(10)
    fig.set_figwidth(16)

    gt = dat._get_all_sequence_ground_truth()
    rand_idx = np.random.choice(range(0, 20))
    ards_seq_idx = np.random.choice(gt[gt.y == 1].index)
    ards_seq = dat.all_sequences[ards_seq_idx][1]
    fft = get_fft(ards_seq)
    masked_signal = fft_to_ts_with_mask(ards_seq, ards_mask)[rand_idx]


def one_two_d_comparison():
    oned_freq_rows = 0
    twod_freq_rows = 0
    oned_cam_data = []
    twod_cam_data = []
    oned_dat = dataset.ARDSRawDataset.from_pickle('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl', False, 1.0, None, -1, 0.2, 1.0, None, False, False, False)
    twod_dat = dataset.ImgARDSDataset(oned_dat, [], False, False, False, False, False)
    oned_dat.butter_filter = None
    oned_dat.train = False
    twod_dat.train = False
    oned_all_img = []
    twod_all_img = []
    dev = torch.device('cuda:0')
    target_map = {0: 'non-ards', 1: 'ards'}
    # part of the problem here will be how to match 1d data to 2d data because of how they handle things differently. It will
    # likely come down to finding a way to match relative patient sequences to each twod.
    oned_seq_idxs = []
    twod_seq_idxs = []
    oned_kfold_idxs = []
    twod_kfold_idxs = []

    oned_file_map = {
        0: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear/model-run-1-fold0.pth',
        1: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear/model-run-1-fold1.pth',
        2: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear/model-run-1-fold2.pth',
        3: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear/model-run-1-fold3.pth',
        4: 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear/model-run-1-fold4.pth',
    }
    twod_file_map = {
        0:'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2/model-run-1-fold0.pth',
        1:'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2/model-run-1-fold1.pth',
        2:'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2/model-run-1-fold2.pth',
        3:'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2/model-run-1-fold3.pth',
        4:'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2/model-run-1-fold4.pth',
    }

    for dim in range(1, 3):
        if dim == 1:
            file_map = oned_file_map
            #n_samps_per_pt = 11
            n_samps_per_pt = 1
            dat = oned_dat
        else:
            file_map = twod_file_map
            n_samps_per_pt = 1
            dat = twod_dat

        for fold in range(5):
            dat.set_kfold_indexes_for_fold(fold)
            gt = dat.get_ground_truth_df()
            avail_idxs = gt.groupby('patient').head(n_samps_per_pt)
            model = torch.load(file_map[fold]).to(dev).double()
            # the model is holding onto some kinda state between samples. Its being
            # caused by graph retention. Not retaining the graph seems to cause things
            # to work just fine.
            g = MaxMinNormCam(model)

            # avail_idxs.index should be the kfold_index, but its dying.
            for kfold_idx in avail_idxs.index:
                # the problem with random indices is that you will not be guaranteed
                # a comparison sequences from 1d -> 2d. So maybe picking first n
                # sequences per patient is better
                idx = np.where(dat.kfold_indexes == kfold_idx)[0][0]
                idx, seq, _, target = dat[idx]
                t_int = int(target.argmax())
                seq = torch.tensor(seq).to(dev)
                seq_size = seq.shape[2]
                n_iters = 20 if dim == 1 else 1
                # focus on ground truth
                for breath_n in range(n_iters):
                    if dim == 1:
                        input = seq[breath_n:breath_n+1]
                        input = input.repeat((20, 1, 1))
                    elif dim == 2:
                        input = seq.unsqueeze(dim=0)
                    cam, out = g.generate_cam(input, target=t_int)
                    if dim == 1:
                        cam = cam_process(cam, (1, seq_size), True)
                        oned_cam_data.append(cam)
                        oned_seq_idxs.append((idx, breath_n))
                        oned_kfold_idxs.append([fold, kfold_idx, breath_n])
                        oned_all_img.append(input[0:1])
                    elif dim == 2:
                        cam = cam_process(cam, (seq_size, seq_size), True)
                        twod_cam_data.append(cam)
                        twod_seq_idxs.append(idx)
                        twod_kfold_idxs.append([fold, kfold_idx, breath_n])
                        twod_all_img.append(input)

    oned_all_img = torch.cat(oned_all_img, dim=0)
    twod_all_img = torch.cat(twod_all_img, dim=0)

    # need to compare 1d to 2d seqs now

    # for now this works because its just 1-1 in 1d/2d seqs. but not always the case
    for i, (idx, breath_n) in enumerate(oned_seq_idxs):

        # XXX twod_seq_match, twod_idx_match

        # ok so i have the oned idx and i need to find a way of matching that to
        # the 2d idx
        fold, kfold_idx, _ = oned_kfold_idxs[i]
        oned_dat.set_kfold_indexes_for_fold(fold)
        twod_dat.set_kfold_indexes_for_fold(fold)
        pt, _, target, __ = oned_dat.all_sequences[idx]
        target_name = {0: 'non-ards', 1: 'ards'}[np.argmax(target,axis=0)]
        # ok so now i have my abs idx, kfold idx, pt, and fold. So what else does it
        # take to get from here.
        #
        # I mean one way would be to actually create a memory construct in
        # ImgARDSDataset so you can find it easily. But then creating the memory
        # construct requires more complexity than I want to put into ImgARDSDataset
        twod_seq_match = twod_seq_idxs[i // 20]
        twod_idx_match = i // 20

        oned_seq = oned_dat[idx][1][breath_n].ravel()
        twod_seq = twod_dat[twod_seq_match][1][:, breath_n].cpu().numpy().ravel()
        oned_cam = oned_cam_data[i]
        twod_cam = twod_cam_data[twod_idx_match][breath_n]

        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].plot(oned_seq)
        axes[1].plot(twod_seq)
        axes[0].scatter(np.arange(224), oned_seq, c=oned_cam, vmin=0, vmax=1)
        axes[1].scatter(np.arange(224), twod_seq, c=twod_cam, vmin=0, vmax=1)
        axes[0].set_title('1d')
        axes[1].set_title('2d')
        plt.suptitle('{} sequence i-{} breath_n-{}'.format(target_name, idx, breath_n))
        plt.savefig('img1d2d/i{}-b{}.png'.format(idx, breath_n), dpi=200)


def butterworth_1d_analytics(dataset_path, file_map, experiment, hz_low, hz_high, n_samps):
    ards_cam_data = []
    other_cam_data = []
    dat = dataset.ARDSRawDataset.from_pickle(
        dataset_path, False, 1.0, None, -1, 0.2, 1.0, hz_low, hz_high,
        False, False, False, False, False
    )
    dat_no_filter = dataset.ARDSRawDataset.from_pickle(
        dataset_path, False, 1.0, None, -1, 0.2, 1.0, None, None,
        False, False, False, False, False
    )
    dat.train = False
    dat_no_filter.train = False

    ards_all_img = None
    other_all_img = None
    dev = torch.device('cuda:0')
    target_map = {0: 'non-ARDS', 1: 'ARDS'}
    ards_seq_idxs = []
    other_seq_idxs = []
    ards_model_out = []
    other_model_out = []
    ards_kfold_idxs = []
    other_kfold_idxs = []

    for fold in range(5):
        dat.set_kfold_indexes_for_fold(fold)
        model = torch.load(file_map[fold], map_location=dev)
        # the model is holding onto some kinda state between samples. Its being
        # caused by graph retention. Not retaining the graph seems to cause things
        # to work just fine.
        g = MaxMinNormCam(model)

        for i in range(n_samps):
            kfold_idx = i if n_samps == len(dat) else np.random.randint(0, len(dat))
            idx, seq, _, target = dat[kfold_idx]
            input = torch.tensor(seq).to(dev).float()
            seq_size = input.shape[2]
            # focus on ground truth
            cam, out = g.generate_cam(input, target=int(target.argmax()))
            cam = cam_process(cam, (1, seq_size), True)
            out_pred = out.argmax()
            # you can do average/median value by frequency. thats a fairly ez thing.
            if out_pred == 1:
                ards_cam_data.append(cam)
                ards_seq_idxs.append(idx)
                ards_model_out.append(out)
                ards_kfold_idxs.append([fold, kfold_idx])
            else:
                other_cam_data.append(cam)
                other_seq_idxs.append(idx)
                other_model_out.append(out)
                other_kfold_idxs.append([fold, kfold_idx])

            if ards_all_img is None and target.argmax() == 1:
                ards_all_img = input
            elif other_all_img is None and target.argmax() == 0:
                other_all_img = input
            elif target.argmax() == 1:
                ards_all_img = torch.cat([ards_all_img, input], dim=0)
            else:
                other_all_img = torch.cat([other_all_img, input], dim=0)

    import seaborn as sns
    cam_intensities = np.vstack(ards_cam_data+other_cam_data).ravel()
    freq_col = np.hstack(list(range(224))*(len(ards_cam_data)+len(other_cam_data)))
    patho_col = ([1] * len(ards_cam_data) * 224) + ([0] * len(other_cam_data) * 224)
    cam_data = pd.DataFrame(np.vstack([cam_intensities, freq_col, patho_col]).T, columns=['Cam Intensity', 'Frequency', 'Patho'])
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    sns.lineplot(data=cam_data, x='Frequency', y='Cam Intensity', hue='Patho', ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    #axes[0].set_title('Non-ARDS')
    #axes[1].set_title('ARDS')
    #axes[1].set_xticks(np.arange(-25, 26, 5))
    plt.ylabel('Cam Intensity', fontsize=16)
    plt.xlabel('')
    #axes[1].set_yticks(np.arange(0, 0.81, 0.1))
    ax.legend(handles, ['Non-ARDS', 'ARDS'], fontsize=16)
    ax.grid(axis='y')
    plt.title('{}-{}Hz Gradcam'.format(hz_low, hz_high), fontsize=18)
    plt.savefig('../img/{}-{}-{}hz-gradcam.png'.format(experiment, hz_low, hz_high), dpi=200)
    plt.close()

    # look for representative waveforms
    ards_cam_data = np.squeeze(np.array(ards_cam_data))
    other_cam_data = np.squeeze(np.array(other_cam_data))

    ards_avgs = np.nanmean(ards_cam_data, axis=0).ravel()
    other_avgs = np.nanmean(other_cam_data, axis=0).ravel()

    ards_rel_min_idx = np.sum((ards_cam_data - ards_avgs) ** 2, axis=1).argmin()
    other_rel_min_idx = np.sum((other_cam_data - other_avgs) ** 2, axis=1).argmin()

    ards_kfold, ards_kfold_idx = ards_kfold_idxs[ards_rel_min_idx]
    other_kfold, other_kfold_idx = other_kfold_idxs[other_rel_min_idx]

    dat.set_kfold_indexes_for_fold(ards_kfold)
    ards_seq = dat[ards_kfold_idx][1]
    mu, std = dat.scaling_factors[ards_kfold]
    ards_seq = (ards_seq * std) + mu

    dat_no_filter.set_kfold_indexes_for_fold(ards_kfold)
    ards_seq_no_filt = dat_no_filter[ards_kfold_idx][1]
    mu, std = dat_no_filter.scaling_factors[ards_kfold]
    ards_seq_no_filt = (ards_seq_no_filt * std) + mu

    dat.set_kfold_indexes_for_fold(other_kfold)
    other_seq = dat[other_kfold_idx][1]
    mu, std = dat.scaling_factors[other_kfold]
    other_seq = (other_seq * std) + mu

    dat_no_filter.set_kfold_indexes_for_fold(other_kfold)
    other_seq_no_filt = dat_no_filter[other_kfold_idx][1]
    mu, std = dat_no_filter.scaling_factors[other_kfold]
    other_seq_no_filt = (other_seq_no_filt * std) + mu

    # I should rly consider doing things breath by breath. Probably for the final paper.
    rand_idx = np.random.randint(0, 20)

    sns.set_style('white')
    cmap = cm.get_cmap('Set2')
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes_ards_proto = axes[0][0].twinx()
    axes_other_proto = axes[0][1].twinx()
    fig.set_figheight(10)
    fig.set_figwidth(20)

    axes_ards_proto.plot(np.median(ards_seq, axis=0).ravel(), color=cmap.colors[2], lw=2, label='Prototype')
    axes_ards_proto.yaxis.tick_right()
    axes_ards_proto.set_ylabel('Flow', color=cmap.colors[2])
    axes_ards_proto.tick_params(axis='y', labelcolor=cmap.colors[2])

    axes_other_proto.plot(np.median(other_seq, axis=0).ravel(), color=cmap.colors[3], lw=2, label='Prototype')
    axes_other_proto.yaxis.tick_right()
    axes_other_proto.set_ylabel('Flow', color=cmap.colors[3])
    axes_other_proto.tick_params(axis='y', labelcolor=cmap.colors[3])

    axes[0][0].plot(ards_avgs, color=cmap.colors[0], lw=3, label='Mean Cam', alpha=.6)
    axes[0][0].set_title('ARDS')
    axes[0][0].grid(axis='y')
    axes[0][0].set_ylabel('Cam Intensity', color=cmap.colors[0])
    axes[0][0].tick_params(axis='y', labelcolor=cmap.colors[0])

    axes[0][1].plot(other_avgs, color=cmap.colors[0], lw=3, label='Mean Cam', alpha=.6)
    axes[0][1].set_title('Non-ARDS')
    axes[0][1].grid(axis='y')
    axes[0][1].tick_params(axis='y', labelcolor=cmap.colors[0])

    c = cmap.colors[6]
    axes[1][0].plot(np.median(ards_seq_no_filt, axis=0).ravel(), color=c, lw=2, label='Prototype No Filter')
    axes[1][0].set_ylabel('Flow', color=c)
    axes[1][0].tick_params(axis='y', labelcolor=c)
    axes[1][0].grid(axis='y')
    axes[1][0].legend(loc='upper right')

    c = cmap.colors[1]
    axes[1][1].plot(np.median(other_seq_no_filt, axis=0).ravel(), color=c, lw=2, label='Prototype No Filter')
    axes[1][1].set_ylabel('Flow', color=c)
    axes[1][1].tick_params(axis='y', labelcolor=c)
    axes[1][1].grid(axis='y')
    axes[1][1].legend(loc='upper right')

    lines, labels = axes[0][0].get_legend_handles_labels()
    lines2, labels2 = axes_ards_proto.get_legend_handles_labels()
    axes_ards_proto.legend(lines + lines2, labels + labels2, loc='upper right')

    lines, labels = axes[0][1].get_legend_handles_labels()
    lines2, labels2 = axes_other_proto.get_legend_handles_labels()
    axes_other_proto.legend(lines + lines2, labels + labels2, loc='upper right')

    #axes[1][1].set_title('Non-ARDS mean prototype')
    #axes[1][1].grid(axis='y')
    plt.suptitle('{}-{}Hz Cam and Mean Prototypes'.format(hz_low, hz_high), fontsize=18)
    plt.savefig('../img/{}-{}-{}hz-prototypes.png'.format(experiment, hz_low, hz_high), dpi=200)


def remove_spines(ax):
    spines = ['top', 'left', 'right', 'bottom']
    for s in spines:
        ax.spines[s].set_visible(False)


def butter_plots(dataset_path, index, experiment, hz_low, hz_high, color, do_baseline=False):
    """
    Do a butterworth plot for a given index of data.
    """
    dat_no_filter = dataset.ARDSRawDataset.from_pickle(
        dataset_path, False, 1.0, None, -1, 0.2, 1.0, None, None,
        False, False, False, False, False
    )

    if hz_low == 0:
        sos = butter(10, hz_high, fs=50, output='sos', btype='lowpass')
    elif hz_high == 25:
        sos = butter(10, hz_low, fs=50, output='sos', btype='highpass')
    else:
        wn = (hz_low, hz_high)
        sos = butter(10, wn, fs=50, output='sos', btype='bandpass')

    dat_no_filter.set_kfold_indexes_for_fold(0)
    rand_breath_idx = np.random.randint(0, 20)
    seq = dat_no_filter[index][1]

    signal = sosfilt(sos, seq[rand_breath_idx].ravel())
    plt.plot(signal, color=color, lw=1.35, label='flow')
    plt.grid(axis='y')
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    fig.axes[0].xaxis.set_ticklabels([])
    fig.axes[0].yaxis.set_ticklabels([])
    remove_spines(fig.axes[0])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
    plt.savefig('../img/butterworth-plt-{}-idx{}-{}-{}hz.png'.format(experiment, index, hz_low, hz_high), dpi=400, bbox_inches='tight', pad_inches=0.0)
    plt.close()


if __name__ == "__main__":
    """
    This runs the frequency exploration experiment
    """
    experiment_map = {
        'experiment_files_unpadded_centered_nb20_cnn_linear_butter': ('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl', 4000),
        'experiment_files_padded_breath_by_breath_cnn_linear_butter': ('/fastdata/deepards/padded_breath_by_breath_nb20-kfold.pkl', 4000),
    }
    i = 0
    cmap = ['darkviolet', 'mediumvioletred', 'royalblue']
    for experiment in ['experiment_files_padded_breath_by_breath_cnn_linear_butter']:
        #for hz_low, hz_high in [(0, 5)]:
        for classification in [0, 1, 1]:
            dataset_path, n_samps = experiment_map[experiment]
            dat_no_filter = dataset.ARDSRawDataset.from_pickle(
                dataset_path, False, 1.0, None, -1, 0.2, 1.0, None, None,
                False, False, False, False, False
            )
            dat_no_filter.set_kfold_indexes_for_fold(0)
            gt = dat_no_filter.get_ground_truth_df()
            idx = np.random.choice(gt[gt.y == classification].index)
            idx = list(dat_no_filter.kfold_indexes).index(idx)

            for hz_low, hz_high in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:
                if hz_low == 0:
                    butter_plots(dataset_path, idx, experiment, hz_low, hz_high, cmap[i], do_baseline=True)
                else:
                    butter_plots(dataset_path, idx, experiment, hz_low, hz_high, cmap[i])
            i+= 1
    # XXX debug
    import sys
    sys.exit(0)

    #for experiment in ['experiment_files_unpadded_centered_nb20_cnn_linear_butter', 'experiment_files_padded_breath_by_breath_cnn_linear_butter']:
    for experiment in ['experiment_files_padded_breath_by_breath_cnn_linear_butter']:
        #for hz_low, hz_high in [(0, 5)]:
        for hz_low, hz_high in [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]:

            file_map = {
                0: 'saved_models/{}_{}_{}hz/model-run-0-fold0.pth'.format(experiment, hz_low, hz_high),
                1: 'saved_models/{}_{}_{}hz/model-run-0-fold1.pth'.format(experiment, hz_low, hz_high),
                2: 'saved_models/{}_{}_{}hz/model-run-0-fold2.pth'.format(experiment, hz_low, hz_high),
                3: 'saved_models/{}_{}_{}hz/model-run-0-fold3.pth'.format(experiment, hz_low, hz_high),
                4: 'saved_models/{}_{}_{}hz/model-run-0-fold4.pth'.format(experiment, hz_low, hz_high),
            }
            dataset_path, n_samps = experiment_map[experiment]
            butterworth_1d_analytics(dataset_path, file_map, experiment, hz_low, hz_high, n_samps)
