"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import csv
from pathlib import Path

import numpy as np
import torch
import argparse
import pickle
import cv2
import torch.nn.functional as F


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
        #print(x.shape)
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        # they forgot relu when they were doing this initially. damn
        x = F.relu(x)
        # pool and flatten
        try:
            x = self.model.breath_block.avgpool(x).view(-1)
        except:
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
        return torch.sum(one_hot.cuda().double() * output)


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


def cam_process(cam, seq_size):
    cam = cv2.resize(cam, (seq_size, seq_size))
    # process cam to 0-1
    cam -= cam.min()
    cam = cam / cam.max()
    return cam


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    from deepards import dataset
    filename = 'saved_models/2d_only_fft/2d_only_fft-epoch1-fold0.pth'
    pkl_dataset = pd.read_pickle('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl')
    dev = torch.device('cuda:0')
    dat = dataset.ImgARDSDataset(pkl_dataset, [], False, True, False, False, False)
    dat.train = False
    dat.set_kfold_indexes_for_fold(0)
    target_map = {0: 'non-ards', 1: 'ards'}
    model = torch.load(filename).to(dev).double()
    # the model is holding onto some kinda state between samples. Its being
    # caused by graph retention. Not retaining the graph seems to cause things
    # to work just fine.
    g = MaxMinNormCam(model)

    ards_freq_avgs = np.zeros(224)
    other_freq_avgs = np.zeros(224)
    ards_freq_rows = 0
    other_freq_rows = 0
    freqs = np.fft.fftshift(np.fft.fftfreq(224, d=0.02))
    n_samps = len(dat)
    ards_all_img = None
    other_all_img = None
    print('Gathering all {} samples'.format(n_samps))

    for i in range(n_samps):
        #idx = np.random.randint(0, len(dat))
        idx, seq, _, target = dat[i]
        target_name = target_map[int(target.argmax())]
        input = seq.unsqueeze(dim=0).to(dev)
        seq_size = input.shape[2]
        # focus on ground truth
        cam, out = g.generate_cam(input, target=int(target.argmax()))
        cam = cam_process(cam, seq_size)
        # you can do average/median value by frequency. thats a fairly ez thing.
        for row in cam:
            if target_name == 'ards':
                ards_freq_avgs += row
                ards_freq_rows += 1
            else:
                other_freq_avgs += row
                other_freq_rows += 1

        if ards_all_img is None and target.argmax() == 1:
            ards_all_img = input
        elif other_all_img is None and target.argmax() == 0:
            other_all_img = input
        elif target.argmax() == 1:
            ards_all_img = torch.cat([ards_all_img, input], dim=0)
        else:
            other_all_img = torch.cat([other_all_img, input], dim=0)

        # XXX do this for now b/c testing my averaging theory
        continue

        output_dir = Path('gradcam_results/tmp_storage/')
        img = img_process(input)

        with open(output_dir.joinpath('gradcam2d-seq-{}-{}.csv'.format(target_name, i)), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(seq.squeeze().cpu().numpy().tolist())

        fig, axes = plt.subplots(nrows=1, ncols=5)
        fig.set_figheight(10)
        fig.set_figwidth(16)

        ax = plt.subplot(1, 5, 1)
        ax.imshow(np.rollaxis(img, 0, 3), aspect='auto')
        ax.set_title('orig')

        ax = plt.subplot(1, 5, 2)
        ax.imshow(cam, cmap='inferno', aspect='auto')
        ax.set_title('cam')

        ax = plt.subplot(1, 5, 3)
        ax.imshow(np.rollaxis(img, 0, 3), aspect='auto')
        ax.imshow(cam, cmap='inferno', alpha=0.6, aspect='auto')
        ax.set_title('cam overlay')

        ax = plt.subplot(1, 5, 4)
        ax.plot(freqs, freq_avgs)
        ax.set_title('average frequency')

        plt.suptitle('patient: {}, target: {}, pred: {}'.format(dat.all_sequences[idx][0], target.argmax(), F.softmax(out).argmax()))

        # i want to use this as a compression/visualization algo.
        # I guess that i can probably use a threshold on the cam. and then 0 out the
        # coefficients below a certain value. kinda similar to how jpeg compression
        # works. Anyways, this will probably enable us to clarify what exactly is
        # being focused on.

        # I mean whats the deal here? what do we do to go backwards? So I do this and
        #
        # Yeah so this doesnt actually do anything. It just calculates an avg cam val
        # and has nothing to actually do with the fft vals.
        #
        # So you can do a few things still. you can just use the thresholding or you
        # can use the avg cam vals as a mask. You could even apply some kind of
        # non-linearity to it if you wanted. well either way you create a mask. one
        # is a binary thresh mask the other is a fractional averaging/median mask
        #
        # apparently masks cannot have absolute value or you are screwed.
        #
        # Hmm... I think one thing i notice here is that the ARDS recon signals are
        # much cleaner compared to the OTHER recon signals. I think this is because ARDS
        # tends to focus more on the center of the picture where a lot of the important
        # frequency information is. OTHER tends to focus on the left hand side of the img.
        # there's a lot of fast oscillation component here. Its quite possible that
        # the fast oscillation component is just capturing a greater degree of
        # spontaneous breath. Or it could also be capturing elements of the COPD type
        # as well

        # well i think the way this is showing is pretty marginal. It might have
        # some more use if we can average the cam contributions across many images
        # and then showcase overall what the cam is looking at.

        #plt.savefig(output_dir.joinpath('gradcam2d-{}-{}.png'.format(target_name, i)), dpi=200)
        #plt.show()

    ards_freq_avgs /= ards_freq_rows
    other_freq_avgs /= other_freq_rows
    thresh, seq_idx = .4, 0

    binary_thresh = True
    if binary_thresh:
        ards_mask = ards_freq_avgs >= thresh
        other_mask = other_freq_avgs >= thresh
        fourth_plot_title = 'threshold mask val={}'.format(thresh)
    else:
        ards_mask = ards_freq_avgs
        other_mask = other_freq_avgs
        fourth_plot_title = 'avg cam value mask'

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
        vals = ards_all_img[:, 0, :, start:end].cpu().numpy().ravel()
        start_freq = [start] * len(vals)
        y = [1] * len(vals)
        rows.append(np.vstack([vals, start_freq, y]).T)
        vals = other_all_img[:, 0, :, start:end].cpu().numpy().ravel()
        start_freq = [start] * len(vals)
        y = [0] * len(vals)
        rows.append(np.vstack([vals, start_freq, y]).T)

    import seaborn as sns
    rows = np.concatenate(rows, axis=0)
    rows = pd.DataFrame(rows, columns=['val', 'freq', 'patho'])
    ax = sns.boxplot(x='freq', y='val', hue='patho', data=rows, palette='Set3', showfliers=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Non-ARDS', 'ARDS'])
    plt.ylabel('')
    plt.xlabel('Frequency Start')
    plt.xticks(range(0, int(224/idx_jump)), [
        '{}'.format(round(freqs[start],1)) for start in range(0, 224, idx_jump)
    ])
    plt.show()

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
    ards_seq = dat.all_sequences[ards_seq_idx][1][rand_idx]

    real = ards_seq[:, 0]
    imag = ards_seq[:, 1]
    fft_with_shift = real + (1j*imag)
    fft_noshift = np.fft.ifftshift(fft_with_shift*ards_mask)
    signal = np.fft.ifft(fft_noshift)

    ax = plt.subplot(1, 4, 1)
    ax.plot(freqs, ards_freq_avgs)
    ax.set_title('avg freq')

    ax = plt.subplot(1, 4, 2)
    ax.plot(np.fft.ifft(np.fft.ifftshift(fft_with_shift)).real)
    ax.set_title('orig VWD')

    ax = plt.subplot(1, 4, 3)
    ax.plot(freqs, ards_seq[:, 0].ravel())
    ax.set_title('FFT sequence')

    ax = plt.subplot(1, 4, 4)
    ax.plot(signal)
    ax.set_title(fourth_plot_title)

    plt.suptitle('ARDS masking')
    plt.show()
    #plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=4)
    fig.set_figheight(10)
    fig.set_figwidth(16)

    other_seq_idx = np.random.choice(gt[gt.y == 0].index)
    other_seq = dat.all_sequences[other_seq_idx][1][rand_idx]

    real = other_seq[:, 0]
    imag = other_seq[:, 1]
    fft_with_shift = real + (1j*imag)
    fft_noshift = np.fft.ifftshift(fft_with_shift*other_mask)
    signal = np.fft.ifft(fft_noshift)

    ax = plt.subplot(1, 4, 1)
    ax.plot(freqs, other_freq_avgs)
    ax.set_title('avg freq')

    ax = plt.subplot(1, 4, 2)
    ax.plot(np.fft.ifft(np.fft.ifftshift(fft_with_shift)).real)
    ax.set_title('orig VWD')

    ax = plt.subplot(1, 4, 3)
    ax.plot(freqs, other_seq[:, 0].ravel())
    ax.set_title('FFT sequence')

    ax = plt.subplot(1, 4, 4)
    ax.plot(signal)
    ax.set_title(fourth_plot_title)

    plt.suptitle('OTHER masking')
    plt.show()
