"""
Created on Thu Oct 26 11:06:51 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
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
        except torch.nn.modules.module.ModuleAttributeError:
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
        output.backward(retain_graph=True)
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
    #print(len(data))
    first_seq = None
    count = 0
    for i, d in enumerate(data.all_sequences):
        if d[2][1] == ards:
            if count == c:
                first_seq = d
                break
            count = count + 1
            continue
    #first_seq = data[1]
    #print(first_seq[2])
    br = first_seq[1]
    #br = np.expand_dims(br, 0)
    #br = torch.from_numpy(br)
    br = torch.FloatTensor(br).cuda()
    return br


def img_process(model_in):
    img = model_in.squeeze().cpu().numpy()
    # process img to 0-1
    img -= img.min()
    img = img / img.max()
    return img


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
    filename = 'saved_models/experiment_files_unpadded_centered_nb20_cnn_linear_2d_bs2/model-run-0-epoch10-fold0.pth'
    pkl_dataset = pd.read_pickle('/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl')
    dev = torch.device('cuda:0')
    model = torch.load(filename).to(dev).double()
    g = MaxMinNormCam(model)
    dat = dataset.ImgARDSDataset(pkl_dataset, [])
    dat.train = False
    dat.set_kfold_indexes_for_fold(0)
    idx = np.random.randint(0, len(dat))

    idx, seq, _, target = dat[idx]
    input = seq.unsqueeze(dim=0).to(dev)
    cam, out = g.generate_cam(input)
    seq_size = input.shape[2]
    img = img_process(input)
    cam = cam_process(cam, seq_size)
    fig, axes = plt.subplots(nrows=1, ncols=3)

    ax = plt.subplot(1, 3, 1)
    ax.imshow(img)
    ax.set_title('orig')

    ax = plt.subplot(1, 3, 2)
    ax.imshow(cam, cmap='inferno')
    ax.set_title('cam')

    ax = plt.subplot(1, 3, 3)
    ax.imshow(img)
    ax.imshow(cam, cmap='inferno', alpha=0.6)
    ax.set_title('cam overlay')
    plt.suptitle('patient: {}, target: {}, pred: {}'.format(dat.all_sequences[idx][0], target.argmax(), F.softmax(out).argmax()))
    plt.show()
