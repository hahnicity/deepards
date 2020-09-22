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

#from breath_visualize import visualize_sequence

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
        '''
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        '''
        x = self.model.breath_block.features(x)
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
        x = self.model.breath_block.avgpool(x).view(-1)
        x = self.model.linear_final(x).unsqueeze(0)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model)

    def generate_one_hot_grad_and_output(self, input, target):
        return self._generate_grad_and_output(input, target, self.one_hot_model_output)

    def generate_static_grad_and_output(self, input, target):
        return self._generate_grad_and_output(input, target, self.static_model_output)

    def _generate_grad_and_output(self, input, target, model_out_func):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        conv_output, model_output = self.extractor.forward_pass(input)
        # this line ensures grad cam is done wrt model prediction
        output = model_out_func(model_output, target)
        # Target for backprop
        self.model.zero_grad()
        # Backward pass wrt output
        output.backward(retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().data.numpy()
        # Get convolution outputs
        conv_output = conv_output.cpu().data.numpy()
        return conv_output, guided_gradients, model_output

    def static_model_output(self, output, target):
        # XXX this isnt working....
        return torch.sum(output)

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
    def generate_read_cam(self, input, target):
        conv_output, grad, mo = self.generate_one_hot_grad_and_output(input, target)
        weights = np.mean(grad, axis=(2,))
        cam = np.zeros((conv_output.shape[0], conv_output.shape[2]), dtype=np.float32)
        for i, b in enumerate(conv_output):
            for j, w in enumerate(weights[i,:]):
                cam[i] += w * conv_output[i, j, :]
            cam[i] = self.normalize(cam[i])
        return cam

    def generate_cam(self, input, target=None):
        conv_output, grad, mo = self.generate_one_hot_grad_and_output(input, target)
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

        return self.normalize(cam)

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
        return cam

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='path to the saved_model')
    parser.add_argument('-pdp', '--pickled-data-path', help = 'PATH to pickled data', required=True)
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()

    pretrained_model = torch.load(args.model_path)
    # ensure that model is on same cuda device that data will be on
    if not isinstance(pretrained_model, torch.nn.DataParallel):
        pretrained_model = pretrained_model.to(torch.cuda.current_device())
    else:
        pretrained_model = pretrained_model.module

    #taking the pickle file we need to extract data from
    pickle_file_name = args.pickled_data_path
    # Get params
    #target_example = 0  # Snake
    target = None
    file_name_to_export = 'gradcam_output.png'

    for i in range(2):
        cam_outputs = np.empty((0,7))
        for j in range(100):
            breath_sequence = get_sequence(pickle_file_name, i , j, args.fold)
            grad_cam = GradCam(pretrained_model)
            cam = grad_cam.generate_cam(breath_sequence, target)
            #print(cam)
            cam = np.expand_dims(cam, axis = 0)
            cam_outputs = np.append(cam_outputs, cam, axis = 0)
        cam_outputs = np.mean(cam_outputs, axis = 0)
        cam_outputs = cv2.resize(cam_outputs,(1,224))
        #print(cam_outputs.shape)
        outfile = None
        if i == 0:
            outfile = open('others_gradcam.pkl','wb')
            #print("Gradcam output for others : {}".format(cam_outputs))
        else:
            outfile = open('ARDS_gradcam.pkl','wb')
            #print("Gradcam output for ARDS: {}".format(cam_outputs))
        pickle.dump(cam_outputs, outfile)
        #visualize_sequence(cam_outputs)

    #breath_sequence = get_sequence(pickle_file_name)
    # Grad cam
    #grad_cam = GradCam(pretrained_model)
    # Generate cam mask
    #cam = grad_cam.generate_cam(breath_sequence, target)
    # Save mask
    #save_class_activation_images(breath_sequence, cam, file_name_to_export)
    print('Grad cam completed')
