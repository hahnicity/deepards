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

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        #print(model_output)
        if target_class is None:
            target_class = np.argmax(model_output.cpu().data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output.cuda(), retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().data.numpy()
        # Get convolution outputs
        target = conv_output.cpu().data.numpy()
        weights = np.mean(guided_gradients, axis=(0, 2))
        # Take averages across all breaths because of the way we are structuring
        # our model
        target = np.mean(target, axis=0)
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :]

        #cam = np.mean(cam, axis = 0)
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
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
    target_class = None
    file_name_to_export = 'gradcam_output.png'

    for i in range(2):
        cam_outputs = np.empty((0,7))
        for j in range(100):
            breath_sequence = get_sequence(pickle_file_name, i , j, args.fold)
            grad_cam = GradCam(pretrained_model)
            cam = grad_cam.generate_cam(breath_sequence, target_class)
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
    #cam = grad_cam.generate_cam(breath_sequence, target_class)
    # Save mask
    #save_class_activation_images(breath_sequence, cam, file_name_to_export)
    print('Grad cam completed')
