import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from deepards.models.densenet import densenet18


def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding,
                          previous_layer_rf_info):
    """
    This method isnt too rough. Just follows the receptive field for a network
    across the different layers. Works forwards. seems like it wants to find
    approx position of prototype relative to the start input
    """
    n_in = previous_layer_rf_info[0] # input size
    j_in = previous_layer_rf_info[1] # receptive field jump of input layer
    r_in = previous_layer_rf_info[2] # receptive field size of input layer
    start_in = previous_layer_rf_info[3] # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1

    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1)*j_in
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
    return [n_out, j_out, r_out, start_out]


def compute_rf_protoL_at_spatial_location(seq_len, width_index, protoL_rf_info):
    # XXX removed height from eq here. but still need to reconcile
    # protoL_rf_info
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    # XXX start hasn't changed from 0.5 even though a number of other params have.
    start = protoL_rf_info[3]
    assert(width_index < n)

    center_w = start + (width_index*j)

    rf_start_width_index = max(int(center_w - (r/2)), 0)
    rf_end_width_index = min(int(center_w + (r/2)), seq_len)

    return [rf_start_width_index, rf_end_width_index]


def compute_rf_prototype(seq_len, prototype_patch_index, protoL_rf_info):
    img_index = prototype_patch_index[0]
    width_index = prototype_patch_index[2]
    rf_indices = compute_rf_protoL_at_spatial_location(seq_len,
                                                       width_index,
                                                       protoL_rf_info)
    return [img_index, rf_indices[0], rf_indices[1]]


def compute_rf_prototypes(seq_len, prototype_patch_indices, protoL_rf_info):
    rf_prototypes = []
    for prototype_patch_index in prototype_patch_indices:
        proto = compute_rf_prototype(seq_len, prototype_patch_index, protoL_rf_info)
        rf_prototypes.append(proto)
    return rf_prototypes


def compute_proto_layer_rf_info_v2(seq_len, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):
    assert(len(layer_filter_sizes) == len(layer_strides))
    assert(len(layer_filter_sizes) == len(layer_paddings))

    # XXX im trying to figure this out.
    # so the first is receptive field input size
    # second is receptive field jump (so stride?) (yeah think its stride)
    # third is receptive field size
    # fourth is receptive field center.
    #
    # does this need modification?
    # I don't think this needs modification at all.
    rf_info = [seq_len, 1, 1, 0.5]

    for i in range(len(layer_filter_sizes)):
        filter_size = layer_filter_sizes[i]
        stride_size = layer_strides[i]
        padding_size = layer_paddings[i]

        rf_info = compute_layer_rf_info(layer_filter_size=filter_size,
                                layer_stride=stride_size,
                                layer_padding=padding_size,
                                previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info


class PPNet(nn.Module):
    def __init__(self, features, seq_len, prototype_shape, sub_batch_size, batch_size,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.seq_len = seq_len
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        self.sub_batch_size = sub_batch_size

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        # Here we are initializing the class identities of the prototypes
        # Without domain specific knowledge we allocate the same number of
        # prototypes for each class
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity_orig = torch.zeros(self.num_prototypes, self.num_classes)
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity_orig[j, j // num_prototypes_per_class] = 1
        self.prototype_class_identity = self.prototype_class_identity_orig.repeat((20, 1))

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.BatchNorm1d)][-1].num_features

        # this is evaluated after regular features. Probably to downsize n channels
        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv1d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv1d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv1d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                # sigmoid is basically used for on/off purposes.
                nn.Sigmoid()
                )

        # XXX interesting that they just choose random init, rather than xavier
        # or kaiming. but its probably a minor point
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        # Set ones to ensure that prototypes contribute to a prediction, rather
        # than from a prediction.:
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes*sub_batch_size,
                                         self.num_classes,
                                         bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features.forward_no_pool(x)
        x = self.add_on_layers(x)
        return x

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x

        produces a map of the distances between the jth prototype and all patches
        of the conv output that have the same shape as p_j.

        Follows l2 distance where ||q-p|| = sqrt(||q||^2 + ||p||^2 - 2*p*q)
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv1d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1)
        p2_reshape = p2.view(-1, 1)

        # This is how the prototype vectors get updated, because theyre
        # a weight in a 1d conv
        xp = F.conv1d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def seq_forward(self, x):
        distances = self.prototype_distances(x)
        # global min pooling that follows the l2 distance calcs
        min_distances = -F.max_pool1d(-distances, kernel_size=distances.size()[2])
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        return prototype_activations, min_distances

    def forward(self, x, metadata):
        outputs, min_distances = self.seq_forward(x[0])
        # I think I can just do 1 linear op to save time at the end before
        # I return outputs from the function
        outputs, min_distances = self.last_layer(outputs.view(-1)).unsqueeze(0), min_distances.unsqueeze(0)
        for i in range(1, x.shape[0]):
            tmp_o, tmp_md = self.seq_forward(x[i])
            outputs = torch.cat([outputs, self.last_layer(tmp_o.view(-1)).unsqueeze(0)], dim=0)
            min_distances = torch.cat([min_distances, tmp_md.unsqueeze(0)], dim=0)
        return outputs, min_distances.view((min_distances.shape[0], -1))

    def push_forward(self, x):
        # basically similar ops as the forward, except no pooling on the distances
        # and no classification layer is used
        outputs = self.conv_features(x[0])
        min_distances = self._l2_convolution(outputs)
        outputs, min_distances = outputs.unsqueeze(0), min_distances.unsqueeze(0)
        # this part comes from the fact we are using windows
        for i in range(1, x.shape[0]):
            tmp_o = self.conv_features(x[0])
            tmp_md = self._l2_convolution(tmp_o)
            outputs = torch.cat([outputs, tmp_o.unsqueeze(0)], dim=0)
            min_distances = torch.cat([min_distances, tmp_md.unsqueeze(0)], dim=0)

        return outputs, min_distances

    def __repr__(self):
        # PPNet(self, features, seq_len, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\tseq_len: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.seq_len,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        Preset weights on final linear layer so that things will be weighted
        toward using class specific prototypes to make class specific predictions

        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        weights = (correct_class_connection * positive_one_weights_locations
                   + incorrect_class_connection * negative_one_weights_locations)
        self.last_layer.weight.data.copy_(weights)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv1d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


# first num in prototype_shape has to do with how many total prototypes we want.
# second one has to do with the num channels our filter map
def construct_PPNet(base_architecture, sub_batch_size, seq_len=224,
                    prototype_shape=(20, 128, 1), num_classes=2,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck', batch_size=16):
    # add_on_layers_type is set to "regular" in the original settings
    layer_filter_sizes, layer_strides, layer_paddings = base_architecture.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(seq_len=seq_len,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return PPNet(features=base_architecture,
                 seq_len=seq_len,
                 prototype_shape=prototype_shape,
                 sub_batch_size=sub_batch_size,
                 batch_size=batch_size,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)



if __name__ == "__main__":
    base = densenet18()
    ppnet = construct_PPNet(base)
    breaths = torch.rand((2, 20, 1, 224))
    output, min_distances = ppnet(breaths, None)
    print(output.shape, min_distances.shape)
