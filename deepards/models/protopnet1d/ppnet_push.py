import copy
import math
from pathlib import Path
import os
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample
import torch

from deepards.models.protopnet1d.model import compute_rf_prototype
from deepards.models.protopnet1d.ppnet_helpers import makedir, find_high_activation_crop
from deepards.patient_gradcam import do_makedirs


class PrototypeVisualizer(object):
    def __init__(self, model, results_base_dir):
        """
        :param model: The pytorch model
        """
        self.model = model.cuda()
        self.seq_size = 224
        self.max_dist = self.model.prototype_shape[1] * self.model.prototype_shape[2]
        self.results_base_dir = results_base_dir

    def sanitize_inputs(self, protoL_input, proto_dist):
        protoL_input = np.copy(protoL_input.detach().cpu().numpy())
        if len(protoL_input.shape) == 4:
            protoL_input = protoL_input[0]
        elif len(protoL_input.shape) != 3:
            raise Exception('input sequences should be of 3 dimensions')
        proto_dist = np.copy(proto_dist.detach().cpu().numpy())
        if len(proto_dist.shape) == 4:
            proto_dist = proto_dist[0]
        elif len(proto_dist.shape) != 3:
            raise Exception('input distances should be of 3 dimensions')
        return protoL_input, proto_dist

    def plot_cam(self, seq, upsampled_act_seq_j, show_cbar):
        cam_j = upsampled_act_seq_j - np.amin(upsampled_act_seq_j)
        cam_j = (cam_j / np.amax(cam_j) * 255).ravel()
        t = np.arange(0, len(seq), 1)
        plt.scatter(t, seq, c=cam_j, vmin=0, vmax=255)
        if show_cbar:
            cbar = plt.colorbar()
            cbar.set_label('cam outputs', labelpad=-1)

    def show_proto_bounds(self, seq, proto_bound_j, rect_label, j):
        x_min = max([0, proto_bound_j[0]-2])
        y_min = seq[:, proto_bound_j[0]:proto_bound_j[1]].min() - .01
        width = proto_bound_j[1] - x_min + 2
        height = seq[:, proto_bound_j[0]:proto_bound_j[1]].max() - y_min + .02
        rect = Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none', label=rect_label)
        ax = plt.gca()
        ax.add_patch(rect)
        # XXX i want to have an algo that sets up new annos so that they dont cluster
        #prior_annos = [child for child in ax.get_children() if isinstance(child, matplotlib.text.Annotation)]
        ax.annotate(
            '{}'.format(j),
            arrowprops=dict(arrowstyle='->', facecolor='black'),
            xy=(x_min+(width/2), y_min+height),
            # XXX this anno location is good enuf for now, altho its still pretty rough
            xytext=(x_min+j**1.5, (y_min+height-math.sqrt(j))),
            xycoords='data',
            fontsize=6,
        )

    def viz_prototypes_for_patient_and_label_by_hour(self, dataset, patient_id, hour_start, hour_end, n_sequences_per_hour):
        """
        Visualize prototypes for patient using their ground truth label and searching
        over a start to end hour. This function visualizes all prototypes available,
        not just a single prototype
        """
        mapping = {0: 'non_ards', 1: 'ards'}
        gt = dataset.get_ground_truth_df()
        batch_size = dataset.all_sequences[0][1].shape[0]
        patient_idxs = gt[(gt.patient == patient_id) & (gt.hour >= hour_start) & (gt.hour < hour_end)].index
        if n_sequences_per_hour is not None:
            n_sequences_per_hour = n_sequences_per_hour if n_sequences_per_hour < len(patient_idxs) else len(patient_idxs)
            patient_idxs = np.random.choice(patient_idxs, size=n_sequences_per_hour, replace=False)

        if len(patient_idxs) == 0:
            return
        ground_truth = gt.loc[patient_idxs].y.iloc[0]
        dirname = Path(self.results_base_dir, 'prototype_results', 'hour_sequences', mapping[ground_truth], patient_id, str(hour_start))
        do_makedirs(str(dirname))

        for abs_idx in patient_idxs:
            rel_idx = list(dataset.kfold_indexes).index(abs_idx)
            # relative index specifies indices that we use for the kfold indexes.
            # abs_id specifies absolute ordering of the data in dataset.all_sequences.
            idx, seq, meta, target = dataset[rel_idx]
            with torch.no_grad():
                protoL_input, proto_dist = self.model.push_forward(torch.FloatTensor(seq).cuda().unsqueeze(0))

            for seq_idx in range(batch_size):
                self.viz_prototypes_for_target(seq, ground_truth, protoL_input, proto_dist, seq_idx)
                filename = dirname.joinpath('seq-{}-{}-target-{}.pkl'.format(abs_idx, seq_idx, mapping[ground_truth]))
                # saving to pickle is much faster, and we can convert to png if we
                # want to later
                ax = plt.gca()
                pd.to_pickle(ax, str(filename), compression=None)
                # clear axis so we're not redrawing the figure and axes each time
                # this speeds up processing **considerably**
                plt.cla()

    def viz_prototypes_for_target(self, seq, target, protoL_input, proto_dist, breath_idx):
        """
        Visualize all prototypes on target sequence. Don't use heatmap, because
        it will just be a mess

        :param XXX:
        """
        n_prototypes = self.model.num_prototypes
        protos_of_interest = range(n_prototypes / 2 * target, n_prototypes / 2 * (1 + target))
        self.viz_single_prototype(seq, target, protoL_input, proto_dist, protos_of_interest[0], breath_idx, True, False, True, False, False, None)
        for proto in protos_of_interest[1:]:
            self.viz_single_prototype(seq, target, protoL_input, proto_dist, proto, breath_idx, False, False, False, False, False, None)

    def viz_single_prototype(self,
                             seq,
                             target,
                             protoL_input,
                             proto_dist,
                             prototype_idx,
                             breath_idx,
                             show_title,
                             show_legend,
                             plot_sequence,
                             plot_cam=True,
                             show_cbar=True,
                             rect_label='prototype activation zone'):
        """
        For a single breath in single read, visualize the receptive field focused on by
        a prototype

        :param seq: sequence of data for the read
        :param target: 0/1 for Non-ARDS/ARDS
        :param protoL_input: input into the prototype layer
        :param proto_dist: prototype distances
        :param prototype_idx: the specific prototype we want to focus on
        :param breath_idx: the specific breath we want to focus on
        :param show_title: true/false on whether to show the plot title
        :param show_legend: true/false on whether to show plot legend
        :param plot_sequence:
        :param plot_cam:
        :param show_cbar: true/false on whether to show the color bar
        :param rect_label:
        """
        protoL_input, proto_dist = self.sanitize_inputs(protoL_input, proto_dist)

        proto_dist_j = proto_dist[breath_idx, prototype_idx, :]
        width_index = np.argmin(proto_dist_j)
        batch_argmin_proto_dist_j = [0, breath_idx, width_index]

        # get the receptive field boundary of the image patch
        # that generates the representation
        protoL_rf_info = self.model.proto_layer_rf_info
        rf_prototype_j = compute_rf_prototype(self.seq_size, batch_argmin_proto_dist_j, protoL_rf_info)

        # get the breath-specific instance
        seq = seq[breath_idx]
        if isinstance(seq, torch.Tensor):
            seq = seq.numpy()

        # crop out the receptive field
        rf_seq_j = seq[:, rf_prototype_j[1]:rf_prototype_j[2]]

        # find the highly activated region of the original image
        proto_dist_seq_j = proto_dist[breath_idx, prototype_idx, :]
        if self.model.prototype_activation_function == 'log':
            proto_act_seq_j = np.log((proto_dist_seq_j + 1) / (proto_dist_seq_j + self.model.epsilon))
        elif self.model.prototype_activation_function == 'linear':
            proto_act_seq_j = self.max_dist - proto_dist_seq_j
        else:
            proto_act_seq_j = proto_activation_function_in_numpy(proto_dist_seq_j)
        # use scipy.signal.resample.
        upsampled_act_seq_j = resample(proto_act_seq_j, self.seq_size)
        proto_bound_j = find_high_activation_crop(upsampled_act_seq_j)
        # save the prototype boundary (rectangular boundary of highly activated region)

        # save the whole image containing the prototype as png
        seq_reshape = seq.reshape((self.seq_size, 1))

        t = np.arange(0, len(seq_reshape), 1)
        if plot_sequence:
            #import IPython; IPython.embed()
            plt.plot(seq[0])

        if plot_cam:
            self.plot_cam(seq_reshape, upsampled_act_seq_j, show_cbar)

        if show_title:
            mapping = {0: 'Non-ARDS', 1: 'ARDS'}
            gt_lab = mapping[target]
            title = 'gt: {}, breath_idx: {} proto: {}'.format(
                gt_lab,
                breath_idx,
                prototype_idx,
            )
            plt.title(title)

        self.show_proto_bounds(seq, proto_bound_j, rect_label, prototype_idx)
        if show_legend:
            plt.legend()


# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    model, # pytorch network with prototype_vectors
                    cuda_wrapper,
                    class_specific=True,
                    proto_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_num=None, # if not provided, prototypes saved previously will be overwritten
                    proto_seq_filename_prefix=None,
                    proto_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_proto_class_identity=True, # which class the prototype image comes from
                    proto_activation_function_in_numpy=None):

    model.eval()
    print('\tpush')

    start = time.time()
    prototype_shape = model.prototype_shape
    n_prototypes = model.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes, prototype_shape[1], prototype_shape[2]]
    )
    # proto_rf_boxes and proto_bound_boxes column:
    # 0: image index in the entire dataset
    # 1: width start index
    # 2: width end index
    # 3: (optional) class identity
    if save_proto_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 4],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 4],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 3],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 3],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_num != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_num))
            try:
                makedir(proto_epoch_dir)
            except OSError:
                pass
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = model.num_classes

    for batch_idx, (obs_idx, seq, metadata, target) in enumerate(dataloader):
        # start_index_of_search keeps track of the index of the image
        # assigned to serve as prototype
        start_index_of_search_batch = batch_idx * search_batch_size
        update_prototypes_on_batch(seq,
                                   start_index_of_search_batch,
                                   model,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   cuda_wrapper,
                                   class_specific=class_specific,
                                   target=target,
                                   num_classes=num_classes,
                                   proto_layer_stride=proto_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   proto_seq_filename_prefix=proto_seq_filename_prefix,
                                   proto_self_act_filename_prefix=proto_self_act_filename_prefix,
                                   proto_activation_function_in_numpy=proto_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_num) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_num) + '.npy'),
                proto_bound_boxes)

    print('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    # XXX need to make this device agnostic
    model.prototype_vectors.data.copy_(cuda_wrapper(torch.tensor(prototype_update, dtype=torch.float32)))
    # model.cuda()
    end = time.time()
    print('\tpush time: \t{0}'.format(end -  start))


# update each prototype for current search batch
# push is linked with visualization of prototypes this might not be
# the best practice if you want to debug whats happening with the network
def update_prototypes_on_batch(seq,
                               start_index_of_search_batch,
                               model,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               cuda_wrapper,
                               class_specific=True,
                               target=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               proto_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               proto_seq_filename_prefix=None,
                               proto_self_act_filename_prefix=None,
                               proto_activation_function_in_numpy=None):

    model.eval()
    search_batch = seq

    with torch.no_grad():
        # XXX need to make this device agnostic
        search_batch = cuda_wrapper(search_batch.float())
        protoL_input, proto_dist = model.push_forward(search_batch)

    protoL_input = np.copy(protoL_input.detach().cpu().numpy())
    proto_dist = np.copy(proto_dist.detach().cpu().numpy())

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(target):
            img_label = img_y.argmax().item()
            class_to_img_index_dict[img_label].append(img_index)

    n_prototypes = model.prototype_shape[0]
    proto_w = model.prototype_shape[2]
    max_dist = model.prototype_shape[1] * model.prototype_shape[2]

    for j in range(n_prototypes):
        if class_specific:
            target_class = torch.argmax(model.prototype_class_identity_orig[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist[class_to_img_index_dict[target_class],:,j,:]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist[:,:,j,:]

        # just gets the global min of the distances
        batch_min_proto_dist_j = np.amin(proto_dist_j)
        # So that would likely be why push is coming up with
        # different zones for the prototypes, because they can change in different
        # mins. I imagine the authors intended for this instability to ease over time
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            # just finds the indexing of the min in the array. they use unravel_index
            # they just wanted a clever 1-liner to find this.
            #
            # maybe we can just have this get a sub-batch specific item?? I dont know
            # the rest of the code well enuf to know if this makes sense or not.
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                # change the argmin index from the index among
                # images of the target class to the index in the entire search
                # batch
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            breath_index_in_read = batch_argmin_proto_dist_j[1]
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * proto_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input[img_index_in_batch,
                                                  breath_index_in_read,
                                                  :,
                                                  fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            # this variable is essential for the prototype push
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = model.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(3), batch_argmin_proto_dist_j, protoL_rf_info)

            # get the whole image
            original_seq_j = seq[rf_prototype_j[0], breath_index_in_read]
            original_seq_j = original_seq_j.numpy()
            original_seq_size = original_seq_j.shape[1]

            # crop out the receptive field
            rf_seq_j = original_seq_j[:, rf_prototype_j[1]:rf_prototype_j[2]]

            # find the highly activated region of the original image
            proto_dist_seq_j = proto_dist[img_index_in_batch, breath_index_in_read, j, :]
            if model.prototype_activation_function == 'log':
                proto_act_seq_j = np.log((proto_dist_seq_j + 1) / (proto_dist_seq_j + model.epsilon))
            elif model.prototype_activation_function == 'linear':
                proto_act_seq_j = max_dist - proto_dist_seq_j
            else:
                proto_act_seq_j = proto_activation_function_in_numpy(proto_dist_seq_j)
            # use scipy.signal.resample.
            upsampled_act_seq_j = resample(proto_act_seq_j, original_seq_size)
            proto_bound_j = find_high_activation_crop(upsampled_act_seq_j)
            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            if proto_rf_boxes.shape[1] == 4 and target is not None:
                proto_rf_boxes[j, 3] = target[rf_prototype_j[0]].argmax().item()

            # XXX I'd like to split the viz work into separate functions. Need
            # to consolidate this with the Visualizer class above
            if dir_for_saving_prototypes is not None:
                if proto_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         proto_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_seq_j)
                if proto_seq_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    filename = "{}-batch_idx_{}-breath_num_{}-proto_{}.png".format(
                        proto_seq_filename_prefix,
                        proto_rf_boxes[j, 0],
                        breath_index_in_read,
                        j,
                    )
                    cam_j = upsampled_act_seq_j - np.amin(upsampled_act_seq_j)
                    cam_j = (cam_j / np.amax(cam_j) * 255).ravel()
                    if rf_seq_j.shape[1] != original_seq_size:
                        seq_reshape = rf_seq_j.reshape((rf_seq_j.shape[1], 1))
                        cam_j = cam_j[rf_prototype_j[1]:rf_prototype_j[2]]
                        filename = filename.replace('.png', '-receptive-field.png')
                        x_min = max([0, proto_bound_j[0]-rf_prototype_j[1]-2])
                        y_min = original_seq_j[:, proto_bound_j[0]:proto_bound_j[1]].min() - .01
                        width = proto_bound_j[1] - proto_bound_j[0] + 2.5
                        height = original_seq_j[:, proto_bound_j[0]:proto_bound_j[1]].max() - y_min + .02
                    else:
                        seq_reshape = original_seq_j.reshape((original_seq_size, 1))
                        x_min = max([0, proto_bound_j[0]-2])
                        y_min = original_seq_j[:, proto_bound_j[0]:proto_bound_j[1]].min() - .01
                        width = proto_bound_j[1] - x_min + 2
                        height = original_seq_j[:, proto_bound_j[0]:proto_bound_j[1]].max() - y_min + .02

                    t = np.arange(0, len(seq_reshape), 1)
                    plt.scatter(t, seq_reshape, c=cam_j, vmin=0, vmax=255)
                    plt.plot(t, seq_reshape)
                    cbar = plt.colorbar()
                    cbar.set_label('cam outputs', labelpad=-1)
                    mapping = {0: 'Non-ARDS', 1: 'ARDS'}
                    gt_lab = mapping[target[rf_prototype_j[0]].argmax().item()]
                    title = 'gt: {}, batch_idx: {} breath_idx: {} proto: {}'.format(
                        gt_lab,
                        proto_rf_boxes[j, 0],
                        breath_index_in_read,
                        j,
                    )
                    plt.title(title)
                    final_path = str(Path(dir_for_saving_prototypes).joinpath(filename))
                    rect = Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none', label='prototype activation zone')
                    ax = plt.gca()
                    ax.add_patch(rect)
                    plt.legend()
                    plt.savefig(final_path)
                    plt.close()
                    # and why add fractional vals of the img and heatmap?
                    # probably for opacity reasons when viewing the image
                    #overlayed_original_seq_j = 0.5 * original_seq_j + 0.3 * heatmap
    if class_specific:
        del class_to_img_index_dict


# push each prototype to the nearest patch in the training set
def prototype_viz(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                  model, # pytorch network with prototype_vectors
                  root_dir_for_saving_prototypes,
                  proto_seq_filename_prefix,
                  epoch_num,
                  cuda_wrapper,
                  class_specific=True,
                  proto_layer_stride=1):

    model.eval()
    prototype_shape = model.prototype_shape
    n_prototypes = model.num_prototypes
    proto_rf_boxes = np.full(shape=[n_prototypes, 4],
                                fill_value=-1)
    proto_bound_boxes = np.full(shape=[n_prototypes, 4],
                                        fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_num != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_num))
            try:
                makedir(proto_epoch_dir)
            except OSError:
                pass
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = model.num_classes

    for batch_idx, (obs_idx, seq, metadata, target) in enumerate(dataloader):
        # start_index_of_search keeps track of the index of the image
        # assigned to serve as prototype
        start_index_of_search_batch = batch_idx * search_batch_size
        viz_proto_batch(seq,
                        start_index_of_search_batch,
                        model,
                        proto_rf_boxes,
                        proto_bound_boxes,
                        cuda_wrapper,
                        class_specific=class_specific,
                        target=target,
                        num_classes=num_classes,
                        proto_layer_stride=proto_layer_stride,
                        dir_for_saving_prototypes=proto_epoch_dir,
                        proto_seq_filename_prefix=proto_seq_filename_prefix)


# update each prototype for current search batch
# push is linked with visualization of prototypes this might not be
# the best practice if you want to debug whats happening with the network
def viz_proto_batch(seq,
                    start_index_of_search_batch,
                    model,
                    proto_rf_boxes,
                    proto_bound_boxes,
                    cuda_wrapper,
                    class_specific=True,
                    target=None,
                    num_classes=None,
                    proto_layer_stride=1,
                    dir_for_saving_prototypes=None,
                    proto_seq_filename_prefix=None):

    model.eval()
    search_batch = seq

    with torch.no_grad():
        search_batch = cuda_wrapper(search_batch.float())
        protoL_input, proto_dist = model.push_forward(search_batch)

    protoL_input = np.copy(protoL_input.detach().cpu().numpy())
    proto_dist = np.copy(proto_dist.detach().cpu().numpy())

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(target):
            img_label = img_y.argmax().item()
            class_to_img_index_dict[img_label].append(img_index)

    n_prototypes = model.prototype_shape[0]
    proto_w = model.prototype_shape[2]
    max_dist = model.prototype_shape[1] * model.prototype_shape[2]

    for j in range(n_prototypes):
        if class_specific:
            target_class = torch.argmax(model.prototype_class_identity_orig[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist[class_to_img_index_dict[target_class],:,j,:]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist[:,:,j,:]

        # just gets the global min of the distances
        batch_min_proto_dist_j = np.amin(proto_dist_j)
        batch_argmin_proto_dist_j = \
            list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                  proto_dist_j.shape))
        if class_specific:
            # change the argmin index from the index among
            # images of the target class to the index in the entire search
            # batch
            batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

        # retrieve the corresponding feature map patch
        img_index_in_batch = batch_argmin_proto_dist_j[0]
        breath_index_in_read = batch_argmin_proto_dist_j[1]
        fmap_width_start_index = batch_argmin_proto_dist_j[2] * proto_layer_stride
        fmap_width_end_index = fmap_width_start_index + proto_w

        batch_min_fmap_patch_j = protoL_input[img_index_in_batch,
                                              breath_index_in_read,
                                              :,
                                              fmap_width_start_index:fmap_width_end_index]

        # get the receptive field boundary of the image patch
        # that generates the representation
        protoL_rf_info = model.proto_layer_rf_info
        rf_prototype_j = compute_rf_prototype(search_batch.size(3), batch_argmin_proto_dist_j, protoL_rf_info)

        # get the whole image
        original_seq_j = seq[rf_prototype_j[0], breath_index_in_read]
        original_seq_j = original_seq_j.numpy()
        original_seq_size = original_seq_j.shape[1]

        # crop out the receptive field
        rf_seq_j = original_seq_j[:, rf_prototype_j[1]:rf_prototype_j[2]]

        # find the highly activated region of the original image
        proto_dist_seq_j = proto_dist[img_index_in_batch, breath_index_in_read, j, :]
        if model.prototype_activation_function == 'log':
            proto_act_seq_j = np.log((proto_dist_seq_j + 1) / (proto_dist_seq_j + model.epsilon))
        elif model.prototype_activation_function == 'linear':
            proto_act_seq_j = max_dist - proto_dist_seq_j
        else:
            proto_act_seq_j = proto_activation_function_in_numpy(proto_dist_seq_j)
        # use scipy.signal.resample.
        upsampled_act_seq_j = resample(proto_act_seq_j, original_seq_size)
        proto_bound_j = find_high_activation_crop(upsampled_act_seq_j)

        # save the prototype boundary (rectangular boundary of highly activated region)
        proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
        proto_bound_boxes[j, 1] = proto_bound_j[0]
        proto_bound_boxes[j, 2] = proto_bound_j[1]
        if proto_rf_boxes.shape[1] == 4 and target is not None:
            proto_rf_boxes[j, 3] = target[rf_prototype_j[0]].argmax().item()

        if dir_for_saving_prototypes is not None:
            if proto_seq_filename_prefix is not None:
                # save the whole image containing the prototype as png
                filename = "{}-batch_idx_{}-breath_num_{}-proto_{}.png".format(
                    proto_seq_filename_prefix,
                    proto_rf_boxes[j, 0],
                    breath_index_in_read,
                    j,
                )
                cam_j = upsampled_act_seq_j - np.amin(upsampled_act_seq_j)
                cam_j = (cam_j / np.amax(cam_j) * 255).ravel()
                if rf_seq_j.shape[1] != original_seq_size:
                    seq_reshape = rf_seq_j.reshape((rf_seq_j.shape[1], 1))
                    cam_j = cam_j[rf_prototype_j[1]:rf_prototype_j[2]]
                    filename = filename.replace('.png', '-receptive-field.png')
                    x_min = max([0, proto_bound_j[0]-rf_prototype_j[1]-2])
                    y_min = original_seq_j[:, proto_bound_j[0]:proto_bound_j[1]].min() - .01
                    width = proto_bound_j[1] - proto_bound_j[0] + 2.5
                    height = original_seq_j[:, proto_bound_j[0]:proto_bound_j[1]].max() - y_min + .02
                else:
                    seq_reshape = original_seq_j.reshape((original_seq_size, 1))
                    x_min = max([0, proto_bound_j[0]-2])
                    y_min = original_seq_j[:, proto_bound_j[0]:proto_bound_j[1]].min() - .01
                    width = proto_bound_j[1] - x_min + 2
                    height = original_seq_j[:, proto_bound_j[0]:proto_bound_j[1]].max() - y_min + .02

                t = np.arange(0, len(seq_reshape), 1)
                plt.scatter(t, seq_reshape, c=cam_j, vmin=0, vmax=255)
                plt.plot(t, seq_reshape)
                cbar = plt.colorbar()
                cbar.set_label('cam outputs', labelpad=-1)
                mapping = {0: 'Non-ARDS', 1: 'ARDS'}
                gt_lab = mapping[target[rf_prototype_j[0]].argmax().item()]
                title = 'gt: {}, batch_idx: {} breath_idx: {} proto: {}'.format(
                    gt_lab,
                    proto_rf_boxes[j, 0],
                    breath_index_in_read,
                    j,
                )
                plt.title(title)
                final_path = str(Path(dir_for_saving_prototypes).joinpath(filename))
                rect = Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none', label='prototype activation zone')
                ax = plt.gca()
                ax.add_patch(rect)
                plt.legend()
                plt.savefig(final_path)
                plt.close()
                # and why add fractional vals of the img and heatmap?
                # probably for opacity reasons when viewing the image
                #overlayed_original_seq_j = 0.5 * original_seq_j + 0.3 * heatmap
