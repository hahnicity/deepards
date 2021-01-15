"""
explainer_comparison
~~~~~~~~~~~~~~~~~~~~

Be able to visualize explanations of different explanation algos
"""
import argparse
import re

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from yaml import FullLoader, load

from deepards.dataset import ARDSRawDataset
from deepards.models.densenet import densenet18
from deepards.models.protopnet import construct_PPNet
from deepards.models.torch_cnn_linear_network import CNNLinearNetwork
from deepards.patient_gradcam import PatientGradCam
from deepards.ppnet_push import PrototypeVisualizer
from deepards.results import ModelCollection


class ExplainerComparison(object):
    def __init__(self, config_file_path):
        with open(config_file_path) as file:
            self.config = load(file, Loader=FullLoader)

        self.model_collection = pd.read_pickle(self.config['rf_results_file'])
        self.results_base_dir = self.config['results_base_dir']
        self.dataset = ARDSRawDataset.from_pickle(self.config['dataset_file'], False, 1.0, False, -1, 0)

    def find_correct_classified_subjects(self):
        """
        Return correct patients as a mapping with following information

        {
            <K-Fold idx1>: {
                'pts': [<patient1 with correct pred>, <patient2>, ...],
                'model_idx': <model index patients were selected from>,
                'gt': [<patient1 ground truth>, <patient2 gt>, ...],
            },
            <K-Fold idx2>: {
                ...
            },
            ...
        }
        """
        # we used 5 folds for our original random forest results
        correct_pts = {}
        for fold in range(5):
            all_pt_results = self.model_collection.get_all_patient_results_in_fold_dataframe(fold)
            model_idxs = all_pt_results.model_idx.unique()
            n_models_per_fold = len(model_idxs)
            # select model at random for each fold.
            model_idx = np.random.choice(model_idxs)
            model_results = all_pt_results[all_pt_results.model_idx == model_idx]
            pts = list(model_results[model_results.majority_prediction == model_results.ground_truth].patient_id)
            gt = list(model_results[model_results.patient_id.isin(pts)].ground_truth)
            correct_pts[fold] = {'pts': pts, 'model_idx': model_idx, 'gt': gt}
        return correct_pts


    def find_avail_hours(self, correct_pts):
        all_pt_results = self.model_collection.get_all_patient_results_in_fold_dataframe(0)
        pat = re.compile(r'hour_(\d\d?)_')
        hr_rows = []
        for fold_idx, v in correct_pts.items():
            pts = v['pts']
            gt = v['gt']
            model = self.model_collection.models[v['model_idx']]
            hr_df = model.get_patient_hourly_preds()
            for idx, pt in enumerate(pts):
                pt_ser = hr_df[hr_df.patient_id == pt].dropna(axis=1).drop('patient_id', axis=1).iloc[0]
                ards_considered_hrs = []
                non_ards_considered_hrs = []
                hrs = set([int(pat.search(col).groups()[0]) for col in pt_ser.index])
                for hr in hrs:
                    if pt_ser['hour_{}_ards_votes'.format(hr)] == 0:
                        non_ards_considered_hrs.append(hr)
                    elif pt_ser['hour_{}_other_votes'.format(hr)] == 0:
                        ards_considered_hrs.append(hr)

                empty_arr = np.empty((24, ))
                empty_arr.fill(np.nan)
                hr_row = [pt, fold_idx, v['model_idx'], gt[idx]] + list(empty_arr)
                offset = len(hr_row) - 24
                for hr in non_ards_considered_hrs:
                    hr_row[offset+hr] = 0
                for hr in ards_considered_hrs:
                    hr_row[offset+hr] = 1
                hr_rows.append(hr_row)
        return pd.DataFrame(hr_rows, columns=['patient_id', 'fold_idx', 'model_idx', 'ground_truth'] + [str(hr) for hr in range(24)])

    def run_gradcam(self, avail_hours, n_sequences_per_hour):
        for fold_idx in range(5):
            saved = torch.load(self.config['deepards_model_files'][fold_idx])
            # need to load model weights instead of loading the model wholesale
            # because of various changes in environment source code over time have
            # caused errors in running the code
            #
            # track_running_stats should be set based on the models we are using.
            # for final purposes track_running_stats should be set off.
            model = CNNLinearNetwork(densenet18(track_running_stats=self.config['track_running_stats_gradcam']), 20, 0)
            model.load_state_dict(saved.state_dict())

            self.dataset.set_kfold_indexes_for_fold(fold_idx)
            test_dataset = ARDSRawDataset.make_test_dataset_if_kfold(self.dataset)
            # set target as 0 for now, but we can switch based on patient
            gradcam = PatientGradCam(model, test_dataset, 0, self.results_base_dir)
            fold_avail_hours = avail_hours[avail_hours.fold_idx == fold_idx]
            for i, row in fold_avail_hours.iterrows():
                gradcam.target = row.ground_truth
                for hour_start in range(24):
                    if row[str(hour_start)] == row.ground_truth:
                        gradcam.get_cam_by_hour(row.patient_id, hour_start, hour_start+1, n_sequences_per_hour)

    def run_ppnet(self, avail_hours, n_sequences_per_hour):
        for fold_idx in range(5):
            saved = torch.load(self.config['ppnet_model_files'][fold_idx])
            # need to load model weights instead of loading the model wholesale
            # because of various changes in environment source code over time have
            # caused errors in running the code
            model = construct_PPNet(
                densenet18(self.config['track_running_stats_ppnet']),
                20,
                prototype_shape=saved.prototype_shape,
                incorrect_strength=saved.incorrect_strength,
                average_linear=saved.average_linear,
            )
            model.load_state_dict(saved.state_dict())

            self.dataset.set_kfold_indexes_for_fold(fold_idx)
            test_dataset = ARDSRawDataset.make_test_dataset_if_kfold(self.dataset)
            viz = PrototypeVisualizer(model, self.results_base_dir)
            fold_avail_hours = avail_hours[avail_hours.fold_idx == fold_idx]
            for i, row in fold_avail_hours.iterrows():
                for hour_start in range(24):
                    if row[str(hour_start)] == row.ground_truth:
                        viz.viz_prototypes_for_patient_and_label_by_hour(test_dataset, row.patient_id, hour_start, hour_start+1, n_sequences_per_hour)

    def visualize_gradcam(self, n_sequences_per_hour):
        correct_pts = self.find_correct_classified_subjects()
        avail_hours = self.find_avail_hours(correct_pts)
        self.run_gradcam(avail_hours, n_sequences_per_hour)

    def visualize_ppnet(self, n_sequences_per_hour):
        correct_pts = self.find_correct_classified_subjects()
        avail_hours = self.find_avail_hours(correct_pts)
        self.run_ppnet(avail_hours, n_sequences_per_hour)

    def _add_pickled_plot_to_fig(self, filename, new_figure, new_ax):
        saved_fig = pd.read_pickle(filename).get_figure()
        saved_ax = saved_fig.axes[0]
        # I tried a number of ideas getting this to work, but in the end
        # redoing the lines/patches/annos was the only avenue that worked
        # while being able to scale my figure properly.
        for line in saved_ax.lines:
            new_ax.plot(*line.get_data())
        for collect in saved_ax.collections:
            c = collect.get_array()
            cmap = collect.get_cmap()
            offsets = collect.get_offsets()
            x, y = offsets[:, 0], offsets[:, 1]
            vmin, vmax = collect.get_clim()
            new_ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax, s=10)

        for patch in saved_ax.patches:
            x0, x1, y0, y1 = patch._x0, patch._x1, patch._y0, patch._y1
            width, height = x1 - x0, y1 - y0

            rect = Rectangle(
                (x0, y0),
                width,
                height,
                linewidth=patch.get_lw(),
                edgecolor=patch.get_ec(),
                facecolor=patch.get_fc(),
                label=patch.get_label()
            )
            new_ax.add_patch(rect)

        for txt in saved_ax.texts:
            new_ax.annotate(
                txt.get_text().replace('proto_', ''),
                arrowprops=txt.arrowprops,
                xy=txt.xy,
                xytext=txt.xyann,
                xycoords=txt.xycoords,
                fontsize=txt._fontproperties.get_size()-1,
            )
        new_ax.set_xticks([])
        new_ax.set_yticks([])
        plt.close(saved_fig)

    def run_explainer(self, n_graphs_per_pt, results_dirname, results_name):
        for patho in ['ards', 'non_ards']:
            figure, axes = plt.subplots(n_graphs_per_pt, n_graphs_per_pt, figsize=(3*8, 3*4))
            for i, pt_dir in enumerate(Path(self.results_base_dir, results_dirname, 'hour_sequences', patho).iterdir()):
                all_files = list(pt_dir.glob('*/*.pkl'))
                selected = np.random.choice(all_files, size=n_graphs_per_pt)
                pt = pt_dir.name[:4]
                axes[i][0].set_ylabel(pt, rotation=0, size='large', labelpad=20)
                for j, filename in enumerate(selected):
                    self._add_pickled_plot_to_fig(str(filename), figure, axes[i][j])

                # XXX debug
                if i == n_graphs_per_pt-1:
                    break
            plt.suptitle(results_name + ' ' + patho.upper(), size='xx-large')
            plt.savefig('{}_{}.png'.format(results_name, patho))
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file for this experiment')
    parser.add_argument('--only-do', choices=['gradcam', 'ppnet', 'gradcam_explainer', 'ppnet_explainer'], help='only do a specific action instead of everything possible. gradcam=only analyze gradcam. ppnet=only analyze ppnet. gradcam_explainer=only visualize gradcam results. ppnet_explainer=only visualize ppnet results')
    parser.add_argument('--n-sequences-per-hr', type=int, help='only sample a limited number of reads per hour. By default is set to None, so no limit')
    parser.add_argument('--n-graphs-per-pt', type=int, help='only visualize a certain number of randomly selected graphs per pt', default=8)
    args = parser.parse_args()

    cls = ExplainerComparison(args.config_file)
    if args.only_do == 'gradcam' or (args.only_do is None):
        cls.visualize_gradcam(args.n_sequences_per_hr)
    if args.only_do == 'ppnet' or (args.only_do is None):
        cls.visualize_ppnet(args.n_sequences_per_hr)
    if args.only_do == 'gradcam_explainer' or (args.only_do is None):
        cls.run_explainer(args.n_graphs_per_pt, 'gradcam_results', 'GradCam')
    if args.only_do == 'ppnet_explainer' or (args.only_do is None):
        cls.run_explainer(args.n_graphs_per_pt, 'prototype_results', 'ProtoPNet')

    # XXX need to save the model index information and all that somewhere. to
    # ensure the results are reproducible


if __name__ == "__main__":
    main()
