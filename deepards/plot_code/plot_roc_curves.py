import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import auc, roc_auc_score, roc_curve

from deepards.mean_metrics import find_matching_experiments


class SillyPlottingClass(object):
    def __init__(self, reg_ml_results, dl_experiment_name):
        self.reg_ml_results = pd.read_pickle(reg_ml_results)
        # renaming so that we can maintain consistency with DL cols
        self.reg_ml_results = self.reg_ml_results.rename(columns={
            'fold_idx': 'fold_num',
            'patient_id': 'patient',
            'ground_truth': 'patho',
            'frac_votes': 'pred_frac',
        })
        self.dl_experiment_ids = find_matching_experiments(dl_experiment_name)

        # plot chance initially because we don't want to doublt plot
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.grid(alpha=.2)

    def plot_dl_results(self):
        df_patient_results_list = []
        for time in self.dl_experiment_ids:
            df = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', "results/{}_patient_results.pkl".format(time)))
            df_patient_results_list.append(df)
        df = pd.concat(df_patient_results_list)
        df['patho'] = df.patho.astype(int)
        epoch_to_auc = [(ep, roc_auc_score(frame.patho, frame.pred_frac)) for ep, frame in df.groupby('epoch_num')]
        best_epoch = sorted(epoch_to_auc, key=lambda x: x[1])[-1][0]
        df = df[df.epoch_num == best_epoch]
        self.plot_results(df, 'orange', 'goldenrod', 'Deep Learning')

    def plot_reg_ml_results(self):
        """Convenience method"""
        self.plot_results(self.reg_ml_results, 'forestgreen', 'lightgreen', 'Random Forest')

    def plot_results(self, results, line_color, std_color, type_roc):
        # I might be able to find confidence std using p(1-p). Nah. we actually cant do
        # this because polling is using the identity of std from a binomial distribution. So
        # in order to have conf interval we need some kind of observable std.
        tprs = []
        aucs = []
        threshes = set()
        mean_fpr = np.linspace(0, 1, 100)
        uniq_pts = len(results.patient.unique())

        for fold_num in results.fold_num.unique():
            fold_preds = results[results.fold_num == fold_num]
            fpr, tpr, thresh = roc_curve(fold_preds.patho, fold_preds.pred_frac)
            threshes.update(thresh)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = round(auc(mean_fpr, mean_tpr), 2)
        std_auc = np.std(aucs)

        auc_ci = (1.96 * np.sqrt(mean_auc * (1-mean_auc) / uniq_pts)).round(3)
        plt.plot(mean_fpr, mean_tpr, color=line_color,
                 label=r'%s ROC (AUC = %0.2f$\pm$%0.3f)' % (type_roc, mean_auc, auc_ci),
                 lw=2.0, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=std_color, alpha=.2,
                         label=r'{} 1 std. dev.'.format(type_roc))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reg_ml_results_file')
    parser.add_argument('dl_experiment_name')
    args = parser.parse_args()
    cls = SillyPlottingClass(args.reg_ml_results_file, args.dl_experiment_name)
    cls.plot_reg_ml_results()
    cls.plot_dl_results()
    plt.legend(loc="lower right")
    plt.savefig('/home/greg/ardsresearch/paper_imgs/roc-dl-ml.png', dpi=400, bbox_inches='tight')


if __name__ == "__main__":
    main()
