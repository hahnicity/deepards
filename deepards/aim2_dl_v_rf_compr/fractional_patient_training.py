from glob import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepards.mean_metrics import confidence_score, get_metrics

marker_size = (plt.rcParams['lines.markersize'] ** 2) * 3/2
results_path = lambda x: os.path.join(os.path.dirname(__file__), '..', 'results', x)
experiment_prefixes = [
    (100, results_path("linear_baseline_recheck_back_on_master_after_eval_off")),
    (75, results_path("fractional_dataset_training75_")),  # 75%
    (50, results_path("fractional_dataset_training50_")),  # 50%
    (25, results_path("fractional_dataset_training25_")),  # 25%
    (12.5, results_path("fractional_dataset_training125_")),  # 12.5%
    (10, results_path("fractional_dataset_training1_")),  # 10%
    (7.5, results_path("fractional_dataset_training075_")),  # 7.5%
    (5, results_path("fractional_dataset_training05_")),  # 5%
    (2.5, results_path("fractional_dataset_training025_")),  # 2.5%
]

stats = dict()
accuracies = dict()
aucs = dict()
for tupl in experiment_prefixes:
    perc, exp = tupl
    start_times = list(set([os.path.splitext(file_.split('_')[-1])[0] for file_ in glob(exp + '*')]))

    mean_df_stats, all_stats = get_metrics(start_times)
    stats[perc] = all_stats

    max_acc = 0
    max_auc = 0
    if perc == 100:
        final_epoch = 10
    else:
        final_epoch = 9  # somehow things went from 0-9 here not 1-10
    final = all_stats[all_stats.epoch==final_epoch]
    max_acc = round(final.groupby('fold').Accuracy.mean().mean(), 3)
    max_auc = round(final.groupby('fold').AUC.mean().mean(), 3)

    accuracies[perc] = {'mean': max_acc, 'conf': confidence_score(max_acc, 100)}
    aucs[perc] = {'mean': max_auc, 'conf': confidence_score(max_auc, 100)}

## ACCURACY ##

# deep learning accuracy plot
keys = sorted(accuracies.keys())
x_dl = [i * 80 / 100 for i in keys]
dl_acc = np.array([accuracies[k]['mean'] for k in keys])
dl_acc_conf = np.array([accuracies[k]['conf'] for k in keys])
dl_color = 'mediumblue'
plt.plot(x_dl, dl_acc, label='Deep Learning', color=dl_color)
plt.fill_between(x_dl, dl_acc+dl_acc_conf, dl_acc-dl_acc_conf, alpha=.2, color=dl_color)

# RF accuracy plot
rf_results = pd.read_pickle('random_forest_pt_frac_results.pkl')
x_rf = rf_results.columns*80
acc_confidence = 1.96 * np.sqrt(rf_results.loc['accuracy'] * (1-rf_results.loc['accuracy']) / 100)
acc_fill_y1 = rf_results.loc['accuracy'] + acc_confidence
acc_fill_y2 = rf_results.loc['accuracy'] - acc_confidence
rf_color = 'darkorange'
plt.plot(x_rf, rf_results.loc['accuracy'], color=rf_color, label='Random Forest')
plt.fill_between(x_rf, acc_fill_y1, acc_fill_y2, alpha=.2, color=rf_color)
fig = plt.gcf()

plt.ylabel('Mean Accuracy', fontsize=12)
plt.xlabel('N Training Patients', fontsize=12)
plt.xticks(x_rf, fontsize=10)
plt.legend()
plt.grid(alpha=.2)
plt.savefig('dl-rf-fractional-pt-accuracy-plt.png', dpi=400, bbox_inches='tight')
plt.show()

## AUC ##
plt.close()
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12

# deep learning auc plot
dl_auc = np.array([aucs[k]['mean'] for k in keys])
dl_conf = np.array([aucs[k]['conf'] for k in keys])
dl_color = 'steelblue'
plt.plot(x_dl, dl_auc, label='DNN', color=dl_color, marker='D', markersize=5, linestyle='dashed', zorder=2)
plt.fill_between(x_dl, dl_auc+dl_conf, dl_auc-dl_conf, alpha=.2, color=dl_color)
plt.grid(alpha=.2, axis='y')
fig = plt.gcf()
plt.ylabel('AUC', fontsize=12)
plt.xlabel('N Training Patients', fontsize=12)
plt.xlim([1, 83])
plt.ylim(.48, 1.02)

# RF auc plot
rf_color = 'sandybrown'
rf_auc = rf_results.loc['auc']
rf_conf = 1.96 * np.sqrt(rf_results.loc['auc'] * (1-rf_results.loc['auc']) / 100)
plt.plot(x_rf, rf_auc, label='Random Forest', color=rf_color, marker='s', markersize=5, linestyle='dotted', zorder=1)
plt.fill_between(x_rf, rf_auc+rf_conf, rf_auc-rf_conf, alpha=.2, color=rf_color)
plt.xticks(x_rf, fontsize=9)
plt.yticks(np.arange(.5, 1.01, .05), fontsize=10)

plt.legend(fontsize=12, loc='lower right')
plt.savefig('dl-rf-fractional-pt-auc-plt.png', dpi=400, bbox_inches='tight')
plt.show()
