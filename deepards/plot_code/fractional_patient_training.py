from glob import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepards.mean_metrics import confidence_score, get_metrics


results_path = lambda x: os.path.join(os.path.dirname(__file__), '..', 'results', x)
experiment_prefixes = [
    (100, results_path("q1_cnn_linear_benchmark_unpadded_centered_sequences_cnn_linear_densenet18")),
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
    if len(start_times) != 10:
        continue

    mean_df_stats, all_stats = get_metrics(start_times)
    stats[perc] = all_stats

    max_acc = 0
    max_auc = 0
    for e, sub_df in all_stats.groupby('epoch'):
        if sub_df.Accuracy.mean() > max_acc:
            max_acc = sub_df.Accuracy.mean()
        if sub_df.AUC.mean() > max_auc:
            max_auc = sub_df.AUC.mean()

    accuracies[perc] = {'mean': max_acc, 'conf': confidence_score(max_acc, 100)}
    aucs[perc] = {'mean': max_auc, 'conf': confidence_score(max_auc, 100)}

# deep learning accuracy plot
keys = sorted(accuracies.keys())
x_dl = [i * 80 / 100 for i in keys]
dl_acc = np.array([accuracies[k]['mean'] for k in keys])
dl_acc_conf = np.array([accuracies[k]['conf'] for k in keys])
plt.plot(x_dl, dl_acc, label='DL', color='blue')
plt.fill_between(x_dl, dl_acc+dl_acc_conf, dl_acc-dl_acc_conf, alpha=.2, color='blue')

# RF accuracy plot
rf_results = pd.read_pickle('random_forest_pt_frac_results.pkl')
x_rf = rf_results.columns*80
acc_confidence = 1.96 * np.sqrt(rf_results.loc['accuracy'] * (1-rf_results.loc['accuracy']) / 100)
acc_fill_y1 = rf_results.loc['accuracy'] + acc_confidence
acc_fill_y2 = rf_results.loc['accuracy'] - acc_confidence
plt.plot(x_rf, rf_results.loc['accuracy'], color='orange', label='RF')
plt.fill_between(x_rf, acc_fill_y1, acc_fill_y2, alpha=.2, color='orange')
fig = plt.gcf()
fig.set_size_inches(20, 10)

plt.ylabel('Mean Accuracy')
plt.xlabel('N Training Patients')
plt.xticks(x_rf)
plt.legend()
plt.show()

# deep learning auc plot
dl_auc = np.array([aucs[k]['mean'] for k in keys])
dl_conf = np.array([aucs[k]['conf'] for k in keys])
plt.plot(x_dl, dl_auc, label='DL', color='blue')
plt.fill_between(x_dl, dl_auc+dl_conf, dl_auc-dl_conf, alpha=.2, color='blue')

# RF auc plot
rf_auc = rf_results.loc['auc']
rf_conf = 1.96 * np.sqrt(rf_results.loc['auc'] * (1-rf_results.loc['auc']) / 100)
plt.plot(x_rf, rf_auc, label='RF', color='orange')
plt.fill_between(x_rf, rf_auc+rf_conf, rf_auc-rf_conf, alpha=.2, color='orange')
fig = plt.gcf()
fig.set_size_inches(20, 10)
plt.ylabel('Mean AUC')
plt.xlabel('N Training Patients')
plt.xticks(x_rf)
plt.legend()
plt.show()
