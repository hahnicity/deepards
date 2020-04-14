from glob import glob
import os

import matplotlib.pyplot as plt

from deepards.mean_metrics import confidence_score, get_metrics


experiment_prefixes = [
    (100, "results/q1_cnn_linear_benchmark_unpadded_centered_sequences_cnn_linear_densenet18" ),
    (75, "results/fractional_dataset_training75_"),  # 75%
    (50, "results/fractional_dataset_training50_"),  # 50%
    (25, "results/fractional_dataset_training25_"),  # 25%
    (12.5, "results/fractional_dataset_training125_"),  # 12.5%
    (10, "results/fractional_dataset_training1_"),  # 10%
    (7.5, "results/fractional_dataset_training075_"),  # 7.5%
    (5, "results/fractional_dataset_training05_"),  # 5%
    (2.5, "results/fractional_dataset_training025_"),  # 2.5%
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

keys = sorted(accuracies.keys())
x = [i * 80 / 100 for i in keys]
plt.errorbar(x, [accuracies[k]['mean'] for k in keys], yerr=[accuracies[k]['conf'] for k in keys], uplims=True, lolims=True)
plt.ylabel('Mean Accuracy')
plt.xlabel('N Training Patients')
plt.show()


plt.errorbar(x, [aucs[k]['mean'] for k in keys], yerr=[aucs[k]['conf'] for k in keys], uplims=True, lolims=True)
plt.ylabel('Mean AUC')
plt.xlabel('N Training Patients')
plt.show()
