import argparse
from glob import glob
from warnings import warn

import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns; sns.set()
import numpy as np
from pprint import pprint
from prettytable import PrettyTable
import torch


get_moving_average = lambda x, N: np.convolve(x, np.ones((N,))/N, mode='valid')

def visualize_results_for_start_time(start_time):
    glob_search = 'results/loss_fold_*_{}*'.format(start_time)
    results_files = glob(glob_search)
    if len(results_files) == 0:
        raise Exception('No loss results files found')

    for i, f in enumerate(sorted(results_files)):
        vals = torch.load(f).values.numpy()
        ma = get_moving_average(vals, 100)
        plt.plot(ma, label='Loss Fold {}'.format(i+1))
    plt.legend()
    plt.grid()
    plt.ylabel('loss')
    plt.show()

    glob_search = 'results/loss_epoch_*_fold_*_{}*'.format(start_time)
    results_files = glob(glob_search)
    if len(results_files) == 0:
        raise Exception('No epoch loss results files found')

    for i, f in enumerate(sorted(results_files)):
        vals = torch.load(f).values.numpy()
        ma = get_moving_average(vals, 1)
        plt.plot(ma, label='Loss Fold {}'.format(i+1))
    plt.legend()
    plt.grid()
    plt.ylabel('epoch loss')
    plt.show()

    glob_search = 'results/test_loss_fold*_{}*'.format(start_time)
    results_files = glob(glob_search)
    if len(results_files) == 0:
        results_files = []

    for i, f in enumerate(sorted(results_files)):
        vals = torch.load(f).values.numpy()
        ma = get_moving_average(vals, 100)
        plt.plot(ma, label='Loss Fold {}'.format(i+1))
    plt.legend()
    plt.grid()
    plt.ylabel('loss')
    plt.show()

    all_vals = None
    glob_search = 'results/test_auc*_{}*'.format(start_time)
    results_files = glob(glob_search)
    if len(results_files) == 0:
        raise Exception('No AUC results files found')

    for i, f in enumerate(sorted(results_files)):
        vals = torch.load(f).values.numpy()
        if all_vals is None:
            all_vals = vals
        else:
            all_vals += vals
        plt.plot(vals, label='AUC Fold {}'.format(i+1))
    all_vals = all_vals / len(glob(glob_search))
    plt.plot(all_vals, label='mean_auc')
    plt.legend(loc=3)
    plt.grid()
    plt.ylabel('AUC')
    plt.xlabel('epochs')
    plt.ylim(.45, 1)
    plt.yticks(np.arange(.45, 1.01, .05))
    plt.xticks(range(0, len(all_vals)), range(1, len(all_vals)+1))
    #plt.title(
    plt.show()

    glob_search = 'results/test_accuracy*_{}*'.format(start_time)
    results_files = glob(glob_search)
    if len(results_files) == 0:
        warn('No accuracy results files found')

    for i, f in enumerate(sorted(results_files)):
        vals = torch.load(f).values.numpy()
        ma = get_moving_average(vals, 100)
        plt.plot(ma, label='Test accuracy Fold {}'.format(i+1))
    plt.legend()
    plt.grid()
    plt.ylabel('Accuracy')
    plt.ylim(.5, 1)
    plt.yticks(np.arange(.5, 1.01, .05))
    #plt.title(
    plt.show()

    glob_search = 'results/test_f1_ards*_{}*'.format(start_time)
    results_files = glob(glob_search)
    if len(results_files) == 0:
        raise Exception('No f1 results files found')

    all_vals = None
    for i, f in enumerate(sorted(results_files)):
        vals = torch.load(f).values.numpy()
        if all_vals is None:
            all_vals = vals
        else:
            all_vals += vals
        plt.plot(vals, label='Test ARDS F1 Fold {}'.format(i+1))
    all_vals = all_vals / len(glob(glob_search))
    plt.plot(all_vals, label='Mean F1')
    plt.legend()
    plt.grid()
    plt.ylabel('F1')
    plt.ylim(.5, 1)
    plt.yticks(np.arange(.5, 1.01, .05))
    #plt.title(
    plt.show()


def stats(metric, vals, folds = False):
    row = "runs"
    if folds:
        row = "folds"
    if len(vals) == 0:
        return
    stats_data = {}
    print('---- Overall statistics for {} ----'.format(metric.upper()))
    for run, averages in vals:
        stats_data[run] = {
            'max': np.max(averages).round(4),
            'mean': np.mean(averages).round(4),
            'std': np.std(averages).round(4),
            'epochs_before_max': np.argmax(averages)+1,
            'min': np.min(averages).round(4),
            'median': np.median(averages).round(4),
        }
    cols = list(stats_data[list(stats_data.keys())[0]].keys())
    table = PrettyTable()
    table.title = "Stats for each {}".format(row)
    table.field_names = [row] + cols
    for run, data in stats_data.items():
        table.add_row([run] + list(data.values()))
    print(table)

    table = PrettyTable()
    table.title = "Mean stats across {}".format(row)
    all_means = dict()
    for stat in cols:
        all_means[stat] = []
    for _ , means in stats_data.items():
        for stat, value in means.items():
            all_means[stat].append(value)
    table.field_names = ["stat"] + cols
    table.add_row(["Mean"]+list(np.mean(i).round(4) for i in all_means.values()))
    table.add_row(["Std"]+list(np.std(i).round(4) for i in all_means.values()))
    print(table)

def visualize_results_for_experiment(experiment_name, filter_by_base_network, save, num_folds = None, exp_name_override = None, only_runs = False):
    experiment_files = glob('results/{}_*.pth'.format(experiment_name))
    experiment_data = [torch.load(f) for f in experiment_files]
    for i, exp_data in enumerate(experiment_data):
        if 'n_sub_batches' not in exp_data:
            exp_data['n_sub_batches'] = np.nan

    #experiment_data = sorted(experiment_data, key=lambda x: (x['start_time'], x['base_network']))
    experiment_data = sorted(experiment_data, key=lambda x: x['start_time'])
    if filter_by_base_network:
        tmp = []
        for exp_data in experiment_data:
            if filter_by_base_network == exp_data['base_network']:
                tmp.append(exp_data)
        experiment_data = tmp

    if len(experiment_data) == 0:
        raise Exception('no experiments found with name: {}'.format(experiment_name))

    #if experiment_data[0]['network'] not in ['cnn_regressor']:
    metrics = ['auc', 'patient_accuracy', 'f1_ards', 'f1_other']
    #else:
        #metrics = ['test_mae']

    for metric in metrics:

        metric_data = []
        #plot over folds
        if not only_runs:
            start_times = []
            for i, exp_data in enumerate(experiment_data):
                start_times.append(exp_data['start_time'])
            for fold in range(0, num_folds):
                x = []
                y = []
                vals = None
                for start_time in start_times:
                    metric_files = glob('results/*{}_fold_{}_*_{}.pt'.format(metric, fold, start_time))
                    f = metric_files[0]
                    try:
                        vals_temp = torch.load(f).values.numpy()
                    except:
                        continue

                    for epoch, val in enumerate(vals_temp):
                        x.append(epoch)
                        y.append(val)
                    if vals is None:
                        vals = vals_temp
                    else:
                        vals += vals_temp
                av = vals / len(start_times)
                metric_data.append((fold, av))
                averages = {'epochs': x, metric: y}
                averages = pd.DataFrame(data=averages)
                ax = sns.lineplot(x = 'epochs', y = metric, data = averages, label = 'fold {}'.format(fold))
            stats(metric, metric_data, folds=True)

        #plot over runs
        for i, exp_data in enumerate(experiment_data):
            start_time = exp_data['start_time']

            metric_files = glob('results/*{}_fold*_{}.pt'.format(metric, start_time))
            if len(metric_files) == 0:
                continue
            vals = None
            for f in metric_files:
                if vals is None:
                    vals = torch.load(f).values.numpy()
                else:
                    vals += torch.load(f).values.numpy()
            av = vals / len(metric_files)
            metric_data.append((i, av))
        stats(metric, metric_data)
        #if line_plot:
        x = []
        y = []
        for data_run in metric_data:
            for e, data in enumerate(data_run[1]):
                x.append(e)
                y.append(data)
        averages = {'epochs': x, metric: y}
        averages = pd.DataFrame(data=averages)
        ax = sns.lineplot(x = 'epochs', y = metric, data = averages, label = 'aggregate results')

        plt.legend(loc='lower right', prop={'size': 8})
        sns.set_style('darkgrid')
        if exp_name_override is not None:
                experiment_name = exp_name_override
        plt.title(experiment_name.replace('_', ' '))
        plt.ylabel(metric.replace('_', ' '))
        plt.xlabel('epochs')
        if save:
            if not only_runs:
                file_name = 'plots/{}_{}_{}_{}.png'.format(experiment_name, metric, 'agg', start_time)
            else:
                file_name = 'plots/{}_{}_{}_{}.png'.format( experiment_name, metric, 'runs', start_time)
            plt.savefig(file_name)
            plt.clf()
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser()
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument('-st', '--start-time')
    mutex.add_argument('-exp', '--experiment-name')
    parser.add_argument('--filter-by-base-net', help='filter all results by a base netwwork')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('-eno','--exp-name-override')
    agg_or_runs = parser.add_mutually_exclusive_group(required=True)
    agg_or_runs.add_argument('-nf','--num_folds', type = int)
    agg_or_runs.add_argument('--only-runs', action = 'store_true')
    args = parser.parse_args()

    if args.start_time:
        visualize_results_for_start_time(args.start_time)
    elif args.experiment_name:
        visualize_results_for_experiment(args.experiment_name, args.filter_by_base_net,args.save, args.num_folds, args.exp_name_override, args.only_runs)
        #if args.average_folds:
            #visualize_results_for_experiment(args.experiment_name, args.filter_by_base_net,args.save, True, args.average_folds)
        #else:
            #if not args.line_plot:
                #print("hello")
                #visualize_results_for_experiment(args.experiment_name, args.filter_by_base_net,args.save)
            #else:
                #visualize_results_for_experiment(args.experiment_name, args.filter_by_base_net,args.save, line_plot = args.line_plot)


if __name__ == "__main__":
    main()