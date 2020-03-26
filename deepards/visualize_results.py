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

def visualize_results_for_experiment(experiment_name, filter_by_base_network, save, average_folds = False, num_folds = None, line_plot = False):
    experiment_files = glob('results/{}_*.pth'.format(experiment_name))
    #print(experiment_files)
    experiment_data = [torch.load(f) for f in experiment_files]
    for i, exp_data in enumerate(experiment_data):
        if 'n_sub_batches' not in exp_data:
            exp_data['n_sub_batches'] = np.nan

    experiment_data = sorted(experiment_data, key=lambda x: (x['start_time'], x['base_network']))
    if filter_by_base_network:
        tmp = []
        for exp_data in experiment_data:
            if filter_by_base_network == exp_data['base_network']:
                tmp.append(exp_data)
        experiment_data = tmp

    if len(experiment_data) == 0:
        raise Exception('no experiments found with name: {}'.format(experiment_name))

    if experiment_data[0]['network'] not in ['cnn_regressor']:
        metrics = ['auc', 'patient_accuracy', 'f1_ards', 'f1_other']
    else:
        metrics = ['test_mae']

    for i, exp_data in enumerate(experiment_data):
        print('Run {}. Params: {}'.format(i, exp_data))

    for metric in metrics:

        metric_data = []
        #plot over folds
        if average_folds:
            start_times = []
            for i, exp_data in enumerate(experiment_data):
                start_times.append(exp_data['start_time'])
            for fold in range(0, num_folds):
                vals = None
                for start_time in start_times:
                    metric_files = glob('results/*{}_fold_{}_*_{}.pt'.format(metric, fold, start_time))
                    f = metric_files[0]
                    try:
                        vals_temp = torch.load(f).values.numpy()
                    except:
                        continue
                    if vals is None:
                        vals = vals_temp
                    else:
                        vals += vals_temp
                av = vals / len(start_times)
                metric_data.append((fold, av))
                plt.plot(av, label='fold {}'.format(fold))
            stats(metric, metric_data, folds=True)

        #plot over runs
        if not average_folds:
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
                if not line_plot:
                    plt.plot(av, label='run {}'.format(i))
            stats(metric, metric_data)
            if line_plot:
                x = []
                y = []
                for data_run in metric_data:
                    for e, data in enumerate(data_run[1]):
                        x.append(e)
                        y.append(data)
                averages = {'epochs': x, metric: y}
                averages = pd.DataFrame(data=averages)
                #print("hello")
                ax = sns.lineplot(x = 'epochs', y = metric, data = averages)
            #file_name = 'plots/{}_{}_{}_{}.png'.format( experiment_name, metric, 'runs', start_time)
            #ax.savefig(file_name)
            #plt.savefig(file_name)
            #plt.clf()

        plt.legend(loc='lower right', prop={'size': 8})
        #plt.grid()
        sns.set_style('darkgrid')
        plt.title(experiment_name.replace('_', ' '))
        plt.ylabel(metric.replace('_', ' '))
        plt.xlabel('epochs')
        #plt.ylim((0,1))
        if save:
            if average_folds:
                file_name = 'plots/{}_{}_{}_{}.png'.format(experiment_name, metric, ' folds', start_time)
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
    parser.add_argument('--average-folds', type=int)
    parser.add_argument('--line-plot', action='store_true')
    args = parser.parse_args()

    if args.start_time:
        visualize_results_for_start_time(args.start_time)
    elif args.experiment_name:
        if args.average_folds:
            visualize_results_for_experiment(args.experiment_name, args.filter_by_base_net,args.save, True, args.average_folds)
        else:
            if not args.line_plot:
                #print("hello")
                visualize_results_for_experiment(args.experiment_name, args.filter_by_base_net,args.save)
            else:
                visualize_results_for_experiment(args.experiment_name, args.filter_by_base_net,args.save, line_plot = args.line_plot)


if __name__ == "__main__":
    main()
