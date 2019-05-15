import argparse
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from prettytable import PrettyTable
import torch


get_moving_average = lambda x, N: np.convolve(x, np.ones((N,))/N, mode='valid')


def visualize_results_for_start_time(start_time):
    glob_search = 'results/loss*_{}*'.format(start_time)
    results_files = glob(glob_search)
    if len(results_files) == 0:
        raise Exception('No loss results files found')

    for i, f in enumerate(sorted(results_files)):
        vals = torch.load(f).values.numpy()
        ma = get_moving_average(vals, 1000)
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
        raise Exception('No accuracy results files found')

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


def visualize_results_for_experiment(experiment_name):
    experiment_files = glob('results/{}*.pth'.format(experiment_name))
    # haven't figured out how to do accuracy yet
    metrics = ['auc', 'f1_ards', 'epoch_test_accuracy']

    for i, filename in enumerate(experiment_files):
        print('Run {}. Params: {}'.format(i, torch.load(filename)))

    for metric in metrics:
        for i, filename in enumerate(experiment_files):
            experiment_params = torch.load(filename)
            start_time = experiment_params['start_time']

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
            plt.plot(av, label='average {} run {}'.format(metric, i))
        plt.legend()
        plt.grid()
        plt.ylabel(metric)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    mutex = parser.add_mutually_exclusive_group(required=True)
    mutex.add_argument('-st', '--start-time')
    mutex.add_argument('-exp', '--experiment-name')
    args = parser.parse_args()

    if args.start_time:
        visualize_results_for_start_time(args.start_time)
    elif args.experiment_name:
        visualize_results_for_experiment(args.experiment_name)


if __name__ == "__main__":
    main()
