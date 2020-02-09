import argparse
import subprocess

from deepards.train_ards_detector import base_networks, network_map

bs = str(16)
epochs = str(10)
weight_decay = str(0.0001)
kfolds = str(5)
n_times_each_experiment = 10
# found .001 and .01 to be pretty good via experimentation
grad_clip = .01


def run_experiment(dataset_path, network, bs, epochs, kfolds, base_network, weight_decay, dataset_type, dry_run, experiment_name_prefix):
    experiment_name = "{}_{}_{}_{}".format(experiment_name_prefix, dataset_type, network, base_network)
    command = [
        'ts', 'python', 'train_ards_detector.py', '--train-from-pickle', dataset_path,
        '-n', network, '--cuda', '-b', bs, '-e', epochs, '--no-print-progress',
        '--kfolds', kfolds, '-exp', experiment_name, '--base-network', base_network,
        '--oversample', '-wd', weight_decay, '-dt', dataset_type, '--clip-grad', '--clip-val',
        grad_clip,
    ]
    if dry_run:
        print("Running:\n\n{}".format(" ".join([str(i) for i in command])))
        return

    for i in range(n_times_each_experiment):
        proc = subprocess.Popen(command) #??--reshuffle-oversample-per-epoch
        proc.communicate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--dataset-type', nargs='+', choices=['unpadded_sequences', 'unpadded_centered_sequences', 'unpadded_downsampled_sequences'], required=True)
    parser.add_argument('-n', '--networks', nargs='+', choices=network_map.keys(), required=True)
    parser.add_argument('-b', '--base-networks', nargs='+', choices=base_networks.keys(), required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('-e', '--experiment-name', required=True)
    args = parser.parse_args()

    for dataset_type, dataset_path in [
       ('unpadded_sequences', '/fastdata/deepards/unpadded_sequences-nb150-kfold.pkl'),
       ('unpadded_centered_sequences', '/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl'),
       ('unpadded_downsampled_sequences', '/fastdata/deepards/unpadded_downsampled_sequences-nbXXX-kfold.pkl')
    ]:
        if dataset_type not in args.dataset_type:
            continue

        for network in args.networks:
            if network == 'lstm_only':
                # the base network input doesnt matter here
                run_experiment(dataset_path, network, bs, epochs, kfolds, 'resnet18', weight_decay, dataset_type, args.dry_run, args.experiment_name)
            else:
                for base_network in args.base_networks:
                    run_experiment(dataset_path, network, bs, epochs, kfolds, base_network, weight_decay, dataset_type, args.dry_run, args.experiment_name)


if __name__ == "__main__":
    main()
