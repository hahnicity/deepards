import argparse
import subprocess

from deepards.train_ards_detector import base_networks, network_map

bs = str(16)
epochs = str(10)
weight_decay = str(0.0001)
n_times_each_experiment = 10
# found .001 and .01 to be pretty good via experimentation
grad_clip = .01


def run_experiment(train_path, test_path, network, bs, epochs, base_network, weight_decay, dataset_type, dry_run, experiment_name_prefix, n_sub_batches, clip_val):
    experiment_name = "{}_{}_{}_{}_{}".format(
        experiment_name_prefix, dataset_type, network, base_network, n_sub_batches
    )
    command = [str(i) for i in [
        'ts', 'python', 'train_ards_detector.py', '--train-from-pickle', train_path,
        '--test-from-pickle', test_path,
        '-n', network, '--cuda', '-b', bs, '-e', epochs, '--no-print-progress',
        '-exp', experiment_name, '--base-network', base_network,
        '--oversample', '-wd', weight_decay, '-dt', dataset_type, '--clip-grad', '--clip-val',
        clip_val, '-nb', n_sub_batches,
    ]]
    if dry_run:
        print("Running:\n\n{}".format(" ".join(command)))
        return

    for i in range(n_times_each_experiment):
        proc = subprocess.Popen(command) #??--reshuffle-oversample-per-epoch
        proc.communicate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--networks', nargs='+', choices=network_map.keys(), required=True)
    parser.add_argument('-b', '--base-networks', nargs='+', choices=base_networks.keys(), required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('-e', '--experiment-name', required=True)
    parser.add_argument('--clip-val', type=float, default=.01)
    args = parser.parse_args()

    dataset_type = 'unpadded_centered_sequences'
    dataset_path = '/fastdata/deepards/unpadded_centered_sequences-nb{}-train-holdout.pkl'

    for n_sub_batches in range(5, 45, 5):
        train_path = dataset_path.format(n_sub_batches)
        test_path = dataset_path.format(n_sub_batches).replace('train', 'test')

        for network in args.networks:
            for base_network in args.base_networks:
                run_experiment(train_path, test_path, network, bs, epochs, base_network, weight_decay, dataset_type, args.dry_run, args.experiment_name, n_sub_batches, args.clip_val)


if __name__ == "__main__":
    main()
