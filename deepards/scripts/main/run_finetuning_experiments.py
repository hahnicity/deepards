import argparse
import subprocess

bs = str(16)
epochs = str(10)
weight_decay = str(0.0001)
kfolds = str(5)
n_times_each_experiment = 10
# found .001 and .01 to be pretty good via experimentation
grad_clip = .01

dataset_path_train = '~/deepards-data-finetuning/pickled-data/train.pkl'
dataset_path_test = '~/deepards-data-finetuning/pickled-data/test.pkl'
model_paths = [
    'pretrained-model-7.pth',
    'pretrained-model-8.pth',
]


def run_experiment(dataset_path, network, pretrained_model, bs, epochs, kfolds, base_network, weight_decay, dataset_type, dry_run, experiment_name_prefix, n_sub_batches, clip_val, no_pretrain):
    model_name = "_".join((((pretrained_model.split('/')[-1]).split('.')[0]).split('-')[1:]))
    experiment_name = "{}_{}_{}_{}_{}".format(experiment_name_prefix, model_name, dataset_type, network, base_network)
    if no_pretrain == False:
        command = [str(i) for i in [
            'python', 'train_ards_detector.py', '--train-from-pickle', dataset_path_train,
            '--load-base-network', pretrained_model, '-n', network, '--cuda', '-b', bs, '-e', epochs, '--no-print-progress',
            '--kfolds', kfolds, '-exp', experiment_name, '--base-network', base_network,
            '--oversample', '-wd', weight_decay, '-dt', dataset_type, '--clip-grad', '--clip-val',
            clip_val,
        ]]
    else:
        command = [str(i) for i in [
            'python', 'train_ards_detector.py', '--train-from-pickle', dataset_path_train,
            '-n', network, '--cuda', '-b', bs, '-e', epochs, '--no-print-progress',
            '--kfolds', kfolds, '-exp', experiment_name, '--base-network', base_network,
            '--oversample', '-wd', weight_decay, '-dt', dataset_type, '--clip-grad', '--clip-val',
            clip_val,
        ]]
    if dry_run:
        print("Running:\n\n{}".format(" ".join(command)))
        return

    for i in range(n_times_each_experiment):
        proc = subprocess.Popen(command) #??--reshuffle-oversample-per-epoch
        proc.communicate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--dataset-type', nargs='+', choices=['padded_breath_by_breath','unpadded_sequences', 'unpadded_centered_sequences', 'unpadded_downsampled_sequences'], required=True)
    parser.add_argument('-n', '--networks', nargs='+', required=True)
    parser.add_argument('-b', '--base-networks', nargs='+', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('-e', '--experiment-name', required=True)
    parser.add_argument('--clip-val', type=float, default=.01)
    parser.add_argument('--no-pretrain', action='store_true', default=False)
    args = parser.parse_args()

    for dataset_type, dataset_path, n_sub_batches in [
       ('padded_breath_by_breath', 'XXX', 'XXX'),
       ('unpadded_sequences', '/fastdata/deepards/unpadded_sequences-nb150-kfold.pkl', 150),
       ('unpadded_centered_sequences', '/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl', 20),
       ('unpadded_downsampled_sequences', '/fastdata/deepards/unpadded_downsampled_sequences-nbXXX-kfold.pkl', 'XXX')
    ]:
        if dataset_type not in args.dataset_type:
            continue

        for network in args.networks:
            if network == 'lstm_only':
                # the base network input doesnt matter here
                run_experiment(dataset_path, network, bs, epochs, kfolds, 'resnet18', weight_decay, dataset_type, args.dry_run, args.experiment_name, n_sub_batches, args.clip_val)
            else:
                for base_network in args.base_networks:
                    for pretrained_model in model_paths:
                        run_experiment(dataset_path, network, pretrained_model, bs, epochs, kfolds, base_network, weight_decay, dataset_type, args.dry_run, args.experiment_name, n_sub_batches, args.clip_val, args.no_pretrain)


if __name__ == "__main__":
    main()
