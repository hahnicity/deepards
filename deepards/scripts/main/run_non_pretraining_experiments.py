import subprocess


bs = str(16)
epochs = str(10)
weight_decay = str(0.0001)
kfolds = str(5)
n_times_each_experiment = 10


def run_experiment(dataset_path, network, bs, epochs, kfolds, base_network, weight_decay, dataset_type):
    experiment_name = "main_experiment_{}_{}_{}".format(dataset_type, network, base_network)
    for i in range(n_times_each_experiment):
        proc = subprocess.Popen([
            'ts', 'python', 'train_ards_detector.py', '--train-from-pickle', dataset_path,
            '-n', network, '--cuda', '-b', bs, '-e', epochs, '--no-print-progress',
            '--kfolds', kfolds, '-exp', experiment_name, '--base-network', base_network,
            '--oversample', '-wd', weight_decay, '-dt', dataset_type
        ]) #??--reshuffle-oversample-per-epoch
        proc.communicate()


for dataset_type, dataset_path in [#('padded_breath_by_breath', '/fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl'),
                                   ('unpadded_sequences', '/fastdata/deepards/unpadded_sequences-nb150-kfold.pkl'),
                                   ('unpadded_centered_sequences', '/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl'),

                                   ('unpadded_downsampled_sequences', '/fastdata/deepards/unpadded_downsampled_sequences-nbXXX-kfold.pkl')]:
                                   ]:
    for network in ['cnn_linear_compr_to_rf']:
        if network == 'lstm_only':
            # the base network input doesnt matter here
            run_experiment(dataset_path, network, bs, epochs, kfolds, 'resnet18', weight_decay, dataset_type)
        else:
            for base_network in ['densenet18']:
                run_experiment(dataset_path, network, bs, epochs, kfolds, base_network, weight_decay, dataset_type)
