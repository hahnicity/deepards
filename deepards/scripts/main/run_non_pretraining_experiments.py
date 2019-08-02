import subprocess


bs = 16
epochs = 10
weight_decay = 0.0001
kfolds = 5
n_times_each_experiment = 3

for dataset_type, dataset_path in [('padded_breath_by_breath', '/fastdata/deepards/padded_breath_by_breath-nb100-kfold.pkl'),
                                   ('unpadded_sequences', '/fastdata/deepards/unpadded_sequences-nb150-kfold.pkl'),
                                   ('unpadded_centered_sequences', '/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl')]:
    for base_network in ['resnet18', 'densenet18', 'se_resnet18', 'vgg11']:
        for network in ['cnn_linear', 'cnn_lstm', 'cnn_transformer']:
            experiment_name = "main_experiment_{}_{}_{}".format(dataset_type, network, base_network)
            for i in range(n_times_each_experiment):
                proc = subprocess.Popen([
                    'ts', 'python', 'train_ards_detector.py', '--train-from-pickle', dataset_path,
                    '-n', network, '--cuda', '-b', bs, '-e', epochs, '--no-print-progress',
                    '--kfolds', kfolds, '-exp', experiment_name, '--base-network', base_network,
                    '--oversample', '-wd', weight_decay, '-dt', dataset_type
                ]) #??--reshuffle-oversample-per-epoch
                proc.communicate()
