import subprocess
import os

try:
    os.mkdir('pretrained_models')
except OSError:
    pass

bs = 16
epochs = 10
weight_decay = 0.0001
n_times_each_experiment = 3

# XXX need to add processing for unpadded types to siamese pretraining dataset
for dataset_type, train_dataset_path, test_dataset_path in [
    ('padded_breath_by_breath', '/fastdata/deepards/padded_breath_by_breath-contiguous-nb100-train.pkl', '/fastdata/deepards/padded_breath_by_breath-contiguous-nb100-test.pkl'),
    ('unpadded_sequences', '/fastdata/deepards/unpadded_sequences-contiguous-nb55-train.pkl', '/fastdata/deepards/unpadded_sequences-contiguous-nb55-test.pkl'),
    ('unpadded_centered_sequences', '/fastdata/deepards/unpadded_centered_sequences-contiguous-nb55-train.pkl', '/fastdata/deepards/unpadded_centered_sequences-contiguous-nb55-test.pkl')
]:

    for base_network in 'resnet18' 'densenet18' 'se_resnet18' 'vgg11'
        # XXX add connections for siamese cnn linear
        for network in 'siamese_cnn_linear' 'siamese_cnn_lstm' 'siamese_cnn_transformer'
            experiment_name = "siamese_preprocess_{}_{}_{}".format(dataset_type, network, base_network)
            model_path = 'pretrained_models/{}_{}_{}.pth'.format(dataset_type, network, base_network)
            proc = subprocess.Popen([
                'ts', 'python', 'train_ards_detector.py', '--train-from-pickle', dataset_path,
                '-n', network, '--cuda', '-b', bs, '-e', epochs, '--no-print-progress',
                '-exp', experiment_name, '--base-network', base_network, '-wd', weight_decay,
                '--save-model', model_path
            ]) #??--reshuffle-oversample-per-epoch
            proc.communicate()
