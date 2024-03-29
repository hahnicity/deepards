import subprocess
import os

try:
    os.mkdir('pretrained_models')
except OSError:
    pass

# bs of 32 takes up about 8400Mb with resnet18. Likely will be worse with bigger models
#
# I've also been seeing that larger batch sizes dont seem to be as problematic as they
# initially seemed upon initial evaluation. I'm not sure why I had so much trouble
# with them. It may because I was using improper learning rates and I thought lower
# learning rates wouldnt be able to train.
bs = str(4)
epochs = str(10)
weight_decay = str(0.0001)

# XXX need to add processing for unpadded types to siamese pretraining dataset
for dataset_type, train_dataset_path, test_dataset_path in [
    ('padded_breath_by_breath', '~/workspace/datasets/deepards/padded_breath_by_breath-continuous-nb100-train.pkl', '~/workspace/datasets/deepards/padded_breath_by_breath-continuous-nb100-test.pkl'),
    ('unpadded_sequences', '~/workspace/datasets/deepards/unpadded_sequences-continuous-nb55-train.pkl', '~/workspace/datasets/deepards/unpadded_sequences-continuous-nb55-test.pkl'),
    ('unpadded_centered_sequences', '~/workspace/datasets/deepards/unpadded_centered_sequences-continuous-nb50-train.pkl', '~/workspace/datasets/deepards/unpadded_centered_sequences-continuous-nb50-test.pkl')
]:

    for base_network in ['resnet18', 'densenet18', 'se_resnet18', 'vgg11']:
        # XXX add connections for siamese cnn linear
        for network in ['siamese_cnn_linear', 'siamese_cnn_lstm', 'siamese_cnn_transformer']:
            model_path = 'pretrained_models/{}_{}_{}.pth'.format(dataset_type, network, base_network)
            proc = subprocess.Popen([
                'ts', 'python', 'train_ards_detector.py', '--train-from-pickle', train_dataset_path,
                '-n', network, '--cuda', '-b', bs, '-e', epochs, '--no-print-progress',
                '--base-network', base_network, '-wd', weight_decay, '--save-model', model_path,
                '-dt', dataset_type, '--test-from-pickle', test_dataset_path,

            ]) #??--reshuffle-oversample-per-epoch
            proc.communicate()
