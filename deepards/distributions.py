import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
import seaborn as sns

from deepards.dataset import ARDSRawDataset


def z_filter(data, z=2.5):
    std = np.std(data)
    mean = np.mean(data)
    import IPython; IPython.embed()
    mask = data[np.abs(data) > mean + (z*std)]
    return data[~mask]


def butterworth_filter_simple_dist(data, target, hz_low, hz_high):
    sos = butter(10, (hz_low, hz_high), fs=50, output='sos', btype='bandpass')
    filt = sosfilt(sos, data, axis=-1)
    frame_target = []
    for i in target:
        frame_target.extend([i]*224*20)

    all_frame = np.array([filt.ravel(), frame_target]).T

    ards_dist = all_frame[all_frame[:, 1] == 1][:, 0]
    other_dist = all_frame[all_frame[:, 1] == 0][:, 0]

    import IPython; IPython.embed()
    z_filter(ards_dist)


    #plt.hist(, label='ards', bins=100, alpha=.5)
    #plt.hist(, label='other', bins=100, alpha=.5)
    plt.legend()
    plt.show()


def main():
    dataset = ARDSRawDataset.from_pickle(
        '/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl',
        False, 1.0, None, -1, 0.2, 1.0, None, None, False, False, False, False, False
    )
    dataset.butter_filter = None
    dataset.train = False

    all_data = []
    all_target = []
    for fold in range(5):
        dataset.set_kfold_indexes_for_fold(fold)
        for i in range(len(dataset)):
            _, seq, _, target = dataset[i]
            all_data.append(seq)
            all_target.append(np.argmax(target))
    all_data = np.concatenate(all_data)
    butterworth_filter_simple_dist(all_data, all_target, 10, 15)


if __name__ == "__main__":
    main()
