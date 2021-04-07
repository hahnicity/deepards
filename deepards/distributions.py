import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
import seaborn as sns

from deepards.dataset import ARDSRawDataset


def z_filter(data, z=4):
    std = np.std(data)
    mean = np.mean(data)
    mask = np.abs(data) <= (mean + (z*std))
    return data[mask]


def fft_butterworth_filt_boxplot(data, target, hz_low, hz_high):
    sos = butter(10, (hz_low, hz_high), fs=50, output='sos', btype='bandpass')
    filt = sosfilt(sos, data, axis=-1)
    freqs = np.fft.fftshift(np.fft.fftfreq(224, d=0.02))
    filt_fft = np.fft.fftshift(np.fft.fft(filt, axis=-1))
    import IPython; IPython.embed()
    #pd.DataFrame(
    sns.boxplot(x='freq', y='val', hue='patho', palette='Set2', showfliers=False)
    plt.show()
    pass


def butterworth_filter_simple_dist(data, target, hz_low, hz_high):
    sos = butter(10, (hz_low, hz_high), fs=50, output='sos', btype='bandpass')
    filt = sosfilt(sos, data, axis=-1)
    frame_target = []
    for i in target:
        frame_target.extend([i]*224*20)

    # this code just shows histograms of data without outliers. The main thing
    # here is that this doesnt show that there is any real difference in distribution
    # between non-ards and ards. So I'm pretty sure standard distribution plots arent
    # the way to visualize this.
    all_frame = np.array([filt.ravel(), frame_target]).T

    ards_dist = all_frame[all_frame[:, 1] == 1][:, 0]
    other_dist = all_frame[all_frame[:, 1] == 0][:, 0]

    ards_dist = z_filter(ards_dist)
    other_dist = z_filter(other_dist)

    plt.hist(ards_dist, label='ards', bins=100, alpha=.5)
    plt.hist(other_dist, label='other', bins=100, alpha=.5)
    plt.legend()
    plt.show()
    plt.close()

    # this just shows a seaborn distplot
    df = pd.DataFrame(np.array([filt.ravel(), frame_target]).T, columns=['obs', 'target'])
    sns.distplot(z_filter(df[df.target==1].obs), label='ards')
    sns.distplot(z_filter(df[df.target==0].obs), label='other')
    plt.legend()
    plt.show()
    plt.close()


def main():
    dataset = ARDSRawDataset.from_pickle(
        '/fastdata/deepards/unpadded_centered_sequences-nb20-kfold.pkl',
        True, 1.0, None, -1, 0.2, 1.0, None, None, False, False, False, False, False
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
    fft_butterworth_filt_boxplot(all_data, all_target, 10, 15)


if __name__ == "__main__":
    main()
