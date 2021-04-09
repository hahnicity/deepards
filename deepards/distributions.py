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


def setup_butter_filter(hz_low, hz_high):
    if hz_low == 0:
        sos = butter(10, hz_high, fs=50, output='sos', btype='lowpass')
    elif hz_high == 25:
        sos = butter(10, hz_low, fs=50, output='sos', btype='highpass')
    else:
        wn = (hz_low, hz_high)
        sos = butter(10, wn, fs=50, output='sos', btype='bandpass')
    return sos


def butterworth_filt_boxplot(data, target, hz_low, hz_high):
    sos = setup_butter_filter(hz_low, hz_high)
    filt = sosfilt(sos, data, axis=-1)

    frame_target = []
    for i in target:
        frame_target.extend([i]*20)

    rows = []
    frame_target = np.array(frame_target).astype(bool)
    ards_filt = filt[frame_target, :, :]
    other_filt = filt[~frame_target, :, :]
    # 14 is an idx_jump weve used previously
    idx_jump = 14
    for start in range(0, 224, idx_jump):

        end = start + idx_jump
        vals = ards_filt[:, 0, start:end].ravel()
        y = [1] * len(vals)
        start_freq_arr = [int(start)] * len(vals)
        rows.append(np.vstack([vals.real, start_freq_arr, y]).T)

        vals = other_filt[:, 0, start:end].ravel()
        start_freq_arr = [int(start)] * len(vals)
        y = [0] * len(vals)
        rows.append(np.vstack([vals.real, start_freq_arr, y]).T)

    rows = np.concatenate(rows, axis=0)
    rows = pd.DataFrame(rows, columns=['val', 'freq', 'patho'])
    rows.freq = rows.freq.astype(int)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    sns.set_style('white')
    sns.boxplot(x='freq', y='val', hue='patho', palette='Set2', data=rows, showfliers=False, ax=ax)
    plt.title('{}-{}Hz time series distributions'.format(hz_low, hz_high), fontsize=18)
    plt.ylabel('')
    plt.xlabel('Start Idx', fontsize=16)
    ax.grid(axis='y')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Non-ARDS', 'ARDS'], fontsize=16)
    plt.savefig('../img/{}-{}-bandpass-time-series-dist.png'.format(hz_low, hz_high))


def fft_butterworth_filt_boxplot(data, target, hz_low, hz_high):
    sos = butter(10, (hz_low, hz_high), fs=50, output='sos', btype='bandpass')
    filt = sosfilt(sos, data, axis=-1)
    freqs = np.fft.fftshift(np.fft.fftfreq(224, d=0.02))
    freq_mask = np.logical_and(np.abs(freqs) >= hz_low, np.abs(freqs) <= hz_high)
    filt_fft = np.fft.fftshift(np.fft.fft(filt, axis=-1))

    frame_target = []
    for i in target:
        frame_target.extend([i]*20)

    rows = []
    frame_target = np.array(frame_target).astype(bool)
    ards_filt = filt_fft[frame_target, :, :]
    other_filt = filt_fft[~frame_target, :, :]
    ards_filt = ards_filt[:, :, freq_mask]
    other_filt = other_filt[:, :, freq_mask]

    # given that we have a 5 hz bandwidth then this means that 44-46
    # discrete frequencies will be used.
    idx_jump = 2
    for start in range(0, 224, idx_jump):
        start_freq = np.round(freqs[freq_mask][start], 1)
        end = start + idx_jump

        vals = ards_filt[:, 0, start:end].ravel()
        y = [1] * len(vals)
        start_freq_arr = [start_freq] * len(vals)
        rows.append(np.vstack([vals.real, start_freq_arr, y]).T)

        vals = other_filt[:, 0, start:end].ravel()
        start_freq_arr = [start_freq] * len(vals)
        y = [0] * len(vals)
        rows.append(np.vstack([vals.real, start_freq_arr, y]).T)

    rows = np.concatenate(rows, axis=0)
    rows = pd.DataFrame(rows, columns=['val', 'freq', 'patho'])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(10)
    fig.set_figwidth(16)
    sns.boxplot(x='freq', y='val', hue='patho', palette='Set2', data=rows, showfliers=False, ax=ax)
    plt.ylabel('')
    plt.xlabel('Frequency Start', fontsize=16)
    plt.yticks(np.arange(-4, 5), fontsize=14)
    ax.grid(axis='y')
    plt.show()


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
    for low, high in [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15)]:
        butterworth_filt_boxplot(all_data, all_target, low, high)


if __name__ == "__main__":
    main()
