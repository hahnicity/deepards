import argparse
from subprocess import Popen

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, sosfilt


def main(args):
    freqs = [20, 15, 10, 6, 2]
    breath = pd.read_csv(args.breath_file).values.ravel()
    proc = Popen(['octave', 'generate_matlab_filt.m', args.breath_file])
    proc.communicate()
    matlab_filtered_data = ['matlab-filt{}.csv'.format(i) for i in freqs]
    fig, axes = plt.subplots(ncols=6, nrows=3, figsize=(20, 10))
    ylabs = ['SciPy', 'Matlab', 'Both Overlaid']

    axes[0][0].set_title('baseline')
    for i in range(3):
        axes[i][0].plot(breath)
        axes[i][0].set_ylabel(ylabs[i])

    for k, freq in enumerate(freqs):
        sos = butter(10, freq, fs=50, output='sos', btype='lowpass')
        axes[0][k+1].plot(sosfilt(sos, breath))
        axes[2][k+1].plot(sosfilt(sos, breath))
        matlab_filt = pd.read_csv(matlab_filtered_data[k])
        axes[1][k+1].plot(matlab_filt.values.ravel())
        axes[2][k+1].plot(matlab_filt.values.ravel())
        axes[0][k+1].set_title('{}Hz'.format(freq))
    plt.savefig('breath_file_{}'.format(args.breath_file.replace('.csv', '.png')), dpi=400)
    plt.show()
    pass


parser = argparse.ArgumentParser()
parser.add_argument('breath_file')
args = parser.parse_args()
main(args)
