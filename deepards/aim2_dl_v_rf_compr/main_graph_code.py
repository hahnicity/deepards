import math
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, resample, sosfilt
import seaborn as sns

from ventmap.raw_utils import extract_raw


def remove_spines(ax):
    spines = ['top', 'left', 'right', 'bottom']
    for s in spines:
        ax.spines[s].set_visible(False)


#sns.set_style('whitegrid')
l_rng = 20263
u_rng = 30000-20

breaths = list(extract_raw(io.open('main_graph_vwd.csv'), False))
flow = np.concatenate([b['flow'] for b in breaths])
pressure = np.concatenate([b['pressure'] for b in breaths])

plt.plot(pressure[l_rng:u_rng], color='crimson', lw=1.35, label='pressure')
plt.plot(flow[l_rng:u_rng], color='darkblue', lw=1.35, label='flow')
plt.xlim([-50, u_rng-l_rng])
plt.ylim([-45, 50])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(20, 10)
plt.yticks(fontsize='x-large')
plt.legend(loc='upper left', fontsize='x-large')
fig.axes[0].xaxis.set_ticklabels([])
plt.margins(0, 0)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.savefig('main_fig.svg', dpi=400, pad_inches=0.0, bbox_inches='tight')

# 1st sub img
plt.close()
og_l_rng = l_rng
og_u_rng = u_rng
l_rng = og_l_rng
u_rng = og_l_rng + 500
plt.plot(flow[l_rng:u_rng], color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, u_rng-l_rng])
plt.ylim([-33, 43])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('bi_1.png', dpi=400, bbox_inches='tight', pad_inches=0.0)

# 2nd sub img
plt.close()
l_rng = og_l_rng + 686
u_rng = og_l_rng + 1150
plt.plot(flow[l_rng:u_rng], color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, u_rng-l_rng])
plt.ylim([-29, 39])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('bi_2.png', dpi=400, bbox_inches='tight', pad_inches=0.0)

# 3rd and final sub img
plt.close()
l_rng = og_l_rng + 8940
u_rng = og_l_rng + 9515
plt.plot(flow[l_rng:u_rng], color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, u_rng-l_rng])
plt.ylim([-35, 45])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('bi_3.png', dpi=400, bbox_inches='tight', pad_inches=0.0)

# want to show single breaths: 1st single breath
less_than = []
for b in breaths:
    if len(b['flow']) < 224:
        less_than.append(b)

plt.close()
flow_seq = less_than[1]['flow']
len_seq = len(flow_seq)
flow_seq = np.pad(flow_seq, (0, 224-len_seq), mode='constant')
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-4, 228])
plt.ylim([-45, 65])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('padded_1.png', dpi=400, bbox_inches='tight', pad_inches=0.0)

# 2nd single breath
plt.close()
flow_seq = less_than[7]['flow']
len_seq = len(flow_seq)
flow_seq = np.pad(flow_seq, (0, 224-len_seq), mode='constant')
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-4, 228])
plt.ylim([-29, 45])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('padded_2.png', dpi=400, bbox_inches='tight', pad_inches=0.0)

# 3rd single breath
plt.close()
flow_seq = less_than[20]['flow']
len_seq = len(flow_seq)
flow_seq = np.pad(flow_seq, (0, 224-len_seq), mode='constant')
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-4, 228])
plt.ylim([-40, 10])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('padded_3.png', dpi=400, bbox_inches='tight', pad_inches=0.0)

# 4th single breath
plt.close()
flow_seq = less_than[22]['flow']
len_seq = len(flow_seq)
flow_seq = np.pad(flow_seq, (0, 224-len_seq), mode='constant')
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-4, 228])
plt.ylim([-35, 55])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('padded_4.png', dpi=400, bbox_inches='tight', pad_inches=0.0)

# 5th single breath
plt.close()
flow_seq = less_than[25]['flow']
len_seq = len(flow_seq)
flow_seq = np.pad(flow_seq, (0, 224-len_seq), mode='constant')
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-4, 228])
plt.ylim([-50, 55])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('padded_5.png', dpi=400, bbox_inches='tight', pad_inches=0.0)

# 1st plot continuous
plt.close()
l_rng = 112756
u_rng = l_rng+224
plt.plot(flow[l_rng:u_rng], color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, u_rng-l_rng])
plt.ylim([-33, 43])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('continuous1.png', dpi=400, bbox_inches='tight', pad_inches=0.0)
plt.close()

# 2nd plot continuous
l_rng = 112756+224
u_rng = l_rng+(224)
plt.plot(flow[l_rng:u_rng], color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, u_rng-l_rng])
plt.ylim([-33, 43])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('continuous2.png', dpi=400, bbox_inches='tight', pad_inches=0.0)
plt.close()

# 3rd
l_rng = 112756+(224*2)
u_rng = l_rng+(224)
plt.plot(flow[l_rng:u_rng], color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, u_rng-l_rng])
plt.ylim([-33, 43])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('continuous3.png', dpi=400, bbox_inches='tight', pad_inches=0.0)
plt.close()

# downsampled breaths
l_rng = 0
u_rng = l_rng+(224*10)
flow_seq = flow[l_rng:u_rng]
new_samples = int(math.ceil(len(flow_seq) / float(4)))
flow_seq = list(resample(flow_seq, new_samples))
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, 234*2])
plt.ylim([-33, 43])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('downsampled1.png', dpi=400, bbox_inches='tight', pad_inches=0.0)
plt.close()

# downsampled2
l_rng = u_rng
u_rng = l_rng+(224*10)
flow_seq = flow[l_rng:u_rng]
new_samples = int(math.ceil(len(flow_seq) / float(4)))
flow_seq = list(resample(flow_seq, new_samples))
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, 234*2])
plt.ylim([-33, 43])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('downsampled2.png', dpi=400, bbox_inches='tight', pad_inches=0.0)
plt.close()

# downsampled3
l_rng = u_rng
u_rng = l_rng+(224*10)
flow_seq = flow[l_rng:u_rng]
new_samples = int(math.ceil(len(flow_seq) / float(4)))
flow_seq = list(resample(flow_seq, new_samples))
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, 234*2])
plt.ylim([-33, 43])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('downsampled3.png', dpi=400, bbox_inches='tight', pad_inches=0.0)
plt.close()

# 1st plot continuous centered
plt.close()
l_rng = 112756+19
u_rng = l_rng+224
plt.plot(flow[l_rng:u_rng], color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, u_rng-l_rng])
plt.ylim([-33, 43])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('continuous-centered1.png', dpi=400, bbox_inches='tight', pad_inches=0.0)
plt.close()

l_rng = 112756+19+224+153
u_rng = l_rng+(224)
plt.plot(flow[l_rng:u_rng], color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, u_rng-l_rng])
plt.ylim([-33, 43])
plt.grid(axis='y')
fig = plt.gcf()
fig.set_size_inches(4, 4)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
remove_spines(fig.axes[0])
plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
plt.savefig('continuous-centered2.png', dpi=400, bbox_inches='tight', pad_inches=0.0)
plt.close()

# butterworth filtering
def butter_plots(flow, hz_low, hz_high, l_rng, u_rng, color, do_baseline=False):

    if hz_low == 0:
        sos = butter(2, hz_high, fs=50, output='sos', btype='lowpass')

    elif hz_high == 25:
        sos = butter(2, hz_low, fs=50, output='sos', btype='highpass')

    else:
        wn = (hz_low, hz_high)
        sos = butter(2, wn, fs=50, output='sos', btype='bandpass')

    waveform = flow[l_rng:u_rng]
    signal = sosfilt(sos, waveform)
    pd.Series(waveform).to_csv('butter-{}.csv'.format(l_rng), index=False)
    plt.plot(signal, color=color, lw=1.35, label='flow')
    #plt.ylim([-33, 43])
    #plt.grid(axis='y')
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    fig.axes[0].xaxis.set_ticklabels([])
    fig.axes[0].yaxis.set_ticklabels([])
    remove_spines(fig.axes[0])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
    plt.savefig('butterworth-cc-lrng{}-{}-{}hz.png'.format(l_rng, hz_low, hz_high), dpi=400, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # also perform fft filter and see if that works too.
    freqs = np.fft.fftshift(np.fft.fftfreq(224, d=0.02))
    freq_mask = np.logical_and(np.abs(freqs) > hz_low, np.abs(freqs) < hz_high)  # mask outside frequency bands
    filtered = np.fft.fftshift(np.fft.fft(waveform))
    filtered[~freq_mask] = 0
    recon = np.fft.ifft(np.fft.ifftshift(filtered))

    fix, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(recon, color=color, lw=1.35, label='flow')
    fig.set_size_inches(4, 4)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    remove_spines(ax)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
    plt.savefig('fft-filt-lrng{}-{}-{}hz.png'.format(l_rng, hz_low, hz_high), dpi=400, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    if do_baseline:
        signal = flow[l_rng:u_rng]
        plt.plot(signal, color=color, lw=1.35, label='flow')
        plt.ylim([-33, 43])
        plt.grid(axis='y')
        fig = plt.gcf()
        fig.set_size_inches(4, 4)
        fig.axes[0].xaxis.set_ticklabels([])
        fig.axes[0].yaxis.set_ticklabels([])
        remove_spines(fig.axes[0])
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
        plt.savefig('butterworth-cc-lrng{}-baseline.png'.format(l_rng), dpi=400, bbox_inches='tight', pad_inches=0.0)
        plt.close()


def downsample_plots(flow, factor, l_rng, u_rng, color):
    length = 224
    waveform = flow[l_rng:u_rng]
    len_new = int(length / factor)
    downsamp = resample(waveform, len_new)
    downsamp = np.pad(downsamp, (0, length-len_new))
    plt.plot(downsamp, color=color, lw=1.35, label='flow')
    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    fig.axes[0].xaxis.set_ticklabels([])
    fig.axes[0].yaxis.set_ticklabels([])
    remove_spines(fig.axes[0])
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
    plt.savefig('downsampled-lrng{}-{}x.png'.format(l_rng, factor), dpi=400, bbox_inches='tight', pad_inches=0.0)
    plt.close()


for factor in [1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]:
    l_rng = 112756+19
    u_rng = l_rng+224
    downsample_plots(flow, factor, l_rng, u_rng, 'royalblue')


for hz in [20, 15, 10, 8, 6, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
    # the baseline sequence we are examining is the same one in continuous centered1
    l_rng = 112756+19
    u_rng = l_rng+224
    if hz == 20:
        butter_plots(flow, 0, hz, l_rng, u_rng, 'royalblue', do_baseline=True)
    else:
        butter_plots(flow, 0, hz, l_rng, u_rng, 'royalblue')

    # we are examining continuous centered2
    l_rng = 112756+19+224+153
    u_rng = l_rng+(224)
    if hz == 20:
        butter_plots(flow, 0, hz, l_rng, u_rng, 'darkviolet', do_baseline=True)
    else:
        butter_plots(flow, 0, hz, l_rng, u_rng, 'darkviolet')

    l_rng = og_l_rng + 8940
    u_rng = l_rng+224
    if hz == 20:
        butter_plots(flow, 0, hz, l_rng, u_rng, 'mediumvioletred', do_baseline=True)
    else:
        butter_plots(flow, 0, hz, l_rng, u_rng, 'mediumvioletred')

for hz in [(20, 25), (15, 20), (10, 15), (5, 10), (0, 5)]:
    l_rng = 112756+19
    u_rng = l_rng+224
    butter_plots(flow, hz[0], hz[1], l_rng, u_rng, 'royalblue')

    # we are examining continuous centered2
    l_rng = 112756+19+224+153
    u_rng = l_rng+(224)
    butter_plots(flow, hz[0], hz[1], l_rng, u_rng, 'darkviolet')

    l_rng = og_l_rng + 8940
    u_rng = l_rng+224
    butter_plots(flow, hz[0], hz[1], l_rng, u_rng, 'mediumvioletred')
