import math
import io

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
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

# downsampled breaths
l_rng = 0
u_rng = l_rng+(224*4)
flow_seq = flow[l_rng:u_rng]
new_samples = int(math.ceil(len(flow_seq) / float(4)))
flow_seq = list(resample(flow_seq, new_samples))
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, 234])
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
u_rng = l_rng+(224*4)
flow_seq = flow[l_rng:u_rng]
new_samples = int(math.ceil(len(flow_seq) / float(4)))
flow_seq = list(resample(flow_seq, new_samples))
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, 234])
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
u_rng = l_rng+(224*4)
flow_seq = flow[l_rng:u_rng]
new_samples = int(math.ceil(len(flow_seq) / float(4)))
flow_seq = list(resample(flow_seq, new_samples))
plt.plot(flow_seq, color='darkblue', lw=1.35, label='flow')
plt.xlim([-10, 234])
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
