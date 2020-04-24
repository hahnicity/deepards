import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
from scipy import stats

feature_list = []
bm_pt = pd.read_pickle('~/deepards/deepards/breath_meta.pkl')
bm_ft = pd.read_pickle('~/deepards/deepards/breath_meta_finetuning.pkl')

features = ['iTime', 'eTime', 'inst_RR', 'mean_flow_from_pef', 'I:E ratio']
plots = [(bm_pt,'PT'), (bm_ft['ARDS'],'FT_ARDS'), (bm_ft['OTHERS'],'FT_OTHERS')]

for feature in features:
    print('---------------')
    print(feature+":")
    print('---------------')
    for vals,plot in plots:
        print(plot+':')
        vals = vals[feature]
        v_min = min(vals)
        v_max = max(vals)
        v_range = v_max - v_min
        vals = (vals - v_min)/(v_max - v_min)
        #print(len(vals))
        #v_min = min(vals)
        #v_max = max(vals)
        #v_range = v_max - v_min
        #vals = vals/np.linalg.norm(vals)
        mean = np.mean(vals).round(4)
        mode = stats.mode(vals)
        iqr = stats.iqr(vals)
        median = np.median(vals).round(4)
        std = np.std(vals).round(4)
        print("range: {}".format(v_range))
        print("mean: {}".format(mean))
        print("median: {}".format(median))
        print("mode: {}".format(mode))
        print("std: {}".format(std))
        print("iqr: {}".format(iqr))
        print("-----------------")

        sns.distplot(vals, hist = False, kde = True, label = plot)
    plt.legend()
    plt.ylabel('density')
    plt.xlabel(feature)
    name = "scripts/feature-stats/{}.png".format(feature)
    plt.title("Density plot for {}".format(feature))
    plt.savefig(name)
    plt.clf()

