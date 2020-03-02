import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()

df = pd.read_pickle(args.file)
# first col is pred frac, next is avg dtw, next is max, next is min, next is std
dtw_to_frac = []
for _, pt_rows in df.groupby('patient'):
    patho = pt_rows.iloc[0].y
    for hour in range(24):
        for minute in range(60):
            min_rows = pt_rows[
                (pt_rows.hour >= hour + (minute / 60.0)) &
                (pt_rows.hour < hour + (minute / 60.0) + (1 / 60.0))
            ]
            if len(min_rows) == 0:
                continue

            pred_frac = min_rows.pred.sum() / float(len(min_rows))
            dtw_to_frac.append([pred_frac, patho, min_rows.dtw.mean(), min_rows.dtw.max(), min_rows.dtw.min(), min_rows.dtw.std()])

data = pd.DataFrame(dtw_to_frac, columns=['frac', 'patho', 'mean', 'max', 'min', 'std'])
#sns.scatterplot(x='frac', y='mean', data=data)
#plt.show()
#sns.scatterplot(x='frac', y='max', data=data)
#plt.show()
#sns.scatterplot(x='frac', y='min', data=data)
#plt.show()
#sns.scatterplot(x='frac', y='std', data=data)
#plt.show()
data = data.replace([np.inf, -np.inf], np.nan).dropna()
x2 = sm.add_constant(data[['patho', 'mean', 'max', 'min', 'std']])
est = sm.OLS(data.frac, x2)
res = est.fit()
print(res.summary())
import IPython; IPython.embed()
