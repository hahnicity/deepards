from glob import glob
import os

import pandas as pd

from deepards.mean_metrics import confidence_score, get_metrics


# load+process cnn results
results_path = lambda x: os.path.join(os.path.dirname(__file__), '..', 'results', x)
exp = results_path("q1_cnn_linear_benchmark_unpadded_centered_sequences_cnn_linear_densenet18")
start_times = list(set([os.path.splitext(file_.split('_')[-1])[0] for file_ in glob(exp + '*')]))
df_patient_results_list = []
for i, time in enumerate(start_times):
    df = pd.read_pickle(os.path.join(os.path.dirname(__file__), "../results", "{}_patient_results.pkl".format(time)))
    df['model_num'] = i
    df_patient_results_list.append(df)
df_cnn_pt_results = pd.concat(df_patient_results_list)
# just choose patient results from final epoch
df_cnn_pt_results = df_cnn_pt_results[df_cnn_pt_results.epoch_num == 9]
cnn_mispreds = df_cnn_pt_results[df_cnn_pt_results.patho != df_cnn_pt_results.prediction].patient.value_counts()

# process rf results
rf_pts = pd.read_pickle('rf_full_cohort_all_patient_results.pkl')
rf_mispreds = rf_pts[rf_pts.majority_prediction != rf_pts.ground_truth].patient_id.value_counts() / 10  # div 10 normalizes for n cnn trials

common_mispreds = list(set(cnn_mispreds.index).intersection(rf_mispreds.index))
# the following patients cnn improves on
improved_pts = cnn_mispreds.loc[common_mispreds][
    (cnn_mispreds.loc[common_mispreds] < 5) &
    (rf_mispreds.loc[common_mispreds] >= 5)
].index
improved_data = df_cnn_pt_results[df_cnn_pt_results.patient.isin(improved_pts)].groupby('patient').first()

import IPython; IPython.embed()
# based on this analysis we found that cnn improves on 5 patients enough to classify
# them correctly a majority of the time. All of the patients are non-ARDS. This vibes
# with the recorded improvement in specificity.
