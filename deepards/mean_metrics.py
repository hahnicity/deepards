import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

def computeMetricsFromPatientResults(df, df_stats):
    epochs = df.epoch_num.unique()
    folds = df.fold_num.unique()
    #df_stats = pd.DataFrame(columns = ['fold','epoch','AUC', 'Accuracy', 'sensitivity', 'specificity', 'precision', 'f1'])
    for fold in folds:
        for epoch in epochs:
            sub_df = df.loc[(df['fold_num'] == fold) & (df['epoch_num'] == epoch)]
            y_pred = sub_df.prediction.tolist()
            y_true = sub_df.patho.tolist()
            y_scores = sub_df.pred_frac.tolist()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            auc = roc_auc_score(y_true, y_scores)
            try:
                accuracy = round(((tp + tn)/(tp + tn + fp + fn)),4)
            except ZeroDivisionError:
                accuracy = 0
            try:
                sensitivity = round((tp/(tp + fn)),4)
            except ZeroDivisionError:
                sensitivity = 0
            try:
                specificity = round((tn/(tn + fp)),4)
            except ZeroDivisionError:
                specificity = 0
            try:
                precision = round((tp/(tp + fp)),4)
            except ZeroDivisionError:
                precision = 0
            try:
                f1 = f1 = round(2 *((precision * sensitivity) / (precision + sensitivity)), 4)
            except ZeroDivisionError:
                f1 = 0
            row = {'fold': fold,'epoch': epoch,'AUC': auc, 'Accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'f1': f1}
            df_stats = df_stats.append(row, ignore_index = True)
    
    return df_stats

def getMeanMetrics(start_times):
    df_patient_results_list = []
    for time in start_times:
        df = pd.read_pickle("{}_patient_results.pkl".format(time))
        df_patient_results_list.append(df)
    df_stats = pd.DataFrame(columns = ['fold','epoch','AUC', 'Accuracy', 'sensitivity', 'specificity', 'precision', 'f1'])
    for df in df_patient_results_list:
        df_stats = computeMetricsFromPatientResults(df, df_stats)

    mean_df_stats = df_stats.groupby(['fold', 'epoch'], as_index = False).mean().round(4)
    mean_df_stats = mean_df_stats.sort_values('AUC', ascending = False).drop_duplicates('fold')
    mean_df_stats = mean_df_stats.sort_values('fold')
    mean_df_stats[['fold', 'epoch']] = mean_df_stats[['fold', 'epoch']].astype(int)
    mean_df_stats = mean_df_stats.reset_index(drop = True)
    mean_df_stats.rename(columns = {'epoch' : 'max_epoch'}, inplace = True)
    return mean_df_stats

if __name__ == "__main__":

    start_times = ['1565056824', '1565056824', '1565090993']
    
    mean_df_stats = getMeanMetrics(start_times)
    print(mean_df_stats)



    
