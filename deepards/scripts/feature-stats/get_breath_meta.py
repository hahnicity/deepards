import pandas as pd

f = '/home/bhargav/deepards-data-finetuning/data-with-bm-features.pkl'
features = ['iTime', 'eTime', 'inst_RR', 'mean_flow_from_pef', 'I:E ratio']

cohort = pd.read_csv('cohort-description.csv')
patho_dict = {}
count = 0

for _,row in cohort.iterrows():
    if row['Pathophysiology'] == 'ARDS':
        patho_dict[row['Patient Unique Identifier']] = 'ARDS'
    else:
        count += 1
        patho_dict[row['Patient Unique Identifier']] = 'OTHERS'

print(count)


breath_meta = {}
breath_meta['ARDS'] = {}
breath_meta['OTHERS'] = {}
data = pd.read_pickle(f)
all_seq = data.all_sequences
print(len(all_seq))

for seq in all_seq:
    patho = patho_dict[int(seq[0])]
    for i, feature in enumerate(features):
        if feature not in breath_meta[patho]:
            breath_meta[patho][feature] = []
        breath_meta[patho][feature].append(seq[2][i])

#print(len(breath_meta[feature]))
#print("Done reading {}".format(feature))

pd.to_pickle(breath_meta, 'breath_meta_finetuning.pkl')