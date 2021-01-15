import numpy as np
import torch
from torch.utils.data import DataLoader

from deepards.dataset import ARDSRawDataset
from deepards.ppnet_push import viz_single_prototype


model_path = "saved_models/protopnet_final-epoch6-fold0.pth"
n_prototypes = 8
dataset_path = "/fastdata/deepards/unpadded_centered_with_bm-nb20-kfold.pkl"

x_train = ARDSRawDataset.from_pickle(dataset_path, False, 1.0, None, -1, None)
x_train.set_kfold_indexes_for_fold(0)
train_loader = DataLoader(x_train, batch_size=1, shuffle=False)
model = torch.load(model_path).cuda().eval()

# For shap to work you're going to need to translate the entire dataset from raw data into final
# prototype distances you can do this with the model.seq_forward function. Then after processing,
# roll everything up into its own tensor and you can throw that into sklearn model.
all_outputs = []
all_dists = []
all_targets = []
with torch.no_grad():
    for _, seq, __, target in train_loader:
        inputs = seq.float().cuda()
        outputs, min_distances = model.seq_forward(inputs[0])
        all_outputs.append(outputs.view(-1).unsqueeze(0).cpu().numpy())
        all_dists.append(min_distances.unsqueeze(0).cpu().numpy())
        all_targets.append(target.cpu().numpy())
all_outputs = np.concatenate(all_outputs, axis=0)
all_dists = np.concatenate(all_dists, axis=0)
all_targets = np.concatenate(all_targets, axis=0)


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=[], activation='identity')
# Run fit to initialize all the variables sklearn keeps hidden. We set the weights below so
# this step has no impact on weights
mlp.fit(all_outputs[0:2], all_targets[0:2])

torch_weights = model.last_layer.weight.detach().cpu().numpy().T
mlp.coefs_ = [torch_weights]
# turn off bias
mlp.intercepts_ = [np.array([0, 0])]
# turn off final activation layer.
mlp.out_activation_ = 'identity'

# You can sanity check you're doing it OK by checking outputs versus the pytorch model
print(mlp.predict_proba([all_outputs[1]]))
with torch.no_grad():
    print(model.last_layer(torch.FloatTensor([all_outputs[1]]).cuda()).cpu().numpy())

import pandas as pd
from scipy.special import softmax
import shap
#shap.initjs()

# ensure that you name all the features in the model appropriately
features = []
for i in range(all_outputs.shape[1]):
    proto_n = i % n_prototypes
    # damnit theres some kind of step function here that maps 0-7 -> 0, 8-15 -> 1 ...
    breath_n = (i + n_prototypes - (i % n_prototypes))/n_prototypes - 1
    features.append('breath {}, proto {}'.format(breath_n, proto_n))

X_train = pd.DataFrame(all_outputs, columns=features)
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
# with full dataset the KernelExplainer is dying. probably from memory issues.
explainer = shap.KernelExplainer(mlp.predict_proba, X_train.iloc[0:2000], link="identity")

shap_values = explainer.shap_values(X_train.iloc[0:50], nsamples=64)
model_out = mlp.predict_proba(X_train.iloc[0:50].values)

# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(explainer.expected_value[0], shap_values[0], X_train.iloc[0:50], link="identity")
