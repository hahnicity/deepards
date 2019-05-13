"""
A pytorch redo of our work to make an ARDS classifier using just breath metadata
"""
import torch
from torch.autograd import Variable
import torch.nn as nn


class MetadataOnlyNetwork(nn.Module):
    def __init__(self):
        super(MetadataOnlyNetwork, self).__init__()

        # These parameters were taken from our original work using grid search to
        # find optimal network parameters for sub batch size of 20. There are a
        # variety of other parameters for different sub batches but this is what
        # we are using.
        self.linear1 = nn.Linear(9, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, _, metadata):
        if torch.any(torch.isnan(metadata)):
            raise Exception('Your metadata has nans in it. Perhaps you are utilizing the wrong dataset type?')
        metadata = metadata.mean(dim=1)
        x = self.linear3(self.linear2(self.linear1(metadata)))
        return self.softmax(x)
