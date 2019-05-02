import torch
from torch.autograd import Variable
import torch.nn as nn


class CNNRegressor(nn.Module):
    def __init__(self, breath_block, n_final_features):
        super(CNNRegressor, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.linear_final = nn.Linear(breath_block.inplanes, n_final_features)

    def forward(self, x, metadata):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        outputs = self.linear_final(self.breath_block(x))
        return outputs
