import torch
from torch.autograd import Variable
import torch.nn as nn


class CNNLinearNetwork(nn.Module):
    def __init__(self, breath_block, sequence_size):
        super(CNNLinearNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.linear_final = nn.Linear(self.breath_block.inplanes * sequence_size, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, metadata):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        # XXX add logic to handle metadata eventually
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.linear_final(self.breath_block(x[0]).view(-1)).unsqueeze(0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.linear_final(self.breath_block(x[i]).view(-1)).unsqueeze(0)], dim=0)
        return self.softmax(outputs)
