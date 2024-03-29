import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class CNNLinearToMean(nn.Module):
    def __init__(self, breath_block):
        super(CNNLinearToMean, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.linear_final = nn.Linear(self.breath_block.n_out_filters, 2)

    def forward(self, x, metadata):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        # XXX add logic to handle metadata eventually
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.breath_block(x[0]).unsqueeze(0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.breath_block(x[i]).unsqueeze(0)], dim=0)
        return self.linear_final(torch.mean(outputs, dim=1))


class CNNLinearComprToRF(nn.Module):
    def __init__(self, breath_block):
        super(CNNLinearComprToRF, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.linear_final = nn.Linear(self.breath_block.n_out_filters, 2)

    def forward(self, x, metadata):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        # XXX add logic to handle metadata eventually
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.breath_block(x[0]).unsqueeze(0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.breath_block(x[i]).unsqueeze(0)], dim=0)
        return self.linear_final(torch.median(outputs, dim=1)[0])


class CNNSingleBreathLinearNetwork(nn.Module):
    def __init__(self, breath_block):
        super(CNNSingleBreathLinearNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.linear_final = nn.Linear(self.breath_block.n_out_filters, 2)

    def forward(self, x, metadata):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        # XXX add logic to handle metadata eventually
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.linear_final(self.breath_block(x[0])).unsqueeze(0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.linear_final(self.breath_block(x[i])).unsqueeze(0)], dim=0)
        return outputs


class CNNDoubleLinearNetwork(nn.Module):
    def __init__(self, breath_block, sequence_size, metadata_features):
        super(CNNDoubleLinearNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.linear_intermediate = nn.Linear(self.breath_block.n_out_filters, 2)
        self.linear_final = nn.Linear(2 * sequence_size + metadata_features, 2)

    def forward(self, x, metadata):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        # XXX add logic to handle metadata eventually
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.linear_final(self.linear_intermediate(self.breath_block(x[0])).view(-1).unsqueeze(0))
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.linear_final(self.linear_intermediate(self.breath_block(x[i])).view(-1).unsqueeze(0))], dim=0)
        return outputs


class CNNLinearNetwork(nn.Module):
    """
    Flatten everything by sub-batch. Then perform a linear layer on it. Get sub-batch level
    predictions.
    """
    def __init__(self, breath_block, sequence_size, metadata_features):
        super(CNNLinearNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.linear_final = nn.Linear(self.breath_block.n_out_filters * sequence_size + metadata_features, 2)

    def forward(self, x, metadata):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.linear_final(self.breath_block(x[0]).view(-1)).unsqueeze(0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.linear_final(self.breath_block(x[i]).view(-1)).unsqueeze(0)], dim=0)
        return outputs


class CNNLinearNetwork2D(nn.Module):
    def __init__(self, breath_block):
        super(CNNLinearNetwork2D, self).__init__()
        self.breath_block = breath_block
        self.linear_final = nn.Linear(self.breath_block.n_out_filters, 2)

    def forward(self, x, metadata):
        x = self.breath_block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.linear_final(torch.flatten(x, 1))
