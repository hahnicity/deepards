import torch
from torch.autograd import Variable
import torch.nn as nn

from deepards.models.transformer import Transformer


class CNNToNestedLSTMNetwork(nn.Module):
    def __init__(self, breath_block, window_sequence_size, is_cuda):
        super(CNNToNestedLSTMNetwork, self).__init__()

        self.breath_block = breath_block
        self.window_linear = nn.Linear(self.breath_block.n_out_filters * window_sequence_size, 512)
        self.lstm = nn.LSTM(512, 512, batch_first=True)
        self.linear_final = nn.Linear(512, 2)
        self.is_cuda = is_cuda

    def forward(self, x, metadata):
        # input should be in shape: (n_windows, n_breaths in window, chans, 224)
        #
        # For now only supports patient batch size of 1
        n_patients = x.shape[0]
        if n_patients > 1:
            raise Exception('currently this network only supports patient batch sizes of 1')

        # squeeze patient batch dimension out
        if len(x.shape) == 5:
            x = x.squeeze(0)
        # allocate memory so we dont have to dynamically allocate with cat. However this bit
        # doesnt seem to be the root of our memory issues. Altho it does help make the code a
        # bit cleaner
        outputs = torch.rand([x.shape[-4], x.shape[-3], self.breath_block.n_out_filters])
        if self.is_cuda:
            outputs = outputs.cuda()

        for i in range(x.shape[-4]):
            outputs[i] = self.breath_block(x[i])

        outputs = self.window_linear(outputs.flatten(1)).unsqueeze(0)
        outputs = self.lstm(outputs)[0]
        outputs = self.linear_final(outputs)
        return outputs


class CNNToNestedTransformerNetwork(nn.Module):
    def __init__(self, breath_block, window_sequence_size, is_cuda, transformer_blocks):
        super(CNNToNestedTransformerNetwork, self).__init__()

        self.breath_block = breath_block
        self.window_linear = nn.Linear(self.breath_block.n_out_filters * window_sequence_size, 512)
        self.transformer = Transformer(512, 512, transformer_blocks, 4)
        self.linear_final = nn.Linear(512, 2)
        self.is_cuda = is_cuda

    def forward(self, x, metadata):
        # input should be in shape: (n_windows, n_breaths in window, chans, 224)
        #
        # For now only supports patient batch size of 1
        n_patients = x.shape[0]
        if n_patients > 1:
            raise Exception('currently this network only supports patient batch sizes of 1')

        # squeeze patient batch dimension out
        if len(x.shape) == 5:
            x = x.squeeze(0)
        # allocate memory so we dont have to dynamically allocate with cat. However this bit
        # doesnt seem to be the root of our memory issues. Altho it does help make the code a
        # bit cleaner
        outputs = torch.rand([x.shape[-4], x.shape[-3], self.breath_block.n_out_filters])
        if self.is_cuda:
            outputs = outputs.cuda()

        for i in range(x.shape[-4]):
            outputs[i] = self.breath_block(x[i])

        outputs = self.window_linear(outputs.flatten(1)).unsqueeze(0)
        outputs = self.transformer(outputs)
        outputs = self.linear_final(outputs)
        return outputs
