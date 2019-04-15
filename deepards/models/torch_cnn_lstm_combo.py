import torch
from torch.autograd import Variable
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNNLSTMNetwork(nn.Module):
    def __init__(self, breath_block):
        super(CNNLSTMNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.lstm_hidden_units = 512
        self.lstm_layers = 2
        # If you want to use DataParallel and save hidden state in the future then
        # you may not be able to use batch_first=True
        # https://discuss.pytorch.org/t/multi-layer-rnn-with-dataparallel/4450/10
        self.lstm = nn.LSTM(512 * breath_block.expansion, self.lstm_hidden_units, num_layers=self.lstm_layers, batch_first=True)
        self.linear_final = nn.Linear(self.lstm_hidden_units, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.flatten = Flatten()

    def forward(self, x):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.breath_block(x[0]).unsqueeze(dim=0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.breath_block(x[i]).unsqueeze(dim=0)], dim=0)
        # [0] just gets the lstm outputs and ignores hx,cx
        x = self.lstm(outputs)[0]
        x = self.linear_final(x)
        return self.softmax(x)
