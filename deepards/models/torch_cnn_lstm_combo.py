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

        # For now just stick with a normal NN. You can add skips later if this is successful
        self.breath_block = breath_block
        self.lstm_hidden_units = 512
        self.lstm_layers = 2
        self.lstm = nn.LSTM(512 * breath_block.expansion, self.lstm_hidden_units, num_layers=self.lstm_layers, batch_first=True)
        self.linear_final = nn.Linear(self.lstm_hidden_units, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.flatten = Flatten()

    def init_hidden(self, batch_size):
        weight = next(self.lstm.parameters()).data
        return (Variable(weight.new(self.lstm_layers, batch_size, self.lstm_hidden_units).zero_()),
                Variable(weight.new(self.lstm_layers, batch_size, self.lstm_hidden_units).zero_()))

    def forward(self, x, hidden):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 64')
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.breath_block(x[0]).unsqueeze(dim=0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.breath_block(x[i]).unsqueeze(dim=0)], dim=0)
        x = self.lstm(outputs, hidden)[0]
        x = self.linear_final(x)
        return self.softmax(x)
