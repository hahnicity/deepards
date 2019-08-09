import torch
from torch.autograd import Variable
import torch.nn as nn


class LSTMOnlyNetwork(nn.Module):
    def __init__(self, lstm_hidden_units):
        super(LSTMOnlyNetwork, self).__init__()
        initial_lstm_layers = 5
        final_lstm_layers = 1

        self.lstm_breath_block = nn.LSTM(224, lstm_hidden_units, batch_first=True)
        self.lstm_final_block = nn.LSTM(lstm_hidden_units, lstm_hidden_units, batch_first=True)
        self.linear_final = nn.Linear(lstm_hidden_units, 2)

    def forward(self, x, metadata):
        batches = x.shape[0]
        outputs = self.lstm_breath_block(x[0])[0].squeeze().unsqueeze(0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.lstm_breath_block(x[i])[0].squeeze().unsqueeze(0)], dim=0)
        outputs = self.lstm_final_block(outputs)[0]
        return self.linear_final(outputs)
