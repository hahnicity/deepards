import torch
from torch.autograd import Variable
import torch.nn as nn


class LSTMOnlyNetwork(nn.Module):
    """
    One LSTM layer network to process all sequences, otherwise use linear layers
    to make the classifications
    """
    seq_len = 224
    intermediate_features = 16

    def __init__(self, lstm_hidden_units, sub_batches):
        super(LSTMOnlyNetwork, self).__init__()

        self.lstm_breath_block = nn.LSTM(1, lstm_hidden_units, batch_first=True)
        self.linear_breath_inst = nn.Linear(lstm_hidden_units*self.seq_len, self.intermediate_features)
        self.linear_final = nn.Linear(self.intermediate_features*sub_batches, 2)

    def forward(self, x, metadata):
        batches, sub_batches, n_features, seq_len = x.shape
        x = x.reshape([batches, sub_batches, seq_len, n_features])
        batches = x.shape[0]
        outputs = self.lstm_breath_block(x[0])[0].squeeze().unsqueeze(0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.lstm_breath_block(x[i])[0].squeeze().unsqueeze(0)], dim=0)
        outputs = self.linear_breath_inst(outputs.view([batches, sub_batches, -1]))
        return self.linear_final(outputs.view([batches, -1]))
