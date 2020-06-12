import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class LSTMOnlyWithPacking(nn.Module):
    seq_len = 224
    intermediate_features = 64

    def __init__(self, lstm_hidden_units, sub_batches):
        super(LSTMOnlyWithPacking, self).__init__()

        self.lstm_breath_block = nn.LSTM(1, lstm_hidden_units, batch_first=True)
        self.linear_breath_inst = nn.Linear(lstm_hidden_units*self.seq_len, self.intermediate_features)
        self.linear_final = nn.Linear(self.intermediate_features*sub_batches, 2)

    def pack_sequence(self, x):
        # using this packing scheme is pretty fast. but not guaranteed to be correct in
        # unlikely chance a zero shows up before padding begins. It is likely that exp
        # lim of breath is removed. However for a speed tradeoff this is acceptable.
        #
        # I tried using np.where and np.roll in conjunction but this isn't guaranteed to
        # work.
        maxes = np.argmax(x.cpu() == 0, axis=1).numpy().ravel()
        zeroes = np.where(maxes==0)[0]
        if len(zeroes) > 0:
            np.put(maxes, zeroes, 223)
        lens = maxes + 1
        return nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

    def forward(self, x, metadata):
        batches, sub_batches, n_features, seq_len = x.shape
        x = x.reshape([batches, sub_batches, seq_len, n_features])
        lstm_out = self.lstm_breath_block(self.pack_sequence(x[0]))[0]
        outputs, inputs = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=self.seq_len)
        outputs = outputs.unsqueeze(0)
        for i in range(1, batches):
            lstm_out = self.lstm_breath_block(self.pack_sequence(x[i]))[0]
            padding_out, inputs = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=self.seq_len)
            outputs = torch.cat([outputs, padding_out.unsqueeze(0)], dim=0)
        outputs = self.linear_breath_inst(outputs.view([batches, sub_batches, -1]))
        return self.linear_final(outputs.view([batches, -1]))


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


class DoubleLSTMNetwork(nn.Module):
    """
    One LSTM layer network to process all sequences, otherwise use linear layers
    to make the classifications
    """
    seq_len = 224
    intermediate_features = 16

    def __init__(self, lstm_hidden_units, sub_batches):
        super(DoubleLSTMNetwork, self).__init__()

        self.lstm_breath_block = nn.LSTM(1, lstm_hidden_units, batch_first=True)
        self.lstm_batch_block = nn.LSTM(lstm_hidden_units*self.seq_len, self.intermediate_features)
        self.linear_final = nn.Linear(self.intermediate_features*sub_batches, 2)

    def forward(self, x, metadata):
        batches, sub_batches, n_features, seq_len = x.shape
        x = x.reshape([batches, sub_batches, seq_len, n_features])
        batches = x.shape[0]
        outputs = self.lstm_breath_block(x[0])[0].squeeze().unsqueeze(0)
        for i in range(1, batches):
            outputs = torch.cat([outputs, self.lstm_breath_block(x[i])[0].squeeze().unsqueeze(0)], dim=0)
        outputs = self.lstm_batch_block(outputs.view([batches, sub_batches, -1]))[0]
        return self.linear_final(outputs.view([batches, -1]))
