import torch
from torch.autograd import Variable
import torch.nn as nn


class SiameseCNNLSTMNetwork(nn.Module):
    def __init__(self, breath_block, lstm_hidden_units):
        super(CNNLSTMNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.lstm_hidden_units = lstm_hidden_units
        self.lstm_layers = 1
        # If you want to use DataParallel and save hidden state in the future then
        # you may not be able to use batch_first=True
        # https://discuss.pytorch.org/t/multi-layer-rnn-with-dataparallel/4450/10
        self.lstm = nn.LSTM(
            breath_block.n_out_filters,
            self.lstm_hidden_units,
            num_layers=self.lstm_layers,
            batch_first=True
        )
        self.linear_final = nn.Linear(self.lstm_hidden_units, 2)

    def _run_through_model(self, inputs):
        batches = inputs.shape[0]
        breaths = inputs.shape[1]
        outputs = self.breath_block(inputs[0]).squeeze().unsqueeze(dim=0)
        for i in range(1, batches):
            block_out = self.breath_block(inputs[i]).squeeze().unsqueeze(dim=0)
            outputs = torch.cat([outputs, block_out], dim=0)
        return self.lstm(outputs)[0]

    def forward(self, x, compr):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        x_out = self._run_through_model(x)
        compr_out = self._run_through_model(compr)

        final = torch.abs(compr_out - x_out)
        return self.linear_final(final)
