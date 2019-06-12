import torch
from torch.autograd import Variable
import torch.nn as nn


class CNNLSTMNetwork(nn.Module):
    def __init__(self, breath_block, metadata_features, bm_to_linear, lstm_hidden_units):
        super(CNNLSTMNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.lstm_hidden_units = lstm_hidden_units
        self.lstm_layers = 2
        self.bm_to_linear = bm_to_linear
        # If you want to use DataParallel and save hidden state in the future then
        # you may not be able to use batch_first=True
        # https://discuss.pytorch.org/t/multi-layer-rnn-with-dataparallel/4450/10
        if not bm_to_linear:
            self.lstm = nn.LSTM(breath_block.n_out_filters+metadata_features, self.lstm_hidden_units+metadata_features, num_layers=self.lstm_layers, batch_first=True)
            self.linear_final = nn.Linear(self.lstm_hidden_units+metadata_features, 2)
        else:
            self.lstm = nn.LSTM(breath_block.n_out_filters, self.lstm_hidden_units, num_layers=self.lstm_layers, batch_first=True)
            self.linear_final = nn.Linear(self.lstm_hidden_units+metadata_features, 2)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, metadata):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        batches = x.shape[0]
        breaths = x.shape[1]
        outputs = self.breath_block(x[0])
        if not torch.any(torch.isnan(metadata)) and not self.bm_to_linear:
            outputs = torch.cat([outputs, metadata[0]], dim=-1)
        outputs = outputs.unsqueeze(dim=0)

        for i in range(1, batches):
            block_out = self.breath_block(x[i])
            if not torch.any(torch.isnan(metadata)) and not self.bm_to_linear:
                block_out = torch.cat([block_out, metadata[i]], dim=-1)
            outputs = torch.cat([outputs, block_out.unsqueeze(dim=0)], dim=0)

        # [0] just gets the lstm outputs and ignores hx,cx
        print(outputs.size())
        x = self.lstm(outputs)[0]
        if self.bm_to_linear:
            x = torch.cat([x, metadata], dim=-1)
        x = self.linear_final(x)
        return self.softmax(x)
