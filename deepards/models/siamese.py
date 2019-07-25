import torch
from torch.autograd import Variable
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SiameseARDSClassifier(nn.Module):
    def __init__(self, pretrained_network):
        super(SiameseARDSClassifier, self).__init__()
        self.pretrained_network = pretrained_network
        self.pretrained_network.linear_intermediate = Identity()
        self.pretrained_network.linear_final = nn.Linear(self.pretrained_network.time_layer_hidden_size, 2)

    def forward(self, x, _):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        batches = x.shape[0]
        outputs = self.pretrained_network.breath_block(x[0]).squeeze()
        outputs = outputs.unsqueeze(dim=0)

        for i in range(1, batches):
            block_out = self.pretrained_network.breath_block(x[i]).squeeze()
            block_out = block_out.unsqueeze(dim=0)
            outputs = torch.cat([outputs, block_out], dim=0)

        try:
            x = self.pretrained_network.lstm(outputs)[0]
        except NameError:
            x = self.pretrained_network.transformer(outputs)
        return self.pretrained_network.linear_final(x)


class SiameseCNNTransformerNetwork(nn.Module):
    def __init__(self, breath_block, hidden_units, sub_batch_size):
        super(SiameseCNNTransformerNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        num_blocks = 2
        self.time_layer_hidden_units = hidden_units
        self.transformer = Transformer(breath_block.n_out_filters, hidden_units, num_blocks, 4)
        self.linear_intermediate = nn.Linear(hidden_units, 2)
        self.linear_final = nn.Linear(2 * sub_batch_size, 2)

    def _run_through_model(self, inputs):
        batches = inputs.shape[0]
        breaths = inputs.shape[1]
        outputs = self.breath_block(inputs[0]).squeeze().unsqueeze(dim=0)
        for i in range(1, batches):
            block_out = self.breath_block(inputs[i]).squeeze().unsqueeze(dim=0)
            outputs = torch.cat([outputs, block_out], dim=0)
        return self.transformer(outputs)

    def forward(self, x, compr):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        if x.shape[-1] != 224:
            raise Exception('input breaths must have sequence length of 224')
        x_out = self._run_through_model(x)
        compr_out = self._run_through_model(compr)

        diff = self.linear_intermediate(torch.abs(compr_out - x_out))
        diff = diff.view((diff.shape[0], -1))
        return self.linear_final(diff)


class SiameseCNNLSTMNetwork(nn.Module):
    def __init__(self, breath_block, hidden_units, sub_batch_size):
        super(SiameseCNNLSTMNetwork, self).__init__()

        self.seq_size = 224
        self.breath_block = breath_block
        self.time_layer_hidden_units = hidden_units
        self.lstm_layers = 1
        # If you want to use DataParallel and save hidden state in the future then
        # you may not be able to use batch_first=True
        # https://discuss.pytorch.org/t/multi-layer-rnn-with-dataparallel/4450/10
        self.lstm = nn.LSTM(
            breath_block.n_out_filters,
            hidden_units,
            num_layers=self.lstm_layers,
            batch_first=True
        )
        self.linear_intermediate = nn.Linear(hidden_units, 2)
        self.linear_final = nn.Linear(2 * sub_batch_size, 2)

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

        diff = self.linear_intermediate(torch.abs(compr_out - x_out))
        diff = diff.view((diff.shape[0], -1))
        return self.linear_final(diff)
