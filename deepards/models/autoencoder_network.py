from torch import nn


class AutoencoderNetwork(nn.Module):
    def __init__(self, base_network):
        super(AutoencoderNetwork, self).__init__()
        self.base_network = base_network
        self.breath_block = self.base_network.breath_block

    def forward(self, x, metadata):
        if len(x.shape) != 4:
            raise Exception('Data is not in expected shape. Supply a 4 dimensional input with shape <batches, sub-batches, chans, timeseries data>')

        x = x.view((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.base_network(x)
