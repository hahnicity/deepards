from torch import nn


class AutoencoderNetwork(nn.Module):
    def __init__(self, base_network):
        self.base_network = base_network
        self.breath_block = base_network.encoder

    def forward(self, x, metadata):
        return self.base_network(x)
