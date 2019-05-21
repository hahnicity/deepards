from torch import nn


class AutoencoderNetwork(nn.Module):
    def __init__(self, base_network):
        super(AutoencoderNetwork, self).__init__()
        self.base_network = base_network
        self.breath_block = nn.Sequential(
            self.base_network.dconv_down1,
            self.base_network.maxpool,
            self.base_network.dconv_down2,
            self.base_network.maxpool,
            self.base_network.dconv_down3,
            self.base_network.maxpool,
            self.base_network.dconv_down4,
        )

    def forward(self, x, metadata):
        if len(x.shape) != 4:
            raise Exception('Data is not in expected shape')

        x = x.view((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.base_network(x)