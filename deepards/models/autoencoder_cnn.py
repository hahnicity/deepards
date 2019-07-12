import torch
from torch import nn


class AutoencoderCNN(nn.Module):
    def __init__(self):
        super(AutoencoderCNN, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2, return_indices=True)
        self.conv_down1 = nn.Conv1d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv_down2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv_down3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv_down4 = nn.Conv1d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)

        self.maxunpool = nn.MaxUnpool1d(2)
        self.conv_up1 = nn.ConvTranspose1d(512, 256, 3, padding=1)
        self.conv_up2 = nn.ConvTranspose1d(256, 128, 3, padding=1)
        self.conv_up3 = nn.ConvTranspose1d(128, 64, 3, padding=1)
        self.conv_up4 = nn.ConvTranspose1d(64, 1, 3, padding=1)
        self.n_out_filters = 512

        self.encoder = nn.Sequential(
            self.conv_down1,
            self.bn1,
            nn.MaxPool1d(2),
            self.conv_down2,
            self.bn2,
            nn.MaxPool1d(2),
            self.conv_down3,
            self.bn3,
            nn.MaxPool1d(2),
            self.conv_down4,
            self.bn4,
            nn.MaxPool1d(2),
            nn.MaxPool1d(14),
        )

    def forward(self, x):
        x = self.conv_down1(x)
        x = self.bn1(x)
        x, idx1 = self.maxpool(x)
        x = self.conv_down2(x)
        x = self.bn2(x)
        x, idx2 = self.maxpool(x)
        x = self.conv_down3(x)
        x = self.bn3(x)
        x, idx3 = self.maxpool(x)
        x = self.conv_down4(x)
        x = self.bn4(x)
        x, idx4 = self.maxpool(x)

        x = self.maxunpool(x, idx4)
        x = self.conv_up1(x)
        x = self.maxunpool(x, idx3)
        x = self.conv_up2(x)
        x = self.maxunpool(x, idx2)
        x = self.conv_up3(x)
        x = self.maxunpool(x, idx1)
        x = self.conv_up4(x)
        return x


if __name__ == "__main__":
    ae = AutoencoderCNN().cuda()
    x = torch.randn((1, 1, 224)).cuda()
    out = ae(x)
    print(out.shape)
    out = ae.encoder(x)
    print(out.shape)
