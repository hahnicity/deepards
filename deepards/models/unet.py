import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, n_class):
        super(UNet, self).__init__()

        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv1d(64, n_class, 1)

        # add a 5th downconv for pretrained classifiers only
        self.dconv_down5 = nn.Conv1d(512, 512, 3, 2, padding=1)

        self.breath_block = nn.Sequential(
            self.dconv_down1,
            self.maxpool,
            self.dconv_down2,
            self.maxpool,
            self.dconv_down3,
            self.maxpool,
            self.dconv_down4,
            self.maxpool,
            self.dconv_down5,
            nn.MaxPool1d(7),
        )
        self.breath_block.n_out_filters = 512

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        # now is shape bs, 64, 112

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        # now is shape bs, 128, 56

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        # now is shape bs, 256, 24

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


if __name__ == "__main__":
    unet = UNet(1).cuda()
    x = torch.randn((1, 1, 224)).cuda()
    out = unet(x)
    print(out.shape)

    unet = torch.nn.DataParallel(UNet(1).cuda())
    x = torch.randn((8, 1, 224)).cuda()
    out = unet(x)
    print(out.shape)
