import torch.nn as nn
import math


def conv2x2(in_planes, out_planes, stride=1):
    """2x2 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv2x2(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2x2(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, initial_planes=64, first_pool_type='max', double_conv_first=False):
        self.inplanes = initial_planes
        self.expansion = block.expansion
        super(ResNet, self).__init__()
        # This divides the input by 2 so your output shape will be (batches, chans, 112)
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_alt = nn.Conv1d(1, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)

        self.conv2 = nn.Conv1d(self.inplanes, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(self.inplanes)
        self.double_conv_first = double_conv_first

        self.relu = nn.ReLU(inplace=True)
        # This also divides the input by 2 so your output shape will be (batches, chans, 56)
        if first_pool_type == 'max':
            self.first_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        elif first_pool_type == 'avg':
            self.first_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        # This layer keeps the same data shape
        self.layer1 = self._make_layer(block, initial_planes, layers[0])
        # This layer divides input seq size by 2 again: (batchs, chans, 28)
        self.layer2 = self._make_layer(block, initial_planes * 2, layers[1], stride=2)
        # This layer divides input seq size by 2 again: (batchs, chans, 14)
        self.layer3 = self._make_layer(block, initial_planes * 4, layers[2], stride=2)
        # This layer divides input seq size by 2 again: (batchs, chans, 7)
        self.layer4 = self._make_layer(block, initial_planes * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.n_out_filters = self.inplanes

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.double_conv_first:
            x = self.conv1(x)
            x = self.bn1(x)
        else:
            x = self.conv1_alt(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)

        x = self.relu(x)
        x = self.first_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        print(self.in_planes)
        print(self.n_out_filters)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model



def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model



def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
