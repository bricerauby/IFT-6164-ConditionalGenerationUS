import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm=nn.BatchNorm2d, conv=nn.Conv2d, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = norm(in_planes)
        self.conv1 = conv(in_planes, planes, kernel_size=3,
                          stride=stride, padding=1, bias=False)
        self.bn2 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3,
                          stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion*planes,
                     kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, norm=nn.BatchNorm2d, conv=nn.Conv2d, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(in_planes, planes, kernel_size=3,
                          stride=stride, padding=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3,
                          stride=1, padding=1, bias=False)
        self.bn2 = norm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion*planes,
                     kernel_size=1, stride=stride, bias=False),
                norm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes,norm=nn.BatchNorm2d, conv=nn.Conv2d, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3,
                          stride=stride, padding=1, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, self.expansion*planes,
                          kernel_size=1, bias=False)
        self.bn3 = norm(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion*planes,
                     kernel_size=1, stride=stride, bias=False),
                norm(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, norm=nn.BatchNorm2d, conv=nn.Conv2d, num_classes=10, in_chans=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv(in_chans, 64, kernel_size=3,
                          stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, norm=nn.BatchNorm2d, conv=nn.Conv2d))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, get_intermediate=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        intermediate = out.view(out.size(0), -1)
        out = self.linear(intermediate)
        if get_intermediate:
            return out, intermediate
        return out


def PreActResNet18(in_chans=3,norm=nn.BatchNorm2d, conv=nn.Conv2d):
    return ResNet(PreActBlock, [2, 2, 2, 2],
                  in_chans=in_chans,
                  conv=conv, norm=norm)


def ResNet18(in_chans=3,norm=nn.BatchNorm2d, conv=nn.Conv2d):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_chans=in_chans,
                  conv=conv, norm=norm)


def ResNet34(in_chans=3,norm=nn.BatchNorm2d, conv=nn.Conv2d):
    return ResNet(BasicBlock, [3, 4, 6, 3], in_chans=in_chans,
                  conv=conv, norm=norm)


def ResNet50(in_chans=3,norm=nn.BatchNorm2d, conv=nn.Conv2d):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_chans=in_chans,
                  conv=conv, norm=norm)


def ResNet101(in_chans=3,norm=nn.BatchNorm2d, conv=nn.Conv2d):
    return ResNet(Bottleneck, [3, 4, 23, 3], in_chans=in_chans,
                  conv=conv, norm=norm)


def ResNet152(in_chans=3,norm=nn.BatchNorm2d, conv=nn.Conv2d):
    return ResNet(Bottleneck, [3, 8, 36, 3], in_chans=in_chans,
                  conv=conv, norm=norm)
