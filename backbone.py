# This code is modified from https://github.com/nupurkmr9/S2M2_fewshot
# S2M2_fewshot is itself derived from https://github.com/wyharveychen/CloserLookFewShot
# but lacks proper references to it, including the license
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):  # Residual Block
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        # there is 6 layers in Basic Block
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = None if self.equalInOut else nn.Conv2d(in_planes, out_planes,
                                                                   kernel_size=1, stride=stride,
                                                                   padding=0, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if not self.equalInOut:
            x = out  # x will be pass through a convolution so we ReLU + BN before
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return out + self.convShortcut(x)  # we can sum it (same size !)
        return out + x  # final conv not required (already same size !)

def make_layer(block, in_planes, out_planes, num_layers_per_block, stride, dropRate):
    layers = []
    for i in range(int(num_layers_per_block)):
        if i == 0:
            layers.append(block(in_planes, out_planes, stride, dropRate)) # strides to reduce input size
        else:
            layers.append(block(out_planes, out_planes, 1, dropRate))  # residual blocks with same size
    return nn.Sequential(*layers)

class NetworkBlock(nn.Module):
    def __init__(self, num_layers_per_block, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = make_layer(block, in_planes, out_planes, num_layers_per_block, stride, dropRate)
    def forward(self, x):
        return self.layer(x)

def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    if torch.cuda.is_available():
        y_onehot = y_onehot.cuda()
    y_onehot.zero_()
    x = inp.type(torch.LongTensor)
    if torch.cuda.is_available():
        x = x.cuda()
    x = torch.unsqueeze(x, 1)
    y_onehot.scatter_(1, x , 1)
    return y_onehot

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, strides):
        super(WideResNet, self).__init__()
        dropRate = 0.5
        flatten = True
        num_blocks = 3
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        # We remove 4 layers (first conv, final ReLU, final BN, final Linear)
        # Then we divise by the number of layers per basic block
        assert (depth - 4) % 6 == 0
        num_layers_per_block = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.blocks = nn.ModuleList([NetworkBlock(num_layers_per_block, nChannels[i], nChannels[i+1],
                                                  block, strides[i], dropRate) for i in range(num_blocks)])
        # global average pooling and linear
        self.bn1 = nn.BatchNorm2d(nChannels[-1])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[-1]

        self.num_classes = num_classes
        if flatten:
            self.final_feat_dim = nChannels[-1]
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # custom weight initialization
            elif isinstance(m, nn.BatchNorm2d):  # initialized assuming centered-reduced data
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = x
        out = self.conv1(out)
        for block in self.blocks:
            out = block(out)  # block of ResNet
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.size()[2:])
        out_latent = out.view(out.size(0), -1)  # flatten
        return out_latent


def wrn28_10(num_classes):
    model = WideResNet(depth=28, widen_factor=10, num_classes=num_classes, strides=[1, 2, 2])
    return model
