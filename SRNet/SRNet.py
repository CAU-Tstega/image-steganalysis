import torch
import torch.nn as nn



class LayerType1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LayerType1, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=stride,
                              padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out 
 

class LayerType2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LayerType2, self).__init__()
        self.type1 = LayerType1(in_channels=in_channels,
                                out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=stride,
                              padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)


    def forward(self, x):
        out = self.type1(x)
        out = self.bn(self.conv(out))
        out = torch.add(x, out)
        return out


class LayerType3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LayerType3, self).__init__()
        self.type1 = LayerType1(in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride)
        self.conv = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=stride,
                              padding=1)
        self.convs = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.type1(x)
        out = self.pool(self.bn(self.conv(out)))
        res = self.bn(self.convs(x))
        out = torch.add(out, res)
        return out


class LayerType4(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LayerType4, self).__init__()
        self.type1 = LayerType1(in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride)
        self.conv = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.pool = nn.AvgPool2d(kernel_size=16)

    def forward(self, x):
        x = self.type1(x)
        x = self.pool(self.bn(self.conv(x)))
        return x



class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        self.types = {'type1': LayerType1,
                      'type2': LayerType2,
                      'type3': LayerType3,
                      'type4': LayerType4}
        self.layer1 = self._make_layer(types=self.types['type1'], number=2)
        self.layer2 = self._make_layer(types=self.types['type2'], number=5)
        self.layer3 = self._make_layer(types=self.types['type3'], number=4)
        self.layer4 = self._make_layer(types=self.types['type4'], number=1)
        self.ip = nn.Linear(1 * 1 * 512, 2)
        self.reset_parameters()

    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.ip(x)
        return x

    def _make_layer(self, types, number):
        layers = []
        if types == LayerType1:
            print('type = LayerType1')
            out_channels = [64, 16]
            in_channels = [1, 64]
            for i in range(number):
                layers.append(types(in_channels=in_channels[i],
                                    out_channels=out_channels[i]))
        elif types == LayerType2:
            print('type = LayerType2')
            in_channels = 16
            out_channels = 16
            for i in range(number):
                layers.append(types(in_channels=in_channels,
                                    out_channels=out_channels))
        elif types == LayerType3:
            print('type = LayerType3')
            in_channels = [16, 16, 64, 128]
            out_channels = [16, 64, 128, 256]
            for i in range(number):
                layers.append(types(in_channels=in_channels[i],
                                    out_channels=out_channels[i]))
        elif types == LayerType4:
            print('type = LayerType4')
            for i in range(number):
                in_channels = 256
                out_channels = 512
                layers.append(types(in_channels=in_channels,
                                    out_channels=out_channels))
        return nn.Sequential(*layers)

    def reset_parameters(self):
        print('reset_parameters......')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight) 
                m.bias.data.fill_(0.2)
            if isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
    


def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()

