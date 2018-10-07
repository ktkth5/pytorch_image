import torch
import torch.nn as nn

class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.sSEconv = nn.Conv2d(channel, 1, kernel_size=1)

    def forward(self, x):
        g1 = self.cSE(x)
        g2 = self.sSE(x)
        return y * g1 + y * g2

    def cSE(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

    def sSE(self, x):
        b, c, h, w = x.size()
        y = self.sSEconv(x)
        y = torch.sigmoid(y.view(b, 1, h, w))
        return y