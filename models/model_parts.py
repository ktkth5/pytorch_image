import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1,
                 activation=nn.ReLU(inplace=True)):
        super(ConvActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SEup(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=2, bilinear=True):
        super(SEup, self).__init__()

        if not bilinear:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)
        self.fc = nn.Sequential(
            nn.Linear(out_ch, out_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // reduction, out_ch),
            nn.Sigmoid()
        )
        self.sSE_param = nn.Parameter(
            torch.randn(out_ch), requires_grad=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bilinear = bilinear

    def forward(self, x1, x2=None):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor= 2, mode="bilinear", align_corners=True)
        else:
            x1 = self.up(x1)
        if x2 is not None:
            diffX = x1.size()[2] - x2.size()[2]
            diffY = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                            diffY // 2, int(diffY / 2)))
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            g1 = self.cSE(x)
            g2 = self.sSE(x)
        else:
            x = self.conv(x1)
            g1 = self.cSE(x)
            g2 = self.sSE(x)
        return x * g1 + x * g2

    def cSE(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

    def sSE(self, x):
        b, c, h, w = x.size()
        y = torch.matmul(self.sSE_param, x.view(b, c, -1))
        y = torch.sigmoid(y.view(b,1,h,w))
        return y

class SEup_AttentionGate(nn.Module):
    def __init__(self, in_ch, out_ch, in_channels, gating_channels, reduction=2, bilinear=True):
        super(SEup_AttentionGate, self).__init__()

        if not bilinear:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)
        self.fc = nn.Sequential(
            nn.Linear(out_ch, out_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // reduction, out_ch),
            nn.Sigmoid()
        )
        self.sSE_param = nn.Parameter(
            torch.randn(out_ch), requires_grad=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bilinear = bilinear
        self.ag = SingleAttentionBlock(in_channels, gating_channels)

    def forward(self, x1, x2=None):
        x2 = self.ag(x2, x1)[0]
        if x1.size(-1) != x2.size(-1):
            if self.bilinear:
                x1 = F.interpolate(x1, scale_factor= 2, mode="bilinear", align_corners=True)
            else:
                x1 = self.up(x1)
        if x2 is not None:
            diffX = x1.size()[2] - x2.size()[2]
            diffY = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                            diffY // 2, int(diffY / 2)))
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
            g1 = self.cSE(x)
            g2 = self.sSE(x)
        else:
            x = self.conv(x1)
            g1 = self.cSE(x)
            g2 = self.sSE(x)
        return x * g1 + x * g2

    def cSE(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

    def sSE(self, x):
        b, c, h, w = x.size()
        y = torch.matmul(self.sSE_param, x.view(b, c, -1))
        y = torch.sigmoid(y.view(b,1,h,w))
        return y

class scSELayer(nn.Module):

    def __init__(self, channel, reduction=2):
        super(scSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.sSE_param = nn.Parameter(
            torch.randn(channel), requires_grad=True)

    def forward(self, x):
        g1 = self.cSE(x)
        g2 = self.sSE(x)
        return x*g1 + x*g2

    def cSE(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

    def sSE(self, x):
        b, c, h, w = x.size()
        y = torch.matmul(self.sSE_param, x.view(b, c, -1))
        y = torch.sigmoid(y.view(b,1,h,w))
        return y


class centerconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(centerconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)


class Dilated_centerconv(nn.Module):

    def __init__(self, in_ch, out_ch, bottleneck_depth=3,
                 bottleneck_type='cascade'):
        super(Dilated_centerconv, self).__init__()
        self.bottleneck_path = nn.ModuleList()
        for i in range(bottleneck_depth):
            bneck_in = in_ch if i == 0 else out_ch
            self.bottleneck_path.append(
                ConvActivation(bneck_in, out_ch, 3,
                               dilation=2 ** i, padding=2 ** i,
                               activation=nn.ReLU(inplace=True)))
        self.bottleneck_type = bottleneck_type
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        dilated_layers = []
        for i, bneck in enumerate(self.bottleneck_path):
            if self.bottleneck_type == 'cascade':
                x = bneck(x)
                dilated_layers.append(x.unsqueeze(-1))
            elif self.bottleneck_type == 'parallel':
                dilated_layers.append(bneck(x.unsqueeze(-1)))
        x = torch.cat(dilated_layers, dim=-1)
        x = torch.sum(x, dim=-1)
        return self.pool(x)

class SingleAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size=None, nonlocal_mode='concatenation',
                 sub_sample_factor=(2,2)):
        super(SingleAttentionBlock, self).__init__()
        self.gate_block_1 = AttentionGate(in_channels=in_size, gating_channels=gate_size,
                                          inter_channels=inter_size, mode=nonlocal_mode,
                                          sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2, mode='concatenation',
                 sub_sample_factor=(2, 2)):
        super(AttentionGate, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Initialise weights
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=False)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)


if __name__=="__main__":
    pass
    res = ResNeXtBottleneck(64, 256, 1, 32, 128, 4)
    print(res)
    # x = torch.randn(3, 64, 16, 16)
    # y = torch.randn(3, 64, 4, 4)
    # ag = AttentionGate(64, 64)
    # out = ag(x, y)
    # print(out[0].shape)
    # print(out[1].shape)
    # print(torch.matmul(w_g, x))
    # d = Dilated_centerconv(512, 256)
    # x = torch.randn(10,512,8,8)
    # out = d(x)
    # print(out.shape)
    # x = torch.cat([x,x,x],0)
    # print(x.shape)
    # print(sc.sSE(x).shape)
    # print(sc.sSE(x))
