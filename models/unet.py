import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from .model_parts import *
from .oc_attention import *
from library.loss_function import *


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        # self.resnet = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=1)
        self.resnet = pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained="imagenet")
        # print(count_parameters(self.resnet))
        # conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        # for c, c1 in zip(conv0.parameters(), self.resnet.conv1.parameters()):
        #     c.data = c1.data

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            # conv0,
            self.resnet.bn1,
            self.resnet.relu,
            # self.resnet.maxpool
        )
        # self.conv1 = inconv(1, 64)
        # self.con1 = nn
        # print(self.resnet.conv1)

        self.encoder2 = self.resnet.layer1 # 64
        self.encoder3 = self.resnet.layer2 #128
        self.encoder4 = self.resnet.layer3 #256
        self.encoder5 = self.resnet.layer4 #512
        # print(self.resnet.layer1)
        # print(self.resnet.layer2)


        self.center = Dilated_centerconv(512, 256)
        # self.center = centerconv(512, 256)

        self.decoder5 = SEup(256+512, 64)
        self.decoder4 = SEup(  64+256, 64)
        self.decoder3 = SEup(  64+128, 64)
        self.decoder2 = SEup(  64+ 64, 64)
        # self.decoder1 = SEup(  64,     64)

        self.ag5 = SingleAttentionBlock(512,  256)
        self.ag4 = SingleAttentionBlock(256,   64)
        self.ag3 = SingleAttentionBlock(128,   64)
        self.ag2 = SingleAttentionBlock( 64,   64)

        self.logit = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([
            (x - mean[2])/std[2],
            (x - mean[1])/std[1],
            (x - mean[0])/std[0],
        ], 1)

        x = self.conv1(x)
        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)
        # print(f.shape)

        _e5 = self.ag5(e5,  f)[0]
        d5 = self.decoder5( f, _e5)
        _e4 = self.ag4(e4, d5)[0]
        d4 = self.decoder4(d5, _e4)
        _e3 = self.ag3(e3, d4)[0]
        d3 = self.decoder3(d4, _e3)
        _e2 = self.ag2(e2, d3)[0]
        d2 = self.decoder2(d3, _e2)
        # d1 = self.decoder1(d2)

        f = torch.cat((
            d2,
            F.interpolate(d3, scale_factor= 2, mode="bilinear", align_corners=False),
            F.interpolate(d4, scale_factor= 4, mode="bilinear", align_corners=False),
            F.interpolate(d5, scale_factor= 8, mode="bilinear", align_corners=False),
            # F.interpolate(d5, scale_factor=16, mode="bilinear", align_corners=False),
        ), 1)

        f = F.dropout2d(f, p=0.5)
        logit = self.logit(f)
        return logit


class UNet_supervision_dilate(nn.Module):

    def __init__(self):
        super(UNet_supervision_dilate, self).__init__()
        # self.resnet = models.resnet34(pretrained=True)
        # self.resnet = se_ResNeXt.resnext50(2)
        self.resnet = pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained="imagenet")
        # print(self.resnet)

        self.conv1 = nn.Sequential(
            self.resnet.layer0.conv1,
            self.resnet.layer0.bn1,
            self.resnet.layer0.relu1
        )
        self.encoder2 = self.resnet.layer1 # 64
        self.encoder3 = self.resnet.layer2 #128
        self.encoder4 = self.resnet.layer3 #256
        self.encoder5 = self.resnet.layer4 #512

        self.center = Dilated_centerconv(2048, 256)
        # self.center = centerconv(2048, 256)

        self.decoder5 = SEup(  256+2048, 64)
        self.decoder4 = SEup(  64+ 1024, 64)
        self.decoder3 = SEup(  64+  512, 64)
        self.decoder2 = SEup(  64+  256, 64)
        self.decoder1 = SEup(  64,       64)

        self.ag5 = SingleAttentionBlock(2048,  256)
        self.ag4 = SingleAttentionBlock(1024,   64)
        self.ag3 = SingleAttentionBlock( 512,   64)
        self.ag2 = SingleAttentionBlock( 256,   64)

        self.fuse_pixel = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.logit_pixel_d1 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.logit_pixel_d2 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.logit_pixel_d3 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.logit_pixel_d4 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.logit_pixel_d5 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.fuse_image = nn.Sequential(
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True)
        )
        self.logit_image = nn.Sequential(
            nn.Linear(64, 1)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.logit = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch.cat([
            x, x, x
        ], 1)
        # x = add_depth_channels(x)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = torch.cat([
            ((x[:,0,:,:] - mean[2]) / std[2]).unsqueeze(1),
            ((x[:,1,:,:] - mean[1]) / std[1]).unsqueeze(1),
            ((x[:,2,:,:] - mean[0]) / std[0]).unsqueeze(1),
        ], 1)
        # print(x)

        x = self.conv1(x)
        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)

        _e5 = self.ag5(e5,  f)[0]
        d5 = self.decoder5( f, _e5)
        e4 = self.ag4(e4, d5)[0]
        d4 = self.decoder4(d5, e4)
        e3 = self.ag3(e3, d4)[0]
        d3 = self.decoder3(d4, e3)
        e2 = self.ag2(e2, d3)[0]
        d2 = self.decoder2(d3, e2)
        d1 = self.decoder1(d2)

        d5 = F.interpolate(d5, scale_factor=16, mode="bilinear", align_corners=False)
        d4 = F.interpolate(d4, scale_factor= 8, mode="bilinear", align_corners=False)
        d3 = F.interpolate(d3, scale_factor= 4, mode="bilinear", align_corners=False)
        d2 = F.interpolate(d2, scale_factor= 2, mode="bilinear", align_corners=False)

        f = torch.cat((d1, d2, d3, d4, d5), 1)
        f = F.dropout2d(f, p=0.5, training=self.training)
        fuse_pixel = self.fuse_pixel(f)
        logit_pixel_d1 = self.logit_pixel_d1(d1)
        logit_pixel_d2 = self.logit_pixel_d2(d2)
        logit_pixel_d3 = self.logit_pixel_d3(d3)
        logit_pixel_d4 = self.logit_pixel_d4(d4)
        logit_pixel_d5 = self.logit_pixel_d5(d5)
        logit_pixel = [logit_pixel_d1, logit_pixel_d2, logit_pixel_d3,
                       logit_pixel_d4, logit_pixel_d5]

        e = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)
        e = F.dropout(e, p=0.50, training=self.training)
        fuse_image = self.fuse_image(e)
        logit_image = self.logit_image(fuse_image)

        fuse = self.fuse(torch.cat([
            fuse_pixel,
            F.interpolate(fuse_image.view(batch_size, -1, 1, 1), scale_factor=128, mode="nearest")
        ], 1))
        logit = self.logit(fuse)

        return logit, logit_pixel, logit_image

    def criterion(self, logit, logit_pixel, logit_image, truth_pixel, truth_image, is_average=True, bce=False):

        loss_image = F.binary_cross_entropy_with_logits(logit_image, truth_image,
                                                        reduction='elementwise_mean')
        # --
        if bce:
            loss_pixel =  F.binary_cross_entropy(logit_pixel[0], truth_pixel)
            loss_pixel += F.binary_cross_entropy(logit_pixel[1], truth_pixel)
            loss_pixel += F.binary_cross_entropy(logit_pixel[2], truth_pixel)
            loss_pixel += F.binary_cross_entropy(logit_pixel[3], truth_pixel)
            loss_pixel += F.binary_cross_entropy(logit_pixel[4], truth_pixel)
            loss_logit = F.binary_cross_entropy(logit, truth_pixel)
        else:
            loss_pixel =  lovasz_hinge(logit_pixel[0], truth_pixel, elu=False)
            loss_pixel += lovasz_hinge(logit_pixel[1], truth_pixel, elu=False)
            loss_pixel += lovasz_hinge(logit_pixel[2], truth_pixel, elu=False)
            loss_pixel += lovasz_hinge(logit_pixel[3], truth_pixel, elu=False)
            loss_pixel += lovasz_hinge(logit_pixel[4], truth_pixel, elu=False)
            loss_logit = lovasz_hinge(logit, truth_pixel, elu=False)
        # --
        loss_pixel = loss_pixel * truth_image  # loss for empty image is weighted 0
        if is_average:
            loss_pixel = loss_pixel.sum() / truth_image.sum()

        weight_pixel, weight_image, weight_logit = 0.1, 0.05, 1  # lovasz?
        return weight_pixel * loss_pixel, weight_image * loss_image, weight_logit * loss_logit

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)