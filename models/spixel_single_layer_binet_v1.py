import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
from train_util import *
import pdb
import seaborn as sns
import math

# define the function includes in import *
__all__ = [
    'SpixelNet1l','SpixelNet1l_bn'
]
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class C2(nn.Module):
    def __init__(self, feature_channel = 16):
        super(C2, self).__init__()
        self.assin_channel = feature_channel
        self.conv5 = BasicConv2d(self.assin_channel+self.assin_channel, self.assin_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, left, down):
        edge_feature = self.conv5(torch.cat((left, down), dim=1))
        return edge_feature



class EnhancedScreeningModule(nn.Module):
    def __init__(self, feature_channel=1024, intern_channel=32,C2_channel=16):
        super(EnhancedScreeningModule, self).__init__()
        self.feature_channel = feature_channel

        self.ra_conv1 = BasicConv2d(feature_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_conv2 = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_conv3 = BasicConv2d(intern_channel, intern_channel, kernel_size=3, padding=1)
        self.ra_out = nn.Conv2d(intern_channel, 1, kernel_size=3, padding=1)

        self.C2 = C2(feature_channel = C2_channel)
    def forward(self, features, cam_guidance, edge_guidance):
        crop_sal = F.interpolate(cam_guidance, size=features.size()[2:], mode='bilinear', align_corners=True)
        crop_edge = F.interpolate(edge_guidance, size=features.size()[2:], mode='bilinear', align_corners=True)
        x_sal = -1 * (torch.sigmoid(crop_sal)) + 1
        x_sal = self.ra_conv1(x_sal).mul(features)
        edge = self.C2(crop_edge, x_sal)
        return edge
    

class SpixelNet(nn.Module):
    expansion = 1

    def __init__(self,dataset='', batchNorm=True, Train=False):
        super(SpixelNet,self).__init__()
        self.Train = Train

        self.batchNorm = batchNorm
        self.assign_ch = 9
        input_chs=3

        self.conv0a = conv(self.batchNorm, input_chs, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)


        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)


        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)


        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32, 16)
   
        self.esm_3 = ReverseRecalibrationUnit(feature_channel=256, intern_channel=32, C2_channel=32)
        self.sp_conv1 = BasicConv2d(32, 16, kernel_size=3, padding=1)
        self.esm_2 = ReverseRecalibrationUnit(feature_channel=256, intern_channel=16, C2_channel=16)
        self.C2 = C2(feature_channel=16)


        self.pred_mask0 = predict_mask(16,self.assign_ch)
   
        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)



    def forward(self, x):
      
        out1 = self.conv0b(self.conv0a(x)) #5*5
        out2 = self.conv1b(self.conv1a(out1)) #11*11
        out3 = self.conv2b(self.conv2a(out2)) #23*23
        out4 = self.conv3b(self.conv3a(out3)) #47*47
        out5 = self.conv4b(self.conv4a(out4)) #95*95


        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)

        out_esm3 = self.esm_3(out_conv1_1, out5, out2)  # 增加esm unit
        cam_out_3 = F.interpolate(out_esm3, scale_factor=2, mode='bilinear', align_corners=True)  # Sup-2 (b,32,104,104 ->b,32,208, 208)
        cam_out_3 = self.sp_conv1(cam_out_3)

        out_esm2 = self.esm_2(out_conv0_1, out5, out1)  # 增加esm unit

        fin_out = self.C2(out_esm2, cam_out_3)
        mask0 = self.pred_mask0(fin_out)
        prob0 = self.softmax(mask0)
        return prob0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def SpixelNet1l( data=None):
    # Model without  batch normalization
    model = SpixelNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def SpixelNet1l_bn(dataset='BDS500',data=None, Train=False):
    # model with batch normalization
    model = SpixelNet(dataset=dataset,batchNorm=True, Train=Train)
    if data is not None:
        try:
            model.load_state_dict(data)
        except:
            model.load_state_dict(data['state_dict'])
    return model
#
