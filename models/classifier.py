# ------------------------------------------------------------------------------
# Reference: https://github.com/qjadud1994/DRS/blob/main/models/vgg.py
# ------------------------------------------------------------------------------
"""
BESTIE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}


class PAM(nn.Module):
    def __init__(self, channel):
        super(PAM, self).__init__()
        self.relu = nn.ReLU()
        
        
        ###############
        self.selector = nn.AdaptiveMaxPool2d(1)
        self.selector_Avg = nn.AdaptiveAvgPool2d(1)
        #############

        self.global_max_pool = nn.AdaptiveMaxPool2d(1)        

        ########## term 2
        self.global_avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid(),
        )

        ########## term 3
        self.global_avg_pool3 = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(),
        )


    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.relu(x)
        
        ################term1
        """ 1: - """
        avg_region = self.selector_Avg(x).view(b, c, 1, 1)
        peak_region = self.selector(x).view(b, c, 1, 1)
        jianfa = (peak_region - avg_region)

        jianfa = jianfa.expand_as(x)<0.5
        x = torch.where(jianfa, torch.zeros_like(x), x)
      
        ###########term2
        """ 2: - """
        max = torch.max(x, dim=1)[0].unsqueeze(1)
        mean = torch.mean(x, dim=1).unsqueeze(1)

        jianfa = abs(max-mean)
        jianfa = -jianfa
        bb = torch.exp(jianfa)
        bb = 1-bb

        x = bb*x
                

        ############# term3
        """ 3: max extractor """

        x_max = self.global_max_pool(x).view(b, c, 1, 1)
        x_max = x_max.expand_as(x)
        
        ############ term4

        """ 4: suppression controller"""
        control1 = self.global_avg_pool2(x).view(b, c)
        control1 = self.fc(control1).view(b, c, 1, 1)
        control1 = control1.expand_as(x)

        

        
        ############ term5
        
        """ 5: suppression controller"""
        control3 = self.global_avg_pool3(x).view(b, c)
        control3 = self.fc3(control3).view(b, c, 1, 1)
        control3 = control3.expand_as(x)
        control3[control3< 1] = 1.2

        

        ################
        thres = control1* control3* x_max

        x[x < thres] = 0

        return x
    
    
class VGG(nn.Module):

    def __init__(self, features, num_classes=20, 
                 init_weights=True):
        
        super(VGG, self).__init__()
        
        self.features = features
        
        self.layer1_conv1 = features[0]
        self.layer1_conv2 = features[2]
        self.layer1_maxpool = features[4]
        
        self.layer2_conv1 = features[5]
        self.layer2_conv2 = features[7]
        self.layer2_maxpool = features[9]
        
        self.layer3_conv1 = features[10]
        self.layer3_conv2 = features[12]
        self.layer3_conv3 = features[14]
        self.layer3_maxpool = features[16]
        
        self.layer4_conv1 = features[17]
        self.layer4_conv2 = features[19]
        self.layer4_conv3 = features[21]
        self.layer4_maxpool = features[23]
        
        self.layer5_conv1 = features[24]
        self.layer5_conv2 = features[26]
        self.layer5_conv3 = features[28]
        
        self.extra_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.extra_conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.extra_conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.extra_conv4 = nn.Conv2d(512, 20, kernel_size=1)
        
        self.pam1 = PAM(512)
        self.pam2 = PAM(512)
        self.pam3 = PAM(512)
        self.relu = nn.ReLU(inplace=True)
        
        if init_weights:
            self._initialize_weights(self.extra_conv1)
            self._initialize_weights(self.extra_conv2)
            self._initialize_weights(self.extra_conv3)
            self._initialize_weights(self.extra_conv4)
            
    def forward(self, x, label=None, size=None):
        if size is None:
            size = x.size()[2:]
        
        # layer1
        x = self.layer1_conv1(x)
        x = self.relu(x)
        x = self.layer1_conv2(x)
        x = self.relu(x)
        x = self.layer1_maxpool(x)
        
        # layer2
        x = self.layer2_conv1(x)
        x = self.relu(x)
        x = self.layer2_conv2(x)
        x = self.relu(x)
        x = self.layer2_maxpool(x)
        
        # layer3
        x = self.layer3_conv1(x)
        x = self.relu(x)
        x = self.layer3_conv2(x)
        x = self.relu(x)
        x = self.layer3_conv3(x)
        x = self.relu(x)
        x = self.layer3_maxpool(x)
        
        # layer4
        x = self.layer4_conv1(x)
        x = self.relu(x)
        x = self.layer4_conv2(x)
        x = self.relu(x)
        x = self.layer4_conv3(x)
        x = self.relu(x)
        x = self.layer4_maxpool(x)
        
        # layer5
        x = self.layer5_conv1(x)
        x = self.relu(x)
        x = self.layer5_conv2(x)
        x = self.relu(x)
        x = self.layer5_conv3(x)
        x = self.relu(x)
        # ==============================
        
        x = self.extra_conv1(x)
        x = self.pam1(x)
        x = self.extra_conv2(x)
        x = self.pam2(x)
        x = self.extra_conv3(x)
        x = self.pam3(x)
        x = self.extra_conv4(x)
        # ==============================
        
        logit = self.fc(x)
        
        if self.training:
            return logit
        
        else:
            cam = self.cam_normalize(x.detach(), size, label)
            return logit, cam

    
    def fc(self, x):
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x = x.view(-1, 20)
        return x
    
    
    def cam_normalize(self, cam, size, label):
        B, C, H, W = cam.size()
        
        cam = F.relu(cam)
        cam = cam * label[:, :, None, None]
                
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=False)
        cam /= F.adaptive_max_pool2d(cam, 1) + 1e-5
        
        return cam
    
    
    def _initialize_weights(self, layer):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups

        
        
        
#######################################################################################################
        
    
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16_pam(pretrained=True):
    model = VGG(make_layers(cfg['D1']))
    
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model


if __name__ == '__main__':
    import copy
    
    model = vgg16(pretrained=True)
    print()
    
    print(model)
    
    input = torch.randn(2, 3, 321, 321)
    label = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])
    label = torch.from_numpy(label)
    
    out = model(input, label)
    
    print(out[1].shape)
    
