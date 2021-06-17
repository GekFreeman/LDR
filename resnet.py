import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import mmd
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import pdb
import scipy.stats as stats
import numpy as np
from collections import OrderedDict
__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

class ADDneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class MFSAN(nn.Module):

    def __init__(self, num_classes=31, _abs=True, sigma=1):
        super(MFSAN, self).__init__()
        self.sharedNet = resnet50(True)
        
        self.block1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(2048, 512, kernel_size=1, bias=False)),
            ("bn1", nn.BatchNorm2d(512)),
            ("relu1", nn.ReLU(inplace=True))
        ]))
        self.block2 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)),
            ("bn1", nn.BatchNorm2d(512)),
            ("relu1", nn.ReLU(inplace=True))
        ]))
        self.block3 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(512, 256, kernel_size=1, bias=False)),
            ("bn1", nn.BatchNorm2d(256)),
            ("relu1", nn.ReLU(inplace=True))
        ]))
        
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.avgpool = nn.AvgPool2d(7, stride=1)        
        self.abs = _abs
        
        self.squash='exp'
        self.min=1
        self.sigma=sigma

    def forward(self, data_src, data_tgt = 0, label_src = 0, stage=1, debug=False, mask1=None, mask2=None, mask3=None):
        mmd_loss = 0
        ldr_loss1 = 0
        ldr_loss2 = 0
        ldr_loss3 = 0
        fm = {}
        if self.training == True:
            if not isinstance(data_tgt, int):
                data_input = torch.cat([data_src, data_tgt], dim=0)
            else:
                data_input = data_src
            x = data_input
            
            for name, module in self._modules.items():
                if name == "sharedNet":
                    x = module(x)
                else:
                    for namex, modulex in module._modules.items():
                        x = modulex(x) 
                        if name == "block1" and namex in ['relu1']:
                            ldr_loss1 += LDR_loss(x, self.sigma)
                        if name == "block2" and namex in ['relu1']:
                            ldr_loss2 += LDR_loss(x, self.sigma)
                        if name == "block3" and namex in ['relu1']:
                            ldr_loss3 += LDR_loss(x, self.sigma)
                    if name == "block3": break
                        
            out = self.avgpool(x)
            out = out.view(out.size(0), -1)
            
            if not isinstance(data_tgt, int):
                data_src, data_tgt_son1 = out.split(data_src.shape[0], dim=0)
                mmd_loss += mmd.mmd(data_src, data_tgt_son1)
            else:
                data_src = out

            pred_src = self.cls_fc_son1(data_src)

            cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

            return cls_loss, mmd_loss, ldr_loss1, ldr_loss2, ldr_loss3

        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.block1(data)

            if stage == -1:
                fm["block1"] = fea_son1.cpu().detach().numpy()
            fea_son1 = self.block2(fea_son1)
            if stage == -1:fm["block2"] = fea_son1.cpu().detach().numpy()
            fea_son1 = self.block3(fea_son1)
            if stage == 3:
                fea_son1 = fea_son1 * mask3
            if stage == -1:fm["block3"] = fea_son1.cpu().detach().numpy()
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            return pred1, fm
                        
def LDR_loss(x, scale=32):
    x = torch.mean(x, dim=[2, 3])
    
    cov=(1/x.size(0))*torch.mm(torch.transpose(x,0,1),x)
    
    normal_weights=np.fromfunction(lambda i, j: stats.norm.pdf(abs(i-j), loc=0, scale=scale)/stats.norm.pdf(0, loc=0, scale=scale), cov.size(), dtype=int)
    normal_weights=torch.Tensor(normal_weights).cuda()
    cov=cov*normal_weights
    F_cov_norm=cov.norm(1)
    diag=torch.eye(cov.size(0)).cuda()
    diag=Variable(diag, requires_grad=False)
    cov_diag=cov*diag

    cov_diag_norm=cov_diag.norm(1)
    decov_loss=(F_cov_norm-cov_diag_norm)
    if decov_loss.data.item()<0:
        return 0
    return decov_loss
    
        
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir="/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/dataset/"))
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir="/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/dataset/"))
    return model
 
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir="/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/dataset/"))
    return model
 
def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir="/userhome/research/continualLearning/deep-transfer-learning/MUDA/MFSAN/dataset/"))
    return model