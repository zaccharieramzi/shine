from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import sys
import logging
import functools
from termcolor import colored

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from mdeq_lib.models.mdeq_core import MDEQNet

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        A bottleneck block with receptive field only 3x3. (This is not used in MDEQ; only
        in the classifier layer).
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion, momentum=BN_MOMENTUM, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x) + injection
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

    def _copy(self, other):
        self.conv1.weight.data = other.conv1.weight.data.clone()
        self.conv2.weight.data = other.conv2.weight.data.clone()
        self.conv3.weight.data = other.conv3.weight.data.clone()
        self.bn1.running_mean.data = other.bn1.running_mean.data.clone()
        self.bn1.running_var.data = other.bn1.running_var.data.clone()
        self.bn2.running_mean.data = other.bn2.running_mean.data.clone()
        self.bn2.running_var.data = other.bn2.running_var.data.clone()
        self.bn3.running_var.data = other.bn3.running_var.data.clone()
        self.bn3.running_mean.data = other.bn3.running_mean.data.clone()
        if self.downsample:
            self.downsample[0].weight.data = other.downsample[0].weight.data
            self.downsample[1].weight.data = other.downsample[1].weight.data
            self.downsample[1].bias.data = other.downsample[1].bias.data
            self.downsample[1].running_mean.data = other.downsample[1].running_mean.data
            self.downsample[1].running_var.data = other.downsample[1].running_var.data

class BottleneckGroup(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_groups=4):
        """
        A bottleneck block with receptive field only 3x3. (This is not used in MDEQ; only
        in the classifier layer).
        """
        super(BottleneckGroup, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups, planes, affine=False)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.GroupNorm(num_groups, planes, affine=False)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(num_groups, planes*self.expansion, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.num_groups = num_groups

    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x) + injection
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


class MDEQClsNet(MDEQNet):
    def __init__(self, cfg, opa=False, **kwargs):
        """
        Build an MDEQ Classification model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQClsNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        self.head_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['HEAD_CHANNELS']
        self.final_chansize = cfg['MODEL']['EXTRA']['FULL_STAGE']['FINAL_CHANSIZE']
        self.group_norm = cfg['MODEL']['EXTRA']['FULL_STAGE']['GROUP_NORM']
        self.num_groups = cfg['MODEL']['NUM_GROUPS']

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.num_channels)
        self.classifier = nn.Linear(self.final_chansize, self.num_classes)
        # criterion setting
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.opa = opa

    def _make_head(self, pre_stage_channels):
        """
        Create a classification head that:
           - Increase the number of features in each resolution
           - Downsample higher-resolution equilibria to the lowest-resolution and concatenate
           - Pass through a final FC layer for classification
        """
        if self.group_norm:
            head_block = functools.partial(BottleneckGroup, num_groups=self.num_groups)
            norm = lambda x: nn.GroupNorm(self.num_groups, x, affine=True)
        else:
            head_block = Bottleneck
            norm = lambda x: nn.BatchNorm2d(x, momentum=BN_MOMENTUM)
        d_model = self.init_chansize
        head_channels = self.head_channels

        # Increasing the number of channels on each resolution when doing classification.
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], blocks=1, stride=1, norm=norm)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # Downsample the high-resolution streams to perform classification
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * Bottleneck.expansion
            out_channels = head_channels[i+1] * Bottleneck.expansion

            downsamp_module = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=2, bias=True),
                norm(out_channels),
                nn.ReLU(inplace=True),
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        # Final FC layers
        final_layer = nn.Sequential(
            nn.Conv2d(
                head_channels[len(pre_stage_channels)-1] * Bottleneck.expansion,
                self.final_chansize,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # NOTE: on the advice of Shaojie we keep this
            # no matter if we use group norm
            nn.BatchNorm2d(self.final_chansize, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        return incre_modules, downsamp_modules, final_layer

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, norm=None):
        downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                norm(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def apply_classification_head(self, y_list):
        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)

        # Pool to a 1x1 vector (if needed)
        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)
        return y

    def apply_classification_head_copy(self, y_list):
        # Classification Head
        y = self.incre_modules_copy[0](y_list[0])
        for i in range(len(self.downsamp_modules_copy)):
            y = self.incre_modules_copy[i+1](y_list[i+1]) + self.downsamp_modules_copy[i](y)
        y = self.final_layer_copy(y)

        # Pool to a 1x1 vector (if needed)
        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier_copy(y)
        return y

    def copy_modules(self):
        self.classifier_copy =  nn.Linear(self.final_chansize, self.num_classes)
        self.incre_modules_copy, self.downsamp_modules_copy, self.final_layer_copy = self._make_head(self.num_channels)
        # incre modules
        for i_incr_module in range(len(self.incre_modules)):
            incre_module = self.incre_modules[i_incr_module]
            incre_module_copy = self.incre_modules_copy[i_incr_module]
            for i_layer in range(len(incre_module)):
                layer = incre_module[i_layer]
                layer_copy = incre_module_copy[i_layer]
                layer_copy._copy(layer)

        # downsample modules
        for i_downsamp_module in range(len(self.downsamp_modules)):
            downsamp_module = self.downsamp_modules[i_downsamp_module]
            downsamp_module_copy = self.downsamp_modules_copy[i_downsamp_module]
            downsamp_module_copy[0].weight.data = downsamp_module[0].weight.data.clone()
            downsamp_module_copy[0].bias.data = downsamp_module[0].bias.data.clone()
            downsamp_module_copy[1].weight.data = downsamp_module[1].weight.data.clone()
            downsamp_module_copy[1].bias.data = downsamp_module[1].bias.data.clone()
            downsamp_module_copy[1].running_mean.data = downsamp_module[1].running_mean.data.clone()
            downsamp_module_copy[1].running_var.data = downsamp_module[1].running_var.data.clone()

        # final layer
        self.final_layer_copy[0].weight.data = self.final_layer[0].weight.data.clone()
        self.final_layer_copy[0].bias.data = self.final_layer[0].bias.data.clone()
        self.final_layer_copy[1].weight.data = self.final_layer[1].weight.data.clone()
        self.final_layer_copy[1].bias.data = self.final_layer[1].bias.data.clone()
        self.final_layer_copy[1].running_mean.data = self.final_layer[1].running_mean.data.clone()
        self.final_layer_copy[1].running_var.data = self.final_layer[1].running_var.data.clone()

        # classifier
        self.classifier_copy.weight.data = self.classifier.weight.data.clone()
        self.classifier_copy.bias.data = self.classifier.bias.data.clone()

        copy_modules = [
            self.incre_modules_copy,
            self.downsamp_modules_copy,
            self.final_layer_copy,
            self.classifier_copy,
        ]
        for module in copy_modules:
            for param in module.parameters():
                param.requires_grad_(False)


    def get_fixed_point_loss(self, y_est, true_y):
        loss = self.criterion(self.apply_classification_head_copy(y_est), true_y)
        return loss

    def forward(self, x, train_step=0, **kwargs):
        if self.opa:
            state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state(x.device)
            true_y = kwargs.get('y', None)
            self.copy_modules()
            loss_function = lambda y_est: self.get_fixed_point_loss(y_est, true_y)
            kwargs['loss_function'] = loss_function
            torch.set_rng_state(state)
            torch.cuda.set_rng_state(cuda_state, x.device)
        y_list = self._forward(x, train_step, **kwargs)
        y = self.apply_classification_head(y_list)
        return y

    def init_weights(self, pretrained='',):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class MDEQSegNet(MDEQNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Segmentation model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQSegNet, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        extra = cfg.MODEL.EXTRA

        # Last layer
        last_inp_channels = np.int(np.sum(self.num_channels))
        self.last_layer = nn.Sequential(nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1),
                                        nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(last_inp_channels, cfg.DATASET.NUM_CLASSES, extra.FINAL_CONV_KERNEL,
                                                  stride=1, padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0))

    def forward(self, x, train_step=0, **kwargs):
        y = self._forward(x, train_step, **kwargs)

        # Segmentation Head
        y0_h, y0_w = y[0].size(2), y[0].size(3)
        all_res = [y[0]]
        for i in range(1, self.num_branches):
            all_res.append(F.interpolate(y[i], size=(y0_h, y0_w), mode='bilinear', align_corners=True))

        y = torch.cat(all_res, dim=1)
        all_res = None
        # torch.cuda.empty_cache()
        y = self.last_layer(y)
        return y

    def init_weights(self, pretrained=''):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info(f'=> init weights from normal distribution. PRETRAINED={pretrained}')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()

            # Just verification...
            diff_modules = set()
            for k in pretrained_dict.keys():
                if k not in model_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In ImageNet MDEQ but not Cityscapes MDEQ: {sorted(list(diff_modules))}", "red"))
            diff_modules = set()
            for k in model_dict.keys():
                if k not in pretrained_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In Cityscapes MDEQ but not ImageNet MDEQ: {sorted(list(diff_modules))}", "green"))

            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_cls_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.1
    model = MDEQClsNet(config, **kwargs)
    model.init_weights(config.MODEL.PRETRAINED)
    return model


def get_seg_net(config, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.01
    model = MDEQSegNet(config, **kwargs)
    model.init_weights(config.MODEL.PRETRAINED)
    return model
