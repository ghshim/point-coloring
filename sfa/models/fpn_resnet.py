from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

BN_MOMENTUM = 0.1 # 배치 정규화 (Batch Normalization)의 모멘텀 하이퍼파라미터 값을 0.1로 설정합니다.

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
# 3x3 컨볼루션 레이어를 생성하는 함수를 정의합니다. 
# 이 함수는 레이어의 입력 채널 수 (in_planes), 출력 채널 수 (out_planes), 스트라이드(stride)를 인수로 받아 3x3 컨볼루션 레이어를 반환합니다. 
# 이 레이어는 패딩(padding)이 1이며, 편향(bias)을 사용하지 않습니다.

class BasicBlock(nn.Module):
    expansion = 1 # 이 변수는 나중에 블록 내에서 채널 수를 확장하는 데 사용됩니다.

    def __init__(self, inplanes, planes, stride=1, downsample=None):
    # BasicBlock 클래스의 초기화 메서드입니다. 이 메서드는 BasicBlock의 각 인스턴스를 생성할 때 호출됩니다.
    # inplanes: 입력 채널 수 (이전 레이어의 출력 채널 수)
    # planes: 현재 레이어의 출력 채널 수
    # stride: 컨볼루션 레이어의 스트라이드 값 (기본값은 1)
    # downsample: 잔여 연결의 다운샘플링(downsample) 레이어 (기본값은 None)

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # conv1은 3x3 합성곱(Convolution) 레이어를 정의하는데, 이 레이어는 입력 데이터의 특징을 추출하는 역할을 합니다.
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x) # 첫 번째 컨볼루션 레이어를 통과한 결과를 out에 할당합니다. 
        out = self.bn1(out) # 배치 정규화를 적용한 후 결과를 out에 할당합니다.
        out = self.relu(out) # ReLU 활성화 함수를 적용한 후 결과를 out에 할당합니다.

        out = self.conv2(out) # 두 번째 컨볼루션 레이어를 통과한 결과를 out에 할당합니다.
        out = self.bn2(out) # 배치 정규화를 적용한 후 결과를 out에 할당합니다.

        if self.downsample is not None:
            residual = self.downsample(x)
        # 만약 다운샘플링 레이어가 지정되어 있다면, 잔여 연결을 다운샘플링합니다. 그렇지 않으면 잔여 연결은 그대로 유지됩니다.

        out += residual # 잔여 연결을 현재 블록의 출력에 누적합니다.
        out = self.relu(out) # 최종 출력에 ReLU 활성화 함수를 다시 적용합니다.

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
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


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv_up_level1 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        self.conv_up_level2 = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
        self.conv_up_level3 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)

        fpn_channels = [256, 128, 64]
        for fpn_idx, fpn_c in enumerate(fpn_channels):
            for head in sorted(self.heads):
                num_output = self.heads[head]
                if head_conv > 0:
                    fc = nn.Sequential(
                        nn.Conv2d(fpn_c, head_conv, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
                else:
                    fc = nn.Conv2d(in_channels=fpn_c, out_channels=num_output, kernel_size=1, stride=1, padding=0)

                self.__setattr__('fpn{}_{}'.format(fpn_idx, head), fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, input_h, input_w = x.size()
        hm_h, hm_w = input_h // 4, input_w // 4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out_layer1 = self.layer1(x)
        out_layer2 = self.layer2(out_layer1)

        out_layer3 = self.layer3(out_layer2)

        out_layer4 = self.layer4(out_layer3)

        # up_level1: torch.Size([b, 512, 14, 14])
        up_level1 = F.interpolate(out_layer4, scale_factor=2, mode='bilinear', align_corners=True)

        concat_level1 = torch.cat((up_level1, out_layer3), dim=1)
        # up_level2: torch.Size([b, 256, 28, 28])
        up_level2 = F.interpolate(self.conv_up_level1(concat_level1), scale_factor=2, mode='bilinear',
                                  align_corners=True)

        concat_level2 = torch.cat((up_level2, out_layer2), dim=1)
        # up_level3: torch.Size([b, 128, 56, 56]),
        up_level3 = F.interpolate(self.conv_up_level2(concat_level2), scale_factor=2, mode='bilinear',
                                  align_corners=True)
        # up_level4: torch.Size([b, 64, 56, 56])
        up_level4 = self.conv_up_level3(torch.cat((up_level3, out_layer1), dim=1))

        ret = {}
        for head in self.heads:
            temp_outs = []
            for fpn_idx, fdn_input in enumerate([up_level2, up_level3, up_level4]):
                fpn_out = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))(fdn_input)
                _, _, fpn_out_h, fpn_out_w = fpn_out.size()
                # Make sure the added features having same size of heatmap output
                if (fpn_out_w != hm_w) or (fpn_out_h != hm_h):
                    fpn_out = F.interpolate(fpn_out, size=(hm_h, hm_w))
                temp_outs.append(fpn_out)
            # Take the softmax in the keypoint feature pyramid network
            final_out = self.apply_kfpn(temp_outs)

            ret[head] = final_out

        return ret

    def apply_kfpn(self, outs):
        outs = torch.cat([out.unsqueeze(-1) for out in outs], dim=-1)
        softmax_outs = F.softmax(outs, dim=-1)
        ret_outs = (outs * softmax_outs).sum(dim=-1)
        return ret_outs
    
    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # Load pretrained weights for layers starting from the second layer
            for fpn_idx in [1, 2]:  # Start from the second FPN layer
                for head in self.heads:
                    final_layer = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))
                    for i, m in enumerate(final_layer.modules()):
                        if isinstance(m, nn.Conv2d):
                            if m.weight.shape[0] == self.heads[head]:
                                if 'hm' in head:
                                    nn.init.constant_(m.bias, -2.19)
                                else:
                                    nn.init.normal_(m.weight, std=0.001)
                                    nn.init.constant_(m.bias, 0)

            # Load pretrained weights for the rest of the model
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))

            # Filter out the weights for the first layer from the pretrained state_dict
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('conv1.')}

            # Update the model's state_dict with the pretrained weights
            model_dict = self.state_dict()
            model_dict.update(pretrained_state_dict)
            self.load_state_dict(model_dict, strict=False)




resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv, imagenet_pretrained):
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
    model.init_weights(num_layers, pretrained=imagenet_pretrained)
    return model