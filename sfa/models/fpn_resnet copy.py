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
# 여기에 ResNet 모델 이름(예: 'resnet18', 'resnet34')과 해당 모델의 사전 훈련된 가중치가 호스팅되는 URL을 매핑합니다.


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
        # inplanes와 planes는 입력 및 출력 채널의 수를 나타내며, stride는 합성곱 연산의 보폭을 나타냅니다.
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # bn1은 Batch Normalization 레이어로, 합성곱 레이어의 출력을 정규화하여 모델의 학습을 안정화시키고 속도를 향상시킵니다.
        # planes는 정규화할 채널의 수를 나타냅니다.
        self.relu = nn.ReLU(inplace=True)
        # ReLU(Rectified Linear Unit) 활성화 함수를 정의합니다.
        # 이 함수는 비선형성을 도입하여 모델이 비선형 관계를 학습할 수 있도록 합니다.
        self.conv2 = conv3x3(planes, planes)
        # conv2는 다시 3x3 합성곱 레이어를 정의하며, 이전 레이어의 출력을 입력으로 받습니다. 이 레이어는 더 많은 특징을 추출합니다.
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 두 번째 Batch Normalization 레이어로, 두 번째 합성곱 레이어의 출력을 정규화합니다.
        self.downsample = downsample
        # downsample은 다운샘플링을 수행하는 함수나 레이어를 가리킵니다. 이것은 단순히 입력 데이터의 크기를 조절하기 위한 용도로 사용됩니다.
        self.stride = stride
        # stride는 합성곱 연산 중에 사용되는 보폭을 나타냅니다. 이것은 입력 데이터를 합성곱 레이어에서 어떻게 이동할지 결정합니다.

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
        # inplanes 변수를 초기화하며, 이것은 합성곱 레이어의 입력 채널 수를 나타냅니다.
        self.deconv_with_bias = False
        # 역합성곱 레이어를 사용할 때 편향(bias)을 사용할 것인지 여부를 나타내는 변수입니다. 여기서는 사용하지 않음을 나타냅니다.
        self.heads = heads
        # 포즈 추정에 사용되는 머리(head) 수를 나타내는 변수입니다.

        super(PoseResNet, self).__init__()
        # nn.Module의 초기화 메서드를 호출하여 모델을 초기화합니다.
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 2D 합성곱 레이어를 정의하며, 이 레이어는 입력 채널이 6개이고 출력 채널이 64개인데, 입력 이미지에 필터(kernel) 크기가 7x7로 합성곱을 수행합니다.
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # Batch Normalization 레이어를 정의하며, 모델의 학습을 안정화시키고 속도를 향상시킵니다.
        self.relu = nn.ReLU(inplace=True)
        # ReLU 활성화 함수를 정의합니다.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 최대 풀링(Max Pooling) 레이어를 정의하며, 이미지를 다운샘플링하여 공간 해상도를 줄입니다.
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 첫 번째 레이어 그룹을 만듭니다. 이 그룹은 _make_layer 메서드를 사용하여 여러 개의 block을 포함하고, 각각의 block은 64개의 출력 채널을 가집니다.
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 두 번째 레이어 그룹을 만듭니다. 이 그룹은 첫 번째 레이어 그룹과 비슷하지만 보다 깊은 특징을 추출합니다.
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 세 번째 레이어 그룹을 만듭니다. 역시 더 깊은 특징을 추출하는 역할을 합니다.
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 네 번째 레이어 그룹을 만듭니다. 이 그룹은 더 깊은 특징을 추출하며, 네트워크의 가장 깊은 부분입니다.

        self.conv_up_level1 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        # 첫 번째 역합성곱 레이어를 정의합니다. 이 레이어는 다양한 레이어의 특징 맵을 결합합니다.
        self.conv_up_level2 = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
        # 두 번째 역합성곱 레이어를 정의합니다.
        self.conv_up_level3 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        # 세 번째 역합성곱 레이어를 정의합니다.

        fpn_channels = [256, 128, 64]
        # Feature Pyramid Network (FPN)에서 사용되는 특징 맵 채널의 크기를 나타내는 리스트를 생성합니다.
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
        # _make_layer 메서드는 ResNet 블록들을 생성하는 역할을 합니다. 이 메서드는 ResNet 블록의 반복되는 구조를 정의하며, block은 기본 블록 유형을 나타냅니다.
        # block: 이 매개변수는 ResNet 블록의 유형을 나타냅니다. ResNet은 기본 블록을 반복하여 쌓아올립니다.
        # planes: 이 매개변수는 현재 레이어의 출력 채널 수를 나타냅니다. 새로운 레이어의 출력 채널 수가 됩니다.
        # blocks: 이 매개변수는 현재 레이어에 추가할 ResNet 블록의 수를 나타냅니다.
        # stride: 이 매개변수는 합성곱 레이어의 스트라이드 값을 나타냅니다. 스트라이드가 1이 아닌 경우, 다운샘플링이 수행됩니다.
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
        # forward 메서드는 모델의 데이터 흐름을 정의합니다. 주어진 입력 x를 받아서 출력을 계산하는 역할을 합니다.
        _, _, input_h, input_w = x.size()
        # 입력 이미지의 크기를 input_h와 input_w 변수에 저장합니다.
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
            # self.heads에 정의된 머리(head)들을 하나씩 반복합니다. 각 머리는 모델이 다양한 종류의 출력을 생성하는 데 사용됩니다.
            temp_outs = []
            for fpn_idx, fdn_input in enumerate([up_level2, up_level3, up_level4]):
                fpn_out = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))(fdn_input)
                # 각 특징 맵을 현재 머리와 함께 특징 피라미드 네트워크(FPN)에 적용합니다. 
                # self.__getattr__를 사용하여 각 FPN 레이어에 액세스하고, fdn_input를 해당 레이어에 전달하여 새로운 특징을 생성합니다.
                _, _, fpn_out_h, fpn_out_w = fpn_out.size()
                # FPN 출력의 높이와 너비를 fpn_out_h와 fpn_out_w 변수에 저장합니다.
                # Make sure the added features having same size of heatmap output
                if (fpn_out_w != hm_w) or (fpn_out_h != hm_h):
                    fpn_out = F.interpolate(fpn_out, size=(hm_h, hm_w))
                # PN 출력의 크기가 원하는 출력 크기인 hm_w와 hm_h와 다른지 확인합니다. 
                # 만약 다르다면, F.interpolate 함수를 사용하여 FPN 출력을 원하는 크기로 업샘플링합니다.
                temp_outs.append(fpn_out)
            # Take the softmax in the keypoint feature pyramid network
            final_out = self.apply_kfpn(temp_outs)
            # 모든 레벨의 특징을 종합하여 최종 출력을 생성합니다. 이 과정은 apply_kfpn 메서드를 통해 이루어집니다.

            ret[head] = final_out
            # 최종 출력을 딕셔너리 ret에 저장합니다. 
            # 각 머리에 대한 결과는 해당 머리를 키로 하고, 최종 출력을 값으로 하는 딕셔너리로 구성됩니다.

        return ret

    def apply_kfpn(self, outs): # apply_kfpn 메서드는 입력으로 받은 특징 맵들(outs)에 대해 특정 처리를 수행한 후, 결합된 출력을 반환합니다.
        outs = torch.cat([out.unsqueeze(-1) for out in outs], dim=-1)
        # 스트에 있는 각 특징 맵을 torch.cat 함수를 사용하여 마지막 차원(즉, 차원 -1)을 따라 연결합니다. 
        # 이렇게 함으로써, 여러 레벨의 특징을 하나의 텐서로 합칩니다.
        softmax_outs = F.softmax(outs, dim=-1)
        # F.softmax 함수를 사용하여 이러한 특징을 소프트맥스 함수를 통해 정규화합니다. 
        # 이것은 다른 특징 간의 중요도를 고려하고 가중치를 부여하는 역할을 합니다.
        ret_outs = (outs * softmax_outs).sum(dim=-1)
        # 마지막으로, 가중합을 계산하여 최종 출력을 반환합니다. 이렇게 함으로써, 다양한 레벨의 특징을 결합하고 가중 평균을 계산합니다.
        return ret_outs

    def init_weights(self, num_layers, pretrained=True): # 메서드는 모델의 가중치를 초기화하고, 사전 훈련된(pretrained) 가중치를 로드하는 역할을 합니다.
        # 델을 초기화할 때 pretrained=True로 설정하면 사전 훈련된 가중치를 로드하고,
        # pretrained=False로 설정하면 사전 훈련된 가중치를 로드하지 않고 모델을 처음부터 훈련하게 됩니다.
        if pretrained:
            # TODO: Check initial weights for head later
            self.conv1.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet{}'.format(num_layers)]))
            for fpn_idx in [0, 1, 2]:  # 3 FPN layers
            # 루프 내에서, 세 개의 특징 피라미드 네트워크(FPN) 레이어를 반복하며, 
            # 각 FPN 레이어와 self.heads 내의 각 머리(head)에 대해 해당 FPN 레이어 내의 합성곱 레이어의 가중치를 초기화합니다.
                for head in self.heads:
                    final_layer = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))
                    for i, m in enumerate(final_layer.modules()):
                        if isinstance(m, nn.Conv2d):
                            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                            # print('=> init {}.bias as 0'.format(name))
                            if m.weight.shape[0] == self.heads[head]:
                                if 'hm' in head:
                                    nn.init.constant_(m.bias, -2.19)
                                else:
                                    nn.init.normal_(m.weight, std=0.001)
                                    nn.init.constant_(m.bias, 0)
            # pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            # 로드된 사전 훈련된 가중치는 self.load_state_dict(pretrained_state_dict, strict=False)를 사용하여 모델에 할당됩니다.
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv, imagenet_pretrained):
    # num_layers: ResNet의 레이어 수를 나타내며, 모델의 깊이를 결정합니다.
    # heads: 모델의 머리(head)에 대한 구성을 나타내는 딕셔너리입니다. 이 머리들은 모델의 출력을 결정하며, 여러 작업에 사용됩니다.
    # head_conv: 머리의 합성곱 레이어의 출력 채널 수를 나타냅니다.
    # imagenet_pretrained: 사전 훈련된 ResNet 가중치를 사용할지 여부를 나타냅니다.
    block_class, layers = resnet_spec[num_layers]
    # resnet_spec은 ResNet 아키텍처의 레이어 구성 정보를 포함하는 딕셔너리입니다. 
    # num_layers를 사용하여 해당 레이어 수에 대한 블록 클래스 및 레이어 수를 가져옵니다.
    model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
    # PoseResNet 클래스의 객체를 생성합니다. 이 클래스는 PoseNet 모델을 정의하고 구성하는 데 사용됩니다.
    # 매개변수로는 블록 클래스, 레이어 수, 머리(heads) 구성 및 머리의 합성곱 레이어 출력 채널 수가 전달됩니다.
    model.init_weights(num_layers, pretrained=imagenet_pretrained)
    # 이전에 정의한 PoseResNet 모델의 가중치를 초기화합니다. 이때 init_weights 메서드가 호출됩니다.
    # num_layers와 imagenet_pretrained 매개변수를 사용하여 가중치 초기화 및 사전 훈련된 가중치의 로드 여부를 결정합니다.
    return model
