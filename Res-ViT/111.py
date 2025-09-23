import os
import argparse

import timm
import torch
import torch.optim as optim

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from model import swin_tiny_patch4_window7_224 as create_model
from ViTrans import ViT


import torch
import torch.nn as nn

import timm

import torch
import torch.nn as nn


model_weight_path_1 = "./weights/resnet18-f37072fd.pth"
device = torch.device("cuda")


class SerialFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(SerialFusionModel, self).__init__()
        # # 加载ResNet50模型
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.load_state_dict(torch.load(model_weight_path_1))
        # 去掉ResNet50的最后一层全连接层
        # 加载Swin Transformer模型
        self.swin_transformer = create_model()
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.cv = nn.Conv2d(2048, 3, 1)

    def forward(self, x):
        # 通过ResNet50提取特征
        resnet_features = self.resnet50(x)
        swin_features = self.swin_transformer(x)
        print(swin_features.shape)
        print(resnet_features.shape)
        resnet_features = self.cv(resnet_features)
        print(resnet_features.shape)
        # 将ResNet50的特征输入到Swin Transformer中进一步提取特征
        output = self.swin_transformer(resnet_features)
        return output


# 定义结合ViT和ResNet的模型类
class ResNet_ViT(nn.Module):
    def __init__(self, num_classes=10, vit_patch_size=16, vit_dim=768, vit_depth=12, vit_heads=12):
        super(ResNet_ViT, self).__init__()
        # 加载预训练的ResNet模型（这里以ResNet18为例，你可以换为其他ResNet版本）
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load(model_weight_path_1))
        # 去掉ResNet的最后一层全连接层，因为我们要获取其特征输出
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # 定义ViT模型
        self.vit = ViT(image_size=64, patch_size=16, num_classes=vit_dim, channels=3,
                       dim=64, depth=6, heads=8, mlp_dim=128)
        # 全连接层用于最终分类，将融合后的特征映射到类别数
        self.fc = nn.Linear(vit_dim + 512, num_classes)  # 512是ResNet18最后一层特征维度，可根据实际ResNet版本调整

    def forward(self, x):
        # 通过ResNet提取特征
        resnet_feats = self.resnet(x)
        resnet_feats = torch.flatten(resnet_feats, 1)

        # 通过ViT提取特征
        x = self.vit(x)
        print(x.shape)
        # 特征融合，可以采用拼接方式（也可尝试其他融合方式如加权相加等）
        combined_feats = torch.cat([resnet_feats, x], dim=1)
        print(combined_feats.shape)

        # 通过全连接层进行分类
        out = self.fc(combined_feats)

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # 加载预训练的ResNet模型（这里以ResNet18为例，你可以换为其他ResNet版本）
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load(model_weight_path_1))
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)  # 512是ResNet18最后一层特征维度，可根据实际ResNet版本调整
        self.fl = nn.Flatten()

    def forward(self, x):
        # 通过ResNet提取特征
        resnet_feats = self.resnet(x)
        resnet_feats= self.fl(resnet_feats)
        print(resnet_feats.shape)
        # 通过全连接层进行分类
        out = self.fc(resnet_feats)
        return out


# 示例用法，创建模型实例，假设有10个分类类别
model = ResNet_ViT(num_classes=10)
# 随机生成一个输入张量，模拟输入图像数据（这里示例为batch_size=2，图像尺寸3x224x224）
input_tensor = torch.randn(64, 3, 64, 64)
# 前向传播得到输出
output = model(input_tensor)
