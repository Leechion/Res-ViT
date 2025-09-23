import torch
import torch.nn as nn
import torchvision.models as models
from model import swin_tiny_patch4_window7_224  # 这里以Swin-Tiny为例
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from model import swin_tiny_patch4_window7_224 as create_model

model_weight_path_1 = "./weights/resnet50-0676ba61.pth"
model_weight_path_2 = "./weights/swin_tiny_patch4_window7_224.pth"

import torch
import torch.nn as nn
import torchvision.models as models
import timm


class HybridModuleFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModuleFusionModel, self).__init__()
        # 加载ResNet50模型
        self.resnet50 = models.resnet50(pretrained=False).to(device)
        self.resnet50.load_state_dict(torch.load(model_weight_path_1, map_location=device))
        # 去掉ResNet50的最后一层全连接层
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        # 加载Swin Transformer模型
        self.swin_transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        # 混合模块
        self.hybrid_module = HybridModule()
        # 融合后的特征维度
        self.fusion_dim = 2048
        # 融合后的全连接层
        self.fc = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, x):
        # ResNet50前向传播
        resnet_features = self.resnet50(x)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        # Swin Transformer前向传播
        swin_features = self.swin_transformer(x)
        swin_features = swin_features.view(swin_features.size(0), -1)
        # 混合模块融合特征
        fused_features = self.hybrid_module(resnet_features, swin_features)
        # 全连接层输出
        output = self.fc(fused_features)

        return output


class HybridModule(nn.Module):
    def __init__(self):
        super(HybridModule, self).__init__()
        # 卷积层
        self.conv = nn.Conv2d(2048 + 768, 2048, kernel_size=1)
        # 批量归一化层
        self.bn = nn.BatchNorm2d(2048)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, resnet_features, swin_features):
        # 特征拼接
        features = torch.cat([resnet_features, swin_features], dim=1)
        # 卷积操作
        features = self.conv(features)
        # 批量归一化
        features = self.bn(features)
        # 激活函数
        features = self.relu(features)

        return features


device = torch.device("cuda")


def get_data_loader(is_trian):  # 数据加载器
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = torchvision.datasets.CIFAR10("D:\pythonProject\MachineLearning\dataset", is_trian, transform=preprocess,
                                            download=False)
    return DataLoader(data_set, batch_size=64, shuffle=True, drop_last=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for data in test_data:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net.forward(imgs)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == targets[i]:  # 输出的最大概率的数字与当前数字标签对比
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    global cross_loss
    train_data = get_data_loader(is_trian=True)
    test_data = get_data_loader(is_trian=True)
    net = HybridModuleFusionModel(num_classes=10)
    net = net.to(device)

    print("acccuracy:", evaluate(test_data, net))
    ##########################################################
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 学习率
    loss = CrossEntropyLoss()  # 损失 计算  对数损失函数
    for eopch in range(10):
        for data in train_data:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            net.zero_grad()  # 初始化
            outputs = net.forward(imgs)  # 正向传播
            loss = loss.to(device)
            cross_loss = loss(outputs, targets)
            cross_loss.backward()  # 反向误差传播
            optimizer.step()  # 优化网络参数
        print("epoch", eopch, "acccuracy:", evaluate(test_data, net), "loss", cross_loss)


if __name__ == "__main__":
    main()
