import os
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from sklearn.metrics import roc_auc_score
import numpy as np
from ViTrans import ResNet_ViT
from ViTrans import ResNet
from ViTrans import ViT
from model import swin_tiny_patch4_window7_224 as create_model
from dataset import get_data_loader
from utils import train_one_epoch, evaluate
from MyNet import Net as model

model_weight_path_1 = "./weights/resnet50-0676ba61.pth"
device = torch.device("cuda")

import torch
import torch.nn as nn


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
        resnet_features = self.cv(resnet_features)
        # 将ResNet50的特征输入到Swin Transformer中进一步提取特征
        output = self.swin_transformer(resnet_features)
        return output


#####################################################################################################################


def main(args):
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter(log_dir="runs")
    data_dir = 'D:\pythonProject\MachineLearning\dataset'
    train_data_loader, val_data_loader = get_data_loader(data_dir, args.batch_size, 4, aug=True)
    # data_loader = get_data_loader(data_dir, args.batch_size, 4, aug=True)

    #model = create_model(num_classes=args.num_classes).to(device)
    model = ResNet_ViT(num_classes=args.num_classes).to(device)
    # model = ResNet(num_classes=args.num_classes).to(device)
    #model = ViT(image_size=64, patch_size=4, num_classes=5, channels=3,dim=64, depth=6, heads=8, mlp_dim=128).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # weights_dict = torch.load(args.weights, map_location=device)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch / 10 + 1))
    max_acc = 0

    # for name, para in model.named_parameters():
    #     # 除head外，其他权重全部冻结
    #     if "head" not in name:
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))

    for epoch in range(args.epochs):
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        # d_train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_data_loader,
                                                device=device,
                                                epoch=epoch,
                                                scheduler=scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_data_loader,
                                     device=device)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        print(val_acc, ",", max_acc)
        if val_acc > max_acc:
            max_acc = val_acc
            print("save model")
            torch.save(model.state_dict(), "weights/m.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)

    #预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str,
    #                     default='./weights/swin_tiny_patch4_window7_224.pth',
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str,
                        default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)

    opt = parser.parse_args()

    main(opt)
