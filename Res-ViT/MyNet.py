from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import *
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import *
from medmnist import PathMNIST

device = torch.device("cuda")


class Net(torch.nn.Module):  # 神经网络主体包含四个全连接层
    def __init__(self):
        super(Net, self).__init__()
        self.module = Sequential(
            Conv2d(3, 28, 5, padding=2),
            GELU(),
            MaxPool2d(2),
            Conv2d(28, 28, 5, padding=2),
            GELU(),
            MaxPool2d(2),
            Conv2d(28, 64, 5, padding=2),
            GELU()
            # MaxPool2d(2),
            # Flatten(),
            # Linear(576, 64),
            # Linear(64, 9)
        )

    def forward(self, input):
        x = (self.module(input))
        return x


def get_data_loader(is_trian):  # 数据加载器
    to_tensor = transforms.Compose([transforms.ToTensor()])
    #data_set = torchvision.datasets.CIFAR10("../dataset", is_trian, transform=to_tensor, download=False)
    #data_set_2 = PathMNIST("../dataset", is_trian, download=False)
    train_data_set = PathMNIST(split="train", root="D:\pythonProject\Machine Learning\dataset", transform=to_tensor,
                               download=False)
    val_data_set = PathMNIST(split="val", root="D:\pythonProject\Machine Learning\dataset", transform=to_tensor,
                             download=False)

    # train_data, val_data = Data.random_split(data_set_2, [round(0.8 * len(data_set_2)), round(0.2 * len(data_set_2))])
    train_data_loader = DataLoader(train_data_set, batch_size=64, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(val_data_set, batch_size=64, shuffle=True, drop_last=True)
    return train_data_loader, val_data_loader


# 测试
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
    train_data, val_data = get_data_loader(is_trian=True)
    net = Net()
    net = net.to(device)

    writer = SummaryWriter("../../logs")

    # 训练
    print("accuracy:", evaluate(val_data, net))
    ##########################################################
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # 学习率
    loss = CrossEntropyLoss()  # 损失 计算  对数损失函数
    step = 0
    acc = 0
    for eopch in range(20):
        for data in train_data:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net.forward(imgs)  # 正向传播
            loss = loss.to(device)
            cross_loss = loss(outputs, targets.squeeze(dim=1).long())
            net.zero_grad()  # 初始化
            cross_loss.backward()  # 反向误差传播
            optimizer.step()  # 优化网络参数
            writer.add_scalar("loss", cross_loss.item(), step)
            step = step + 1
        if evaluate(val_data, net) > acc:
            acc = evaluate(val_data, net)
            torch.save(net.state_dict(), "../../Net/MyNet.pth")
        print("epoch", eopch + 1, "accuracy:", evaluate(val_data, net), "loss", cross_loss.item())

    writer.close()


if __name__ == "__main__":
    main()
