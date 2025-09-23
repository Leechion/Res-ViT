from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import transforms
from ViTrans import ResNet_ViT
from ViTrans import ResNet
from ViTrans import ViT
from train import SerialFusionModel as models
from medmnist import RetinaMNIST, PathMNIST, PneumoniaMNIST
from model import swin_tiny_patch4_window7_224 as create_model
import torch

img_size = 28
data_transform = {
    "d_train": transforms.Compose(
        [
            transforms.RandomRotation(degrees=15),  # 随机旋转函数-依degrees随机旋转一定角度
            transforms.ColorJitter(),  # 改变颜色的，随机从-0.5 0.5之间对颜色变化
            transforms.RandomResizedCrop(64),  # 随机长宽比裁剪
            transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
            transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
                                 std=[0.229, 0.224, 0.225])]),
    "d_val": transforms.Compose(
        [
            transforms.RandomRotation(degrees=15),  # 随机旋转函数-依degrees随机旋转一定角度
            transforms.ColorJitter(),  # 改变颜色的，随机从-0.5 0.5之间对颜色变化
            transforms.RandomResizedCrop(64),  # 随机长宽比裁剪
            transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
            transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
                                 std=[0.229, 0.224, 0.225])]),
    "d_test": transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),  # 改变颜色的，随机从-0.5 0.5之间对颜色变化
            transforms.RandomResizedCrop(64),  # 随机长宽比裁剪
            transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
            transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
            # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
                                 std=[0.229, 0.224, 0.225])]),
}
device = torch.device("cuda")
model = ResNet_ViT(num_classes=5).to(device)
#model = ViT(image_size=64, patch_size=4, num_classes=5, channels=3,  dim=64, depth=6, heads=8, mlp_dim=128).to(device)
#model = ResNet(num_classes=2).to(device)  # 导入网络结构
model.load_state_dict(torch.load('./weights/model-ResVIT-ret.pth'))  # 导入网络的参数
# model.load_state_dict(torch.load('./weights/model-best_tiny_ret.pth'))  # 导入网络的参数

test_set = RetinaMNIST(root="D:\pythonProject\MachineLearning\dataset", transform=data_transform["d_test"],
                          split="test")

test_data = DataLoader(dataset=test_set, batch_size=32, drop_last=True)

print(len(test_set))


@torch.no_grad()
def evaluate(data_loader, model):
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device).squeeze(dim=1).long()).sum()
        print(accu_num)
        print(sample_num)
    print("=>   test_acc:", accu_num.item() / sample_num)


    return accu_num.item() / sample_num


# def evaluate(test_data, model):
#     n_correct = 0
#     n_total = 0
#     with torch.no_grad():
#         for data in test_data:
#             imgs, targets = data
#             imgs = imgs.to(device)
#             targets = targets.to(device)
#             outputs = model(imgs)
#             for i, output in enumerate(outputs):
#                 if torch.argmax(output) == targets[i]:  # 输出的最大概率的数字与当前数字标签对比
#                     n_correct += 1
#                 n_total += 1
#     return n_correct / n_total
#

print(evaluate(test_data, model))
