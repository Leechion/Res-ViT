
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.utils.data as Data
from medmnist import PathMNIST
from medmnist import PneumoniaMNIST
from medmnist import RetinaMNIST


from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 数据集根目录的路径
        transform: 数据预处理操作（如归一化、裁剪等，可为None）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # 假设按类别划分文件夹的方式组织数据集，遍历文件夹获取图像路径和标签
        classes = os.listdir(root_dir)
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(root_dir, class_name)
            for image_file in os.listdir(class_path):
                self.image_files.append(os.path.join(class_path, image_file))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


img_size = 28
data_transform = {
    "d_train": transforms.Compose(
        [
         transforms.Grayscale(num_output_channels=3),
         transforms.RandomRotation(degrees=15),  # 随机旋转函数-依degrees随机旋转一定角度
         transforms.ColorJitter(),  # 改变颜色的，随机从-0.5 0.5之间对颜色变化
         transforms.RandomResizedCrop(64),  # 随机长宽比裁剪
         transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
         transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
         # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
         transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
                              std=[0.229, 0.224, 0.225])]),
    "d_val": transforms.Compose(
        [
         transforms.Grayscale(num_output_channels=3),
         transforms.RandomRotation(degrees=15),  # 随机旋转函数-依degrees随机旋转一定角度
         transforms.ColorJitter(),  # 改变颜色的，随机从-0.5 0.5之间对颜色变化
         transforms.RandomResizedCrop(64),  # 随机长宽比裁剪
         transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
         transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
         # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
         transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
                              std=[0.229, 0.224, 0.225])])}


def get_data_loader(data_dir, batch_size, num_workers, aug=True):
    # data_set = torchvision.datasets.(data_dir, True, transform=data_transform["d_train" if aug else "d_val"], download=False)
    # data_set = torchvision.datasets.FashionMNIST(data_dir, True, transform=data_transform["d_train" if aug else "d_val"], download=False)
    train_data_set = PathMNIST(split="train", root=data_dir, transform=data_transform["d_train"], download=True)
    val_data_set = PathMNIST(split="val", root=data_dir, transform=data_transform["d_val"], download=True)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=aug, num_workers=num_workers)
    val_loader = DataLoader(val_data_set, batch_size=batch_size, shuffle=aug, num_workers=num_workers)
    # dataset = datasets.ImageFolder(root=data_dir, transform=data_transform["d_train" if aug else "d_val"])
    # data_loader = DataLoader(dataset,
    #                          batch_size=batch_size, shuffle=aug,
    #                          num_workers=num_workers)

    # print()

    return train_loader, val_loader



