import sys

import torch
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


def train_one_epoch(model, scheduler, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    all_pred_probs = []
    all_labels = []
    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device).squeeze(dim=1).long()).sum()
        loss = loss_function(pred, labels.to(device).squeeze(dim=1).long())
        loss.backward()
        accu_loss += loss.detach()

        pred_probs = torch.softmax(pred, dim=1).cpu().detach().numpy()
        all_pred_probs.extend(pred_probs)
        all_labels.extend(labels.cpu().numpy())
        num_classes = pred_probs.shape[1]
        all_labels_bin = label_binarize(all_labels, classes=range(num_classes))
        # 计算One-vs-Rest策略下的多分类AUC值
        auc_ovr = roc_auc_score(all_labels_bin, all_pred_probs, multi_class='ovr')
        print(f"One-vs-Rest AUC: {auc_ovr}")

        data_loader.desc = "[d_train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                 accu_loss.item() / (step + 1),
                                                                                 accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device).squeeze(dim=1).long()).sum()

        loss = loss_function(pred, labels.to(device).squeeze(dim=1).long())

        accu_loss += loss

    print("=> val_loss: {:.4f}   val_acc: {:.4f}".
          format(accu_loss.item() / (step + 1), accu_num.item() / sample_num))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
