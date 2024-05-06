import os
import sys
import json
import pickle
import random
import math

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.nn as nn


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        train_folder = [os.path.join(root, cla, i) for i in os.listdir(cla_path)]
        for folder in train_folder:
            train_images = [os.path.join(folder, j) for j in os.listdir(folder)
                            if os.path.splitext(j)[-1] in supported]
            # 排序，保证各平台顺序一致
            train_images.sort()
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量
            every_class_num.append(len(train_images))
            # # 按比例随机采样验证样本
            # for img_path in train_images:
            #     train_images_path.append(img_path)
            #     train_images_label.append(image_class)
            # 按比例随机采样验证样本
            train_images_path.append(train_images)
            train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label



def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def mse_labels(labels):
    mselabels = torch.zeros(labels.size(0), 2)
    num = 0
    for label in labels:
        mselabels[num][label] = 1
        num += 1
    return mselabels


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels, device="cuda:0"):
        num = torch.tensor([0.]).long().to(device)
        p = torch.zeros(pred.size(0)).to(device)
        all_p = torch.softmax(pred, dim=1).to(device)
        max_values, _ = torch.max(all_p, dim=1)  # 沿着维度1（每行）计算最大值
        sorted_values, _ = torch.sort(all_p, dim=1, descending=True) # 沿着维度1（每行）进行降序排序
        second_max_values = sorted_values[:, 1]  # 取每行的第二个元素（第二大值）
        diff_values = max_values - second_max_values
        # 找到所有差值中的最小值
        min_diff = torch.min(diff_values)
        y = (2*torch.ones(pred.size(0))).to(device)
        for i in labels:
            right_p = all_p[num[0]][i]
            p[num[0]] = right_p
            num[0] += 1
        loss = -torch.mean((torch.pow((torch.ones(pred.size(0)).to(device)-p), y))*torch.log(p))
        return loss


class Myloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels, device="cuda:0"):
        num = torch.tensor([0.]).long().to(device)
        p = torch.zeros(pred.size(0)).to(device)
        all_p = torch.softmax(pred, dim=1).to(device)
        max_values, _ = torch.max(all_p, dim=1) # 沿着维度1（每行）计算最大值
        sorted_values, _ = torch.sort(all_p, dim=1, descending=True) # 沿着维度1（每行）进行降序排序
        second_max_values = sorted_values[:, 1] # 取每行的第二个元素（第二大值）
        diff_values = max_values - second_max_values
        # 找到所有差值中的最小值
        min_diff = torch.min(diff_values)
        y = 3 * diff_values
        y = y.to(device)
        for i in labels:
            right_p = all_p[num[0]][i]
            p[num[0]] = right_p
            num[0] += 1
        for n, j in enumerate(y):
            if j <= 0.01:
                y[n] = 0.
        loss = -torch.mean((torch.pow((torch.ones(pred.size(0)).to(device)-p), y))*torch.log(p))
        return loss

# def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
#     model.train()
#     loss_function = torch.nn.CrossEntropyLoss()
#     # loss_function = Myloss()
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     optimizer.zero_grad()
#
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     for step, data in enumerate(data_loader):
#         images, labels, _ = data
#         sample_num += images.shape[0]
#
#         pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         loss = loss_function(pred, labels.to(device))
#         loss.backward()
#         accu_loss += loss.detach()
#
#         data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num,
#             optimizer.param_groups[0]["lr"]
#         )
#
#         if not torch.isfinite(loss):
#             print('WARNING: non-finite loss, ending training ', loss)
#             sys.exit(1)
#
#         optimizer.step()
#         optimizer.zero_grad()
#         # update lr
#         lr_scheduler.step()
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def cacu_auc(label, prob):
    '''
    :param label: 样本的真实标签
    :param prob: 分类模型的预测概率值，表示该样本为正类的概率
    :return: 分类结果的AUC
    '''
    # 将label 和 prob组合，这样使用一个key排序时另一个也会跟着移动
    temp = list(zip(label, prob))
    # 将temp根据prob的概率大小进行升序排序
    rank = [val1 for val1, val2 in sorted(temp, key=lambda x: x[1])]
    # 将排序后的正样本的rank值记录下来
    rank_list = [i+1 for i in range(len(rank)) if rank[i]==1]
    # 计算正样本个数m
    M = sum(label)
    # 计算负样本个数N
    N=len(label)-M
    return (sum(rank_list)-M*(M+1)/2)/(M*N)


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    # loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = Myloss()
    # loss_function = FocalLoss()
    loss_function = torch.nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    prod_list = []
    label_list = []

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        prod = torch.softmax(pred, dim=1)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        prod_list.append(prod[0][1])
        label_list.append(labels[0])
        loss = loss_function(pred, mse_labels(labels).to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()
    cacu_auc(label_list, prod_list)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = Myloss()
    loss_function = torch.nn.MSELoss()
    # loss_function = FocalLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    prod_list = []
    label_list = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        prod = torch.softmax(pred, dim=1)
        pred_classes = torch.max(pred, dim=1)[1]
        prod_list.append(prod[0][1])
        label_list.append(labels[0])
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        if pred_classes[0] == 0:
            TP += torch.eq(pred_classes, labels.to(device)).sum()
            FP += images.shape[0] - torch.eq(pred_classes, labels.to(device)).sum()
        else:
            TN += torch.eq(pred_classes, labels.to(device)).sum()
            FN += images.shape[0] - torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, mse_labels(labels).to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
    auc = cacu_auc(label_list, prod_list)
    recall = TP / (TP + FN + 0.001*torch.ones(1).to(device))
    specificity = TN / (TN + FP + 0.001*torch.ones(1).to(device))
    precision = TP / (TP + FP + 0.001*torch.ones(1).to(device))
    f1_score = 2 * (precision * recall) / (precision + recall + 0.001*torch.ones(1).to(device))

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc.item(), recall.item(), specificity.item(), precision.item(), f1_score.item()
# def evaluate(model, data_loader, device, epoch):
#     loss_function = torch.nn.CrossEntropyLoss()
#     # loss_function = Myloss()
#
#     model.eval()
#
#     accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
#     accu_loss = torch.zeros(1).to(device)  # 累计损失
#
#     sample_num = 0
#     data_loader = tqdm(data_loader, file=sys.stdout)
#     last_name = 0
#     patch_num = 0
#     patch_acc = 0
#     wsi_num = 0
#     wsi_acc = 0
#     for step, data in enumerate(data_loader):
#         images, labels, path = data
#         sample_num += images.shape[0]
#
#         pred = model(images.to(device))
#         pred_classes = torch.max(pred, dim=1)[1]
#         accu_num += torch.eq(pred_classes, labels.to(device)).sum()
#
#         for i in range(images.shape[0]):
#             name = path[i].split("\\")[-2]
#             if name == last_name:
#                 patch_num += 1
#                 if torch.eq(pred_classes[i], labels[i].to(device)):
#                     patch_acc += 1
#             else:
#                 wsi_num += 1
#                 if patch_acc*2 > patch_num:
#                     wsi_acc += 1
#                 last_name = name
#                 patch_num = 0
#                 patch_acc = 0
#
#
#         loss = loss_function(pred, labels.to(device))
#         accu_loss += loss
#
#         data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, wsi_acc: {:.3f}".format(
#             epoch,
#             accu_loss.item() / (step + 1),
#             accu_num.item() / sample_num,
#             wsi_acc / wsi_num
#         )
#
#     return accu_loss.item() / (step + 1), accu_num.item() / sample_num, wsi_acc / wsi_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
