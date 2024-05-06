import os
import argparse
from PIL import Image
import json

import torch
import torch.optim as optim
from torchvision import transforms

from my_dataset import MyDataSet
from EFPN_model import PyramidMIL as create_model
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate

import csv
import datetime
file_time = datetime.datetime.now()
loss_list = [[], []]
csv_file_path = f"./loss_list/base_fpn_{file_time.strftime('%Y-%m-%d_%H-%M-%S')}"


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    train_images_path, train_images_label = read_split_data(args.train_data_path)
    val_images_path, val_images_label = read_split_data(args.val_data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     # transforms.RandomCrop(img_size),

                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   # transforms.Resize(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(args.num_classes).to(device)
    # model = MLP_Mixer(image_size=256,
    #                   patch_size=16,
    #                   dim=768,
    #                   num_classes=args.num_classes,
    #                   num_blocks=12,
    #                   token_dim=384,
    #                   channel_dim=3072).to(device)
    # VIT
    # model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    # netG = Generator(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=args.wd)
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)

    # VIT
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    best_auc = 0.
    best_recall = 0.
    best_specificity = 0.
    best_precision = 0.
    best_f1_score = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                # netG=netG,
                                                optimizer=optimizer,
                                                # optimizerG=optimizerG,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)
        # 释放显存
        torch.cuda.empty_cache()

        # validate
        val_loss, val_acc, val_auc, recall, specificity, precision, f1_score = evaluate(model=model,
                                                                                        data_loader=val_loader,
                                                                                        device=device,
                                                                                        epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # 释放显存
        torch.cuda.empty_cache()

        loss_list[0].append(train_acc)
        loss_list[1].append(val_acc)
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow((loss_list))

        if best_acc < val_acc:
            torch.save(model.state_dict(), "weights/best_model_public_ours_MSE.pth")
            best_acc = val_acc
            print("acc =", best_acc)
            best_auc = val_auc
            print("best_auc =", best_auc)
            best_recall = recall
            print("best_recall =", best_recall)
            best_specificity = specificity
            print("best_specificity =", best_specificity)
            best_precision = precision
            print("best_precision =", best_precision)
            best_f1_score = f1_score
            print("best_f1_score =", best_f1_score)

        # model.eval()
        # wsi_accu_num = 0
        # wsi_num = 0
        # folders = os.listdir(args.val_data_path)
        # for folder in folders:
        #     dirs = os.listdir(os.path.join(args.val_data_path, folder))
        #     for dir0 in dirs:
        #         wsi_num += 1
        #         true_num = 0
        #         num = 0
        #         image_dir = os.path.join(args.val_data_path, folder, dir0)
        #         files = os.listdir(image_dir)
        #         all_num = len(files)
        #         for file in files:
        #             num += 1
        #             img_path = os.path.join(image_dir, file)
        #             assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        #             img = Image.open(img_path)
        #             # [N, C, H, W]
        #             img = data_transform["val"](img)
        #             # expand batch dimension
        #             img = torch.unsqueeze(img, dim=0)
        #
        #             # read class_indict
        #             json_path = './class_indices.json'
        #             assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
        #
        #             with open(json_path, "r") as f:
        #                 class_indict = json.load(f)
        #
        #             with torch.no_grad():
        #                 # predict class
        #                 # output = torch.squeeze(model(img.to(device))).cpu()
        #                 output = model(img.to(device))
        #                 output = torch.squeeze(output).cpu()
        #                 predict = torch.softmax(output, dim=0)
        #                 predict_cla = torch.argmax(predict).numpy()
        #                 if class_indict[str(predict_cla)] == folder:
        #                     true_num += 1
        #         if true_num >= (all_num - true_num):
        #             wsi_accu_num += 1

        # if best_wsi_acc <= wsi_acc:
        #     best_wsi_acc = wsi_acc
        #     torch.save(model.state_dict(), "./weights/best_wsi_model_epoch{}.pth".format(epoch))
        #     print("best_wsi_acc =", best_wsi_acc)
        # else:
        #     os.remove("./weights/best_wsi_model_epoch{}.pth".format(epoch))

        # print(best_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=1e-4)

    # 数据集所在根目录

    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--train-data-path', type=str,
                        default="your train file path")
    parser.add_argument('--val-data-path', type=str,
                        default="your val file path")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
