import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets import get_model_from_name
from utils.callbacks import LossHistory
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import (get_classes, get_lr_scheduler, set_optimizer_lr, weights_init)
from utils.utils_fit import fit_one_epoch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if __name__ == "__main__":
    # ----------------------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # ----------------------------------------------------#
    # Cuda            = True
    Cuda = False
    classes_path = 'model_data/cls_classes.txt'

    input_shape = [224, 224]
    backbone = "resnet50"
    pretrained = False
    model_path = 'model_data/resnet50-19c8e357.pth'
    Init_Epoch = 0
    Freeze_Epoch = 0
    Freeze_batch_size = 0
    UnFreeze_Epoch = 120
    Unfreeze_batch_size = 64
    Freeze_Train = False
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 5e-4
    lr_decay_type = "cos"
    save_period = 1
    save_dir = 'logs'
    num_workers = 8
    # ------------------------------------------------------#
    #   train_annotation_path   训练图片路径和标签
    #   test_annotation_path    验证图片路径和标签（使用测试集代替验证集）
    # ------------------------------------------------------#
    train_annotation_path = "cls_train.txt"
    test_annotation_path = 'cls_test.txt'

    # ------------------------------------------------------#
    #   获取classes
    # ------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    if backbone != "vit":
        model = get_model_from_name[backbone](num_classes=num_classes, pretrained=pretrained)
    else:
        model = get_model_from_name[backbone](input_shape=input_shape, num_classes=num_classes, pretrained=pretrained)

    if not pretrained:
        weights_init(model)
    if model_path != "":
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    loss_history = LossHistory(save_dir, model, input_shape=input_shape)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train
    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path, "r", encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(test_annotation_path, "r", encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.seed(None)

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            model.freeze_backbone()
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == 'vit':
            nbs = 128
            lr_limit_max = 3e-4 if optimizer_type == 'adam' else 5e-2
            lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam': optim.Adam(model_train.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'sgd': optim.SGD(model_train.parameters(), Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        tra_dataset = DataGenerator(train_lines, input_shape, True)
        val_dataset = DataGenerator(val_lines, input_shape, False)
        gen = DataLoader(tra_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=detection_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=detection_collate)
        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == 'vit':
                    nbs = 128
                    lr_limit_max = 3e-4 if optimizer_type == 'adam' else 5e-2
                    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.Unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen = DataLoader(tra_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=detection_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=detection_collate)

                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          UnFreeze_Epoch, Cuda, save_period, save_dir)

        loss_history.writer.close()
