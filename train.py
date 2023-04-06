import os
import time
import datetime

import torch
from torch import nn

from src import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from utils.my_dataset import VOCSegmentation
from utils import transforms as T

from torch.utils.tensorboard import SummaryWriter
from utils.mytools import calculater_1,Time_calculater
from src.unet_mod import *
from src.nets import *
def _create_folder(args):
    # 用来保存训练以及验证过程中信息
    if not os.path.exists("logs"):
        os.mkdir("logs")
    # 创建时间+模型名文件夹
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M-%S-")
    log_dir = os.path.join("logs", time_str + args.model_name )
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    results_file = os.path.join(log_dir, "results.txt")
    # self.results_file = log_dir + "/{}_results{}.txt".format(self.model_name, time_str)
    print("当前进行训练: {}".format(log_dir))
    return log_dir, results_file
def _load_dataset(args, batch_size):
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2007",
                                    transforms=get_transform(args,train=True),
                                    txt_name="train.txt")
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2007",
                                  transforms=get_transform(args,train=False),
                                  txt_name="val.txt")
    num_workers =0
    # num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)
    return train_loader, val_loader


def initialize_weights(model):
    for m in model.modules():
        # 判断是否属于Conv2d
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            # 判断是否有偏置
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.3)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)
    return model
# 数据集预处理
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [T.Resize(base_size)]
        # 以给定的概率随机水平翻转
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)
class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

def get_transform(args,train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = args.base_size
    crop_size = args.crop_size

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(base_size,mean=mean, std=std)

def main(args):
    #-----------------------初始化-----------------------
    log_dir, results_file=_create_folder(args)
    tb = SummaryWriter(log_dir=log_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1
    #-----------------------加载数据-----------------------
    train_loader, val_loader = _load_dataset(args, batch_size)
    #-----------------------创建模型-----------------------
    model = create_model(args,in_channels=3,num_classes=num_classes,base_c=args.base_c).to(device)
    #-----------------------创建优化器-----------------------
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight,ignore_index=255)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise ValueError("wrong optimizer name")
    #-----------------------创建学习率更新策略-----------------------
    scaler = torch.cuda.amp.GradScaler() if args.amp else None# 混合精度训练
    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    #-----------------------加载断点训练-----------------------
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    #-----------------------训练-----------------------
    best_dice = 0.
    start_time = time.time()
    time_calc = Time_calculater()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model,loss_fn, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        # -----------------------保存日志-----------------------
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_log="train_loss: {:.4f}, lr: {:.6f}".format(mean_loss, lr)
            val_log=confmat
            val_log["dice loss"]=format(1-dice, '.4f')
            print('--train_log:',train_log)
            print('--val_log:',val_log)
            f.write("Epoch: {}  \n".format(epoch))
            f.write("train_log: {}  \n".format(train_log))
            f.write("val_log: {}  \n".format(val_log))
            if epoch == args.epochs - 1:
                # f.write("args: {}  \n".format(args))
                model_size = calculater_1(model, (3, args.base_size, args.base_size), device)
                f.write("model_name: -{}-  \n".format(args.model_name))
                f.write("datasets: {}  \n".format(args.data_path))
                f.write('flops:{:.2f}  params:{:.2f}  \n'.format(model_size[0], model_size[1]))
            # -----------------------打印时间-----------------------
            time_calc.time_cal(epoch, args.epochs)
        # -----------------------保存tensorboard-----------------------
        tb.add_scalar("train/loss", mean_loss, epoch)
        tb.add_scalar("train/lr", lr, epoch)
        tb.add_scalar("val/dice loss", 1-dice, epoch)
        tb.add_scalar("val/miou", confmat["miou"], epoch)
        tb.add_scalar("val/acc", confmat["pa"], epoch)
        # 保存网路结构
        tb.add_graph(model, torch.rand(1, 3, args.base_size, args.base_size).to(device))
        # -----------------------保存模型-----------------------
        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue
        # 模型结构、优化器、学习率更新策略、epoch、参数
        if args.save_method == "all":
            checkpoints = model
        else:
            checkpoints = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     }
            # 混合精度训练
            if args.amp:
                checkpoints["scaler"] = scaler.state_dict()
        # 保存模型
        if args.save_best is True:
            torch.save(checkpoints, log_dir+"/best_model.pth")
        else:
            torch.save(checkpoints, log_dir+"/model_{}.pth".format(epoch))
    #-----------------------训练结束-----------------------
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


# 建立模型
def create_model(args,in_channels, num_classes, base_c=32):
    if args.model_name == "unet":
        model = UNet(in_channels=in_channels, num_classes=num_classes, base_c=base_c)
    elif args.model_name == "Unet0":
        model = Unet0(in_channels=in_channels, num_classes=num_classes, base_c=base_c)
    elif args.model_name == "Unet_c3g":
        model = Unet_c3g(in_channels, num_classes, base_c=base_c)
    # elif args.model_name == "Unet_c3":
    #     model = Unet_c3g(in_channels, num_classes, base_c=base_c,block='C3')
    elif args.model_name == "Unet_c2f":
        model = Unet_c3g(in_channels, num_classes, base_c=base_c,block='C2f')
    elif args.model_name == "Unet_res":
        model = Unet_EX(in_channels, num_classes, base_c=base_c,block_type='resnet')
    elif args.model_name == "Unet_resnest":
        model = Unet_EX(in_channels, num_classes, base_c=base_c, block_type='resnest')
    elif args.model_name == "Unet_res_cbam":
        model = Unet_Attention(in_channels, num_classes, base_c=base_c,block_type='resnest',attention='cbam')
    elif args.model_name == "Unet_res_se":
        model = Unet_Attention(in_channels, num_classes, base_c=base_c,block_type='resnest',attention='se')
    elif args.model_name == "Unet_res_ca":
        model = Unet_Attention(in_channels, num_classes, base_c=base_c,block_type='resnest',attention='ca')
    elif args.model_name == "Unet_res_simam":
        model = Unet_Attention(in_channels, num_classes, base_c=base_c,block_type='resnest',attention='simam')
    elif args.model_name == "Unet_mobile_s":
        model = Unet_lite(in_channels, num_classes, base_c=base_c,block_type='mobile_s')
    elif args.model_name == "Unet_shuffle":
        model = Unet_lite(in_channels, num_classes, base_c=base_c, block_type='shuffle')
    elif args.model_name == "Unet0_drop":
        model = Unet0_drop(in_channels, num_classes, base_c=base_c)
    elif args.model_name == "deeplabV3p":
        model = deeplabv3_resnet50(num_classes=num_classes, pretrained_backbone=False)
    elif args.model_name == "lraspp_mobilenetv3_large":
        model = lraspp_mobilenetv3_large(num_classes=num_classes, pretrain_backbone=False)
    elif args.model_name == "Unet_C3":
        model = Unet_best(in_channels, num_classes, base_c=base_c,block='C3')
    elif args.model_name == "Unet_C3_spp":
        model = Unet_best(in_channels, num_classes, base_c=base_c,block='C3',spp='spp')
    elif args.model_name == "Unet_C3_sppf":
        model = Unet_best(in_channels, num_classes, base_c=base_c,block='C3',spp='sppf')
    elif args.model_name == "Unet_C3_dw":
        model = Unet_best(in_channels, num_classes, base_c=base_c, block='C3', dw=True)
    elif args.model_name == "Unet_C3_cbam":
        model = Unet_best(in_channels, num_classes, base_c=base_c, block='C3', att='cbam')
    elif args.model_name == "Unet_C3_sam":
        model = Unet_best(in_channels, num_classes, base_c=base_c, block='C3', att='sam')
    else:
        raise ValueError("wrong model name")
    return initialize_weights(model)
def parse_args(model_name=None):
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--model_name", default=model_name, help="模型名称")
    parser.add_argument("--optimizer", default='adam',choices=['sgd','adam'] ,help="优化器")
    parser.add_argument("--base_size", default=512, type=int, help="图片缩放大小")
    parser.add_argument("--crop_size", default=512,  type=int, help="图片裁剪大小")
    parser.add_argument("--base_c", default=32, type=int, help="uent的基础通道数")
    parser.add_argument('--save_method',default='all' ,choices=['all','dict'],help='保存模型的方式')

    parser.add_argument("--data-path", default=r"..\VOCdevkit_cap_c5_bin", help="VOC数据集路径")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=10, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args

# tensorboard --logdir logs
# http://localhost:6006/
if __name__ == '__main__':
    args = parse_args('Unet_C3_cbam')
    main(args)
