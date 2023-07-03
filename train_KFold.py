import os
import random
import time
import datetime
import gc
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from utils.my_dataset import VOCSegmentation
from utils import transforms as T
import csv
from torch.utils.tensorboard import SummaryWriter
from utils.mytools import calculater_1
from src.unet_mod import *
from src.nets import *
from src.new_unet import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
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
# 时间统计
class Time_calculater(object):
    def __init__(self):
        self.start=time.time()
        self.last_time=self.start
        self.remain_time=0
    #定义将秒转换为时分秒格式的函数
    def time_change(self,time_init):
        time_list = []
        if time_init/3600 > 1:
            time_h = int(time_init/3600)
            time_m = int((time_init-time_h*3600) / 60)
            time_s = int(time_init - time_h * 3600 - time_m * 60)
            time_list.append(str(time_h))
            time_list.append('h ')
            time_list.append(str(time_m))
            time_list.append('m ')

        elif time_init/60 > 1:
            time_m = int(time_init/60)
            time_s = int(time_init - time_m * 60)
            time_list.append(str(time_m))
            time_list.append('m ')
        else:
            time_s = int(time_init)

        time_list.append(str(time_s))
        time_list.append('s')
        time_str = ''.join(time_list)
        return time_str
    def time_cal(self,i,N,fold):
        now_time=time.time()
        self.remain_time=(now_time-self.last_time)*(N-i-1)
        self.last_time=now_time
        print('#'*20,f"Fold{fold}-剩余时间："+self.time_change(self.remain_time),'#'*20)


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
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            # T.RandomRotation(90),
            # T.ColorJitter(),
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
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    #-----------------------创建模型-----------------------
    model = create_model(args,in_channels=3,num_classes=num_classes,base_c=args.base_c).to(device)




    #-----------------------加载数据-----------------------
    dataset =  VOCSegmentation(args.data_path,
                                    year="2007",
                                    transforms=get_transform(args,train=True),
                                    txt_name="trainval.txt")


    # 定义折数
    num_folds = args.Kfolds

    # 定义交叉验证
    kf = KFold(n_splits=num_folds, shuffle=True)
    #-----------------------训练-----------------------
    time_init = time.time() # 计时开始
    # 开始交叉验证训练
    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        best_score = 0.
        start_time = time.time()
        time_calc = Time_calculater()
        fold_tmp = fold + 1
        print(f"Fold {fold_tmp}")

        # 创建训练集和验证集的子集
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        # 创建训练集和验证集的数据加载器
        train_dataloader = DataLoader(train_subset,
                                               batch_size=args.batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=dataset.collate_fn,
                                               drop_last=True)
        val_dataloader = DataLoader(val_subset,
                                               batch_size=args.batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=dataset.collate_fn,
                                               drop_last=True)
        # -----------------------创建模型-----------------------
        model=initialize_weights(model)

        # -----------------------创建优化器-----------------------
        if num_classes == 2:
            # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
            loss_weight = torch.as_tensor([1.0, 2.0], device=device)
        else:
            loss_weight = None
        loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255)
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
        # -----------------------创建学习率更新策略-----------------------
        scaler = torch.cuda.amp.GradScaler() if args.amp else None  # 混合精度训练
        # -----------------------创建学习率更新策略-----------------------
        scaler = torch.cuda.amp.GradScaler() if args.amp else None  # 混合精度训练
        # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_dataloader), args.epochs, warmup=True)


     # 训练当前折的模型
        for epoch in range(args.start_epoch, args.epochs):
            mean_loss, lr = train_one_epoch(model, loss_fn, args.w_t, optimizer, train_dataloader, device, epoch,
                                            num_classes,
                                            lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
            # -----------------------验证-----------------------
            # 每10个epoch验证一次
            if (epoch+1) % args.val_freq == 0:
                confmat, dice = evaluate(model, val_dataloader, device=device, num_classes=num_classes)
            # -----------------------保存日志-----------------------
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                train_log = "train_loss: {:.4f}, lr: {:.6f}".format(mean_loss, lr)
                print('--train_log:', train_log)
                if (epoch+1) % args.val_freq == 0:
                    val_log = confmat
                    val_log["dice loss"] = format(dice, '.4f')
                    print('--val_log:', val_log)
                if epoch == args.epochs - 1 and fold_tmp == num_folds :
                    # f.write("args: {}  \n".format(args))
                    f.write(str(args))
                    model_size = calculater_1(model, (3, args.base_size, args.base_size), device)
                    f.write("model_name: -{}-  \n".format(args.model_name))
                    f.write("datasets: {}  \n".format(args.data_path))
                    f.write('flops:{:.2f}  params:{:.2f}  \n'.format(model_size[0], model_size[1]))
                # -----------------------打印时间-----------------------
                time_calc.time_cal(epoch, args.epochs,fold_tmp)
            # -----------------------保存tensorboard-----------------------
            tb.add_scalar("train/loss", mean_loss, epoch)
            tb.add_scalar("train/lr", lr, epoch)
            if (epoch+1) % args.val_freq == 0 or epoch == args.epochs - 1:
                tb.add_scalar("val/dice loss", dice, epoch)
                tb.add_scalar("val/miou", confmat["miou"], epoch)
                tb.add_scalar("val/acc", confmat["mpa"], epoch)

                # 写入到csv文件
                header_list = ["epoch", "dice", "miou", "mpa", 'pa', 'train_loss', 'lr', 'cpa', 'iou']
                with open(log_dir + f'/val_log_{fold_tmp}.csv', 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=header_list)
                    if epoch == 0:
                        writer.writeheader()
                    writer.writerow({"epoch": epoch, "dice": dice * 100, "miou": confmat["miou"], "mpa": confmat["mpa"],
                                     'pa': confmat['pa'], 'train_loss': mean_loss,
                                     'lr': lr, 'cpa': confmat['cpa'], 'iou': confmat['iou']})

            # -----------------------保存模型-----------------------
                if args.save_best is True:
                    if best_score <  confmat["miou"]:
                        best_score =  confmat["miou"]
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
                torch.save(checkpoints, log_dir + "/best_model{}.pth".format(fold_tmp))
            else:
                torch.save(checkpoints, log_dir + "/model_{}.pth".format(epoch))
        # -----------------------训练结束-----------------------
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("fold{} 训练总时间 {}".format(fold_tmp,total_time_str))
    #  打印总时间
    total_time = time.time() - time_init
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("训练结束--总时间 {}".format(total_time_str))
    # -----------------------释放内存-----------------------
    del model, optimizer, lr_scheduler, train_dataloader, val_dataloader, loss_fn, confmat, dice
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    gc.collect()



def create_model(args, in_channels, num_classes,base_c=32):
    if args.model_name == "Unet0":
        model = Unet0(in_channels=in_channels, num_classes=num_classes, base_c=base_c)
    elif args.model_name == "deeplabV3p":
        model = deeplabv3_resnet50(num_classes=num_classes, pretrained_backbone=False)
    elif args.model_name == "lraspp_mobilenetv3_large":
        model = lraspp_mobilenetv3_large(num_classes=num_classes, pretrain_backbone=False)
    elif args.model_name == "FCN":
        model = fcn_resnet50(aux=False, num_classes=num_classes)
    elif args.model_name == "SegNet":
        model = SegNet(num_classes=num_classes)
    elif args.model_name == "DenseASPP":
        model = DenseASPP(num_classes, backbone='densenet121',pretrained_base=False)
    elif args.model_name == "X_unet_fin_all8":# 自己提出的模型
        model = X_unet_fin_all8(in_channels=in_channels, num_classes=num_classes, base_c=base_c)
    elif args.model_name == "unet_t1":# 自己提出的模型
        model = unet_t1(in_channels=in_channels, num_classes=num_classes, base_c=base_c)
    elif args.model_name == "unet_t2":# 自己提出的模型
        model = unet_t2(in_channels=in_channels, num_classes=num_classes, base_c=base_c)



    else:
        raise ValueError("wrong model name")
    return initialize_weights(model)

def parse_args(model_name=None):
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--model_name", default=model_name, help="模型名称")
    parser.add_argument("--optimizer", default='adam',choices=['sgd','adam'] ,help="优化器")
    parser.add_argument("--base_size", default=256, type=int, help="图片缩放大小")
    parser.add_argument("--crop_size", default=256,  type=int, help="图片裁剪大小")
    parser.add_argument("--base_c", default=32, type=int, help="uent的基础通道数")
    parser.add_argument("--Kfolds", default=5, type=int, help="k折验证数")
    parser.add_argument('--save_method',default='all' ,choices=['all','dict'],help='保存模型的方式')
    parser.add_argument('--pretrained', default='',help='预训练模型路径')
    parser.add_argument('--w_t', default=0.5,help='dice loss的权重')

    # parser.add_argument("--data-path", default=r"..\VOC_MLCC_6_multi", help="VOC数据集路径")
    parser.add_argument("--data-path", default=r"..\VOCdevkit_cap_c5_bin", help="VOC数据集路径")
    # exclude background
    parser.add_argument("--num-classes", default=6, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=6, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--val_freq', default=5, type=int, help='每隔多少个轮次验证一次')

    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=40, type=int, help='print frequency')
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
    setup_seed(1)
    args = parse_args('X_unet_fin_all8')
    main(args)


