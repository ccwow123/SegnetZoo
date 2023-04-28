import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target


def criterion0(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']
def criterion(x, target,loss_fn, loss_weight=None, num_classes: int = 2, dice: bool = False, ignore_index: int = -100):
    w_t = loss_weight
    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    loss = loss_fn(x, target)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss = w_t * loss + (1 - w_t) * dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        # loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    return loss


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    # confmat = utils.ConfusionMatrix(num_classes)#原版混淆矩阵
    confmat = utils.SegmentationMetric(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            # output = output['out']
            # 原版混淆矩阵
            # confmat.update(target.flatten(), output.argmax(1).flatten())

            confmat.update(target.flatten(), output.argmax(1).flatten())  # 这里要修夫debug试试
            confmat_output = confmat.compute()
            dice.update(output, target)
        # confmat.reduce_from_all_processes()# 原版混淆矩阵
        dice.reduce_from_all_processes()

    # return confmat, dice.value.item()# 原版混淆矩阵
    return confmat_output, dice.value.item()


def train_one_epoch(model,loss_fn,loss_weight, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # if num_classes == 2:
    #     # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
    #     # loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    #     loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    # else:
    #     loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            # loss = loss_fn.to(device)(output, target)
            # loss = loss_fn(output, target)
            loss = criterion(output, target, loss_fn,loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss, lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
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
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
