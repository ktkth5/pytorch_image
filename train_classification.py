import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import models
from library.common import AverageMeter, accuracy, ThreatScore
from dataloader import ImageFolder_cla

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

def trainer(hp):
    best_score = 0
    torch.manual_seed(30)

    root = os.path.join("ROOT_PATH", "MODEL_DIR")
    file_name = os.path.join(root, "checkpoint.pth.tar")
    best_model_file = os.path.join(root, "best_model.pth.tar")
    log_file = os.path.join(root, "log.txt")
    writer = SummaryWriter(f"{root}/runs")

    # define loss function (criterion) and optimizer
    model = models.se_resnet18(num_classes=2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), hp["lr"],
                                momentum=hp["momentum"],
                                weight_decay=hp["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([2,1])).to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, hp["epoch"], eta_min=0.001)

    # optionally resume from a checkpoint
    if hp["resume"]:
        if os.path.isfile(hp["resume"]):
            print("=> loading checkpoint '{}'".format(hp["resume"]))
            checkpoint = torch.load(hp["resume"])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint["scheduler"])
            hp["start_iteration"] = checkpoint["ite"]
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(hp["resume"], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(hp["resume"]))

    train_loader = torch.utils.data.DataLoader(
        ImageFolder_cla(
            csv_file="train.csv",
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(),
                transforms.ToTensor()
            ]),
            mode="train",
        ),
        batch_size=hp["batch_size"], shuffle=True,
        num_workers=hp["workers"], pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        ImageFolder_cla(
            csv_file="test.csv",
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            mode="val",
        ),
        batch_size=hp["batch_size"], shuffle=False,
        num_workers=hp["workers"], pin_memory=False, drop_last=False)

    cudnn.benchmark = True

    for epoch in range(hp["start_epoch"], hp["epoch"]*2):
        start = time.time()
        # adjust_learning_rate(optimizer, epoch)
        if epoch % hp["epoch"] == 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, hp["epoch"], eta_min=0.001)
        scheduler.step()
        print("--------epoch {0} LR {1:.8f}---------".format(epoch,
                                                             optimizer.param_groups[0]["lr"]))

        # train for one epoch
        loss, acc, precision, recall, prec = train(train_loader, model, criterion, optimizer, epoch, hp)
        v_loss, v_acc, v_precision, v_recall, v_prec = validate(val_loader, model, criterion, epoch, hp)

        # remember best prec@1 and save checkpoint
        writer.add_scalar(f"data/cv_{fold}_train_loss", loss, epoch)
        writer.add_scalar(f"data/cv_{fold}_val_loss", v_loss, epoch)
        writer.add_scalar(f"data/cv_{fold}_train_accuracy", acc, epoch)
        writer.add_scalar(f"data/cv_{fold}_val_accuracy", v_acc, epoch)
        log = f"Train Loss: {loss:.4f}\tAccuracy: {acc:.4f}\n" \
              f"Val   Loss: {v_loss:.4f}\tAccuracy: {v_acc:.4f}\n"
        print(log)
        with open(log_file, "a") as f:
            log = f"-----Epoch[{fold}:{epoch}] LR {optimizer.param_groups[0]['lr']}-----\n" + log
            f.write(log)

        is_best = v_acc > best_score
        best_score = max(v_acc, best_score)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': v_loss,
                'optimizer': optimizer.state_dict(),
                'best_score': v_acc,
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            filename=file_name,
            best_filename=best_model_file)


def train(train_loader, model, criterion, optimizer, epoch, hp):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    prec = ThreatScore()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        ### if input channel is Gray Scale => conccatenate to 3 channels
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input = torch.cat([
            (input - mean[2]) / std[2],
            (input - mean[1]) / std[1],
            (input - mean[0]) / std[0],
        ], 1)
        """

        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        output = output.detach().cpu()
        target = target.detach().cpu()
        prec.update(output, target)
        prec1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0].item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print(type(top1.avg))

        if i % hp["print_freq"] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.2f}({data_time.avg:.2f})\t'
                  'Loss {loss.avg:.4f}\t'
                  'Prec@1 {top1.avg:.4f}\t'
                  'Precision {prec.precision:.4f}\t'
                  'Recall {prec.recall:.4f}\t'
                  'FP_TN {prec.fp_tn:.4f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, prec=prec))
        # if i > 10:
        #     break
    return losses.avg, top1.avg, prec.precision, prec.recall, prec


def validate(val_loader, model, criterion, epoch, hp):
    losses = AverageMeter()
    top1 = AverageMeter()
    prec = ThreatScore()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            ### if input channel is Gray Scale => conccatenate to 3 channels
            """
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            input = torch.cat([
                (input - mean[2]) / std[2],
                (input - mean[1]) / std[1],
                (input - mean[0]) / std[0],
            ], 1)
            """

            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            output = output.detach().cpu()
            target = target.detach().cpu()
            prec.update(output, target)
            prec1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].item(), input.size(0))

            if i % hp["print_freq"] == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.avg:.4f}\t'
                      'Prec@1 {top1.avg:.4}\t'
                      'Precision {prec.precision:.4f}\t'
                      'Recall {prec.recall:.4f}\t'
                      'FP/(FP+TN) {prec.fp_tn:.4f}'.format(
                    epoch, i, len(val_loader), loss=losses, top1=top1, prec=prec))
        return losses.avg, top1.avg, prec.precision, prec.recall, prec


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 20:
        lr = 0.01
    elif epoch < 50:
        lr = 0.001
    elif epoch < 110:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, is_best_score, filename, best_filename, best_filename_score):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)
    if is_best_score:
        shutil.copyfile(filename, best_filename_score)

if __name__ == "__main__":
    from hyper_parameter import hp
    trainer(hp)