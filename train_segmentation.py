import argparse
import os
import shutil
import time
import csv
from PIL import Image
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

from library import iou_metric_batch, AverageMeter, get_mask_type
from dataloader import ImageFolder_supervision
from models.unet import UNet_supervision_dilate
from hyper_parameter import hp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_loss = 10
torch.manual_seed(30)


def main():
    root = os.path.join("./",
                        "something")

    # best_loss = 10
    cudnn.benchmark = True

    scores = []
    scores_2 = []
    losses = []
    # for ite in range(hp["start_iteration"], hp["KFold"]):
    print("START TRAINING : UNet")
    for ite in range(8):
        model = UNet_supervision_dilate().to(device)
        optimizer = torch.optim.SGD(model.parameters(), hp["lr"],
                                    momentum=hp["momentum"],
                                    weight_decay=hp["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0.001)
        best_loss = 10
        best_score = 0
        best_score_2 = 0

        if 0 == 0:
            hp["resume"] = os.path.join(root,  "cv_{}_best_score.pth.tar".format(ite))
            if hp["resume"]:
                if os.path.isfile(hp["resume"]):
                    # model = UNet_default()
                    model = UNet_supervision_dilate().to(device)
                    optimizer = torch.optim.SGD(model.parameters(), hp["lr"],
                                                momentum=hp["momentum"],
                                                weight_decay=hp["weight_decay"])
                    print("=> loading checkpoint '{}'".format(hp["resume"]))
                    checkpoint = torch.load(hp["resume"])
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    # best_loss = checkpoint["best_loss"]
                    if ite == 0:
                        hp["start_epoch"] = checkpoint["epoch"]
                    else:
                        hp["start_epoch"] = 0
                    # best_score_2 = checkpoint["best_score"]
                    print(checkpoint["best_score"])
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(hp["resume"], checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(hp["resume"]))
                    hp["start_epoch"] = 0
        else:
            hp["start_epoch"] = 0

        train_loader = torch.utils.data.DataLoader(
            ImageFolder_supervision(
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(101, scale=(0.9, 1.0)),
                    transforms.Pad((13, 13, 14, 14), padding_mode="reflect"),
                    transforms.ColorJitter(),
                    transforms.ToTensor()
                ]),
                eval=False, iteration=ite,n_splits=8
            ),
            batch_size=hp["batch_size"], shuffle=True,
            num_workers=hp["workers"], pin_memory=False, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFolder_supervision(
                transform=transforms.Compose([
                    transforms.Pad((13, 13, 14, 14), padding_mode="reflect"),
                    transforms.ToTensor(),
                ]),
                eval=True, iteration=ite, n_splits=8
            ),
            batch_size=hp["batch_size"], shuffle=False,
            num_workers=hp["workers"], pin_memory=False, drop_last=False)

        # if 1 == 1:
        #     train_loader = torch.utils.data.DataLoader(
        #         ImageFolder_supervision(
        #             transform=transforms.Compose([
        #                 transforms.Resize(128),
        #                 transforms.ToTensor()
        #             ]),
        #             eval=False, iteration=ite, n_splits=hp["KFold"]
        #         ),
        #         batch_size=1, shuffle=True,
        #         num_workers=hp["workers"], pin_memory=False, drop_last=True)
        #
        #     val_loader = torch.utils.data.DataLoader(
        #         ImageFolder_supervision(
        #             transform=transforms.Compose([
        #                 # transforms.Pad((13, 13, 14, 14), padding_mode="reflect"),
        #                 transforms.Resize(128),
        #                 transforms.ToTensor(),
        #             ]),
        #             eval=True, iteration=ite, n_splits=10
        #         ),
        #         batch_size=1, shuffle=False,
        #         num_workers=hp["workers"], pin_memory=False, drop_last=False)
        #
        #     model = UNet_deep_supervision().to(device)
        #     checkpoint = torch.load("/media/kento/HDCZ-UT/kaggle/TGS_Salt_Identification_Challenge/results/UNet_deep_supervision/128/cv_{}_best_score.pth.tar".format(ite))
        #     model.load_state_dict(checkpoint['state_dict'])
        #     print(checkpoint["best_score"])
        #     print("--------ite {0}---------".format(ite))
        #     check_score(train_loader, model)
        #     check_score(val_loader, model)
        #     break

        cycle_score = None
        cycle_iter = 0
        for epoch in range(hp["start_epoch"], hp["epoch"]):
            # adjust_learning_rate(optimizer, epoch)
            if epoch % 50 == 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=0.001)
            if epoch < 0:
                # optimizer.step()
                pass
            else:
                scheduler.step()
            print("--------ite {1} epoch {0} LR {2:.8f} {3}---------".format(epoch, ite,
                                                                         optimizer.param_groups[0]["lr"],
                                                                             time.ctime()))

            # train for one epoch
            loss, loss2, score, score5, acc, acc5, super = train(train_loader, model, optimizer, epoch, ite)
            v_loss, v_loss2, v_score, v_score5, v_acc, v_acc5, v_super = validate(val_loader, model, epoch, ite)

            print("Train Loss: {0:.4f}({loss2:.4f})\tTrain score: {1:.4f}\tscore5: {2:.4f}\t"
                  "Train Acc: {3:.4f}\tAcc5: {4:.4f}\tsuper: {super:.4f}\n"
                  "Val   Loss: {5:.4f}({v_loss2:.4f})\tVal   score: {6:.4f}\tscore5: {7:.4f}"
                  "\tVal   Acc: {8:.4f}\tAcc5: {9:.4f}\tsuper: {v_super:.4f}".format(
                loss, score, score5, acc, acc5, v_loss, v_score, v_score5, v_acc, v_acc5, loss2=loss2, v_loss2=v_loss2,
                super=super, v_super=v_super))

            # if epoch>=0 and (epoch)%50==0:
            #     cycle_score = 0
            #     cycle_iter += 1
            # if cycle_score is not None:
            #     cycle_is_best = v_score5 > cycle_score
            #     cycle_score = max(v_score5, cycle_score)
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'state_dict': model.state_dict(),
            #         'best_loss': v_loss,
            #         'optimizer': optimizer.state_dict(),
            #         'best_score': v_score5,
            #         "ite": ite,
            #     }, False, cycle_is_best,
            #         filename=os.path.join(root, "snaps/cv_{0}_{1}_checkpoint.pth.tar".format(ite, cycle_iter)),
            #         best_filename=None,
            #         best_filename_score=os.path.join(root, "snaps/cv_{0}_{1}_best_score.pth.tar".format(ite, cycle_iter)))

            # remember best prec@1 and save checkpoint
            with open(os.path.join(root, "cv_{}_log.txt".format(ite)), "a") as f:
                f.write("--------ite {1} epoch {0}---------".format(epoch, ite) + "\n")
                f.write("Train Loss: {0:.4f}\tTrain score: {1:.4f}\tscore5: {2:.4f}\t"
                        "Train Acc: {3:.4f}\tAcc5: {4:.4f}\n"
                        "Val   Loss: {5:.4f}\tVal   score: {6:.4f}\tscore5: {7:.4f}\t"
                        "Val   Acc: {8:.4f}\tAcc5: {9:.4f} \n".format(
                            loss, score, score5, acc, acc5, v_loss, v_score, v_score5, v_acc, v_acc5))

            is_best = v_loss < best_loss
            is_best_score = v_score5 > best_score_2
            if is_best:
                best_score = v_score5
            best_loss = min(v_loss, best_loss)
            best_score_2 = max(v_score5, best_score_2)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': v_loss,
                'optimizer': optimizer.state_dict(),
                'best_score': v_score5,
                "ite": ite,
                "scheduler": scheduler.state_dict(),
            }, is_best, is_best_score,
                filename=os.path.join(root, "cv_{}_checkpoint.pth.tar".format(ite)),
                best_filename=os.path.join(root, "cv_{}_best_model.pth.tar".format(ite)),
                best_filename_score=os.path.join(root, "cv_{}_best_score.pth.tar".format(ite)))
        scores.append(best_score)
        scores_2.append(best_score_2)
        losses.append(best_loss)
    print("CV Loss: {0}\nCV score: {1}\nCV score2: {2}".format(losses, scores, scores_2))
    print("CV Loss: {0:.4f}\tCV: {1:.4f}\tCV: {2:.4f}".format(sum(losses)/len(losses),
                                                              sum(scores)/len(scores),
                                                              sum(scores_2)/len(scores_2)))


def train(train_loader, model, optimizer, epoch, ite):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses2 = AverageMeter()
    scores = AverageMeter()
    scores5 = AverageMeter()
    accu = AverageMeter()
    accu5 = AverageMeter()
    supee = AverageMeter()
    threshold = 0.4

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, truth) in enumerate(train_loader):
        # print(truth)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = target.size(0)
        input = input.to(device)
        target = target.to(device)
        truth = truth.to(device)

        # compute output
        logit, logit_pixel, logit_image = model(input)
        loss_pixel, loss_image, loss_logit = model.criterion(logit, logit_pixel, logit_image, target, truth)
        loss = loss_pixel + loss_image + loss_logit

        # measure accuracy and record loss
        logit = logit.detach().cpu().numpy()[:, :, 13:114, 13:114]
        target = target.detach().cpu().numpy()[:, :, 13:114, 13:114]

        score = iou_metric_batch(target,
                                 np.int32(logit > 0.))
        score5 = iou_metric_batch(target,
                                  np.int32(logit > threshold))
        ac = accuracy(target,
                      np.int32(logit > 0.))
        ac5 = accuracy(target,
                       np.int32(logit > threshold))
        sup = accuracy(truth.detach().cpu().numpy(),
                       np.int32(logit_image > threshold))

        losses2.update(loss_image.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        scores.update(score.mean().sum(), input.size(0))
        scores5.update(score5.mean().sum(), input.size(0))
        accu.update(ac)
        accu5.update(ac5)
        supee.update(sup)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if 1 + i % hp["print_freq"] == 0:
            print('Epoch: [{3}:{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}({losses2.avg:.4f})\t'
                  'Score {scores.avg:.3f}\t'
                  'Acc {acc.avg:.3f}'.format(
                epoch, i, len(train_loader), ite, batch_time=batch_time,
                data_time=data_time, loss=losses, scores=scores, acc=accu, losses2=losses2))
    return losses.avg, losses2.avg, scores.avg, scores5.avg, accu.avg, accu5.avg, supee.avg


def validate(val_loader, model, epoch, ite):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses2 = AverageMeter()
    scores = AverageMeter()
    scores5 = AverageMeter()
    accu = AverageMeter()
    accu5 = AverageMeter()
    supee = AverageMeter()
    threshold = 0.4

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, truth) in enumerate(val_loader):

            batch_size = target.size(0)
            input = input.to(device)
            target = target.to(device)
            truth = truth.to(device)

            # compute output
            logit, logit_pixel, logit_image = model(input)
            loss_pixel, loss_image, loss_logit = model.criterion(logit, logit_pixel, logit_image, target, truth)
            loss = loss_pixel + loss_image + loss_logit

            # measure accuracy and record loss
            logit = logit.detach().cpu().numpy()[:, :, 13:114, 13:114]
            target = target.detach().cpu().numpy()[:, :, 13:114, 13:114]

            score = iou_metric_batch(target,
                                     np.int32(logit > 0.))
            score5 = iou_metric_batch(target,
                                      np.int32(logit > threshold))
            ac = accuracy(target,
                          np.int32(logit > 0.))
            ac5 = accuracy(target,
                           np.int32(logit > threshold))
            sup = accuracy(truth.detach().cpu().numpy(),
                           np.int32(logit_image > threshold))
            # print(score, ac)

            losses2.update(loss_image.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
            scores.update(score.mean().sum(), input.size(0))
            scores5.update(score5.mean().sum(), input.size(0))
            accu.update(ac)
            accu5.update(ac5)
            supee.update(sup)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % hp["print_freq"] == 0:
                print('Epoch: [{3}:{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}({losses2.avg:.4f})\t'
                      'Score {scores.avg:.3f}\t'
                      'Acc {acc.avg:.3f}'.format(
                    epoch, i, len(val_loader), ite, batch_time=batch_time,
                    data_time=batch_time, loss=losses, scores=scores, acc=accu, losses2=losses2))
        return losses.avg, losses2.avg, scores.avg, scores5.avg, accu.avg, accu5.avg, supee.avg


def SWA(model_swa, model, n, c):
    _n = n/c
    for w_swa, w in zip(model_swa.parameters(), model.parameters()):
        w_swa.data *= _n/(_n+1)
        w_swa.data += w.data/(_n+1)

def accuracy(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return (x==y).sum().item() / x.shape[0]


def save_checkpoint(state, is_best, is_best_score, filename, best_filename, best_filename_score):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)
    if is_best_score:
        shutil.copyfile(filename, best_filename_score)


if __name__ == "__main__":
    main()