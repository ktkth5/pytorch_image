import numpy as np
import pandas as pd
import cv2
import torch

from hyper_parameter import hp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def loss_function(input, target):
    loss = (input - target) ** 2
    weight = target.detach() # (1, 2)
    weight += 1
    return torch.mean(loss * weight)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ThreatScore(object):
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.recall = 0
        self.precision = 0

    def update(self, output, target):
        pred = output.argmax(1)
        pred = pred.reshape(-1, 1)
        target = target.reshape(-1, 1)
        for p, t in zip(pred, target):
            if p == 1:
                if p == t:
                    self.tp += 1
                    # print("a")
                else:
                    self.fp += 1
            else:
                if p == t:
                    self.tn += 1
                    # print("b")
                else:
                    self.fn += 1

        if self.tp+self.fn == 0:
            self.recall = 0
        else:
            self.recall = self.tp / (self.tp + self.fn)
        if self.tp + self.fp == 0:
            self.precision = 0
        else:
            self.precision = self.tp / (self.tp + self.fp)
        if (self.tn+self.fp) == 0:
            self.fp_tn = 0
        else:
            self.fp_tn = self.fp / (self.tn + self.fp)




if __name__=="__main__":
    x = torch.randn(5,2)
    y = torch.randn(5)
    pred = x.argmax(0)
    mask = y > 0
    print(mask)
    print(x)
    print(x.argmax(1))

    print(pred[mask])

