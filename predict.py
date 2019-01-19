import os
import time
import gc
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from hyper_parameter import hp
from library import iou_metric, AverageMeter
from dataloader import SubmitFolder
from models.unet import UNet_supervision_dilate, UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(30)


def main():
    num_models = 32
    threshold = 0.4

    val_loader = torch.utils.data.DataLoader(
        SubmitFolder(
            transform=transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
            ]),
        TTA=True),
        batch_size=32, shuffle=False,
        num_workers=hp["workers"], pin_memory=False, drop_last=False)
    Pred = None

    # 1st Model
    root = os.path.join("./",
                        "something")
    model = UNet_supervision_dilate().to(device)
    for ite in range(8):
        hp["resume"] = os.path.join(root, "cv_{}_best_score.pth.tar".format(ite))
        if hp["resume"]:
            if os.path.isfile(hp["resume"]):
                checkpoint = torch.load(hp["resume"])
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(hp["resume"], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(hp["resume"]))
                raise ValueError(print(hp["resume"]))
        else:
            raise ValueError(print(hp["resume"]))
        if Pred is None:
            Pred, ids = valid_resize(model, val_loader, ite)
            Pred = Pred / num_models
        else:
            Pred = Pred + valid_resize(model, val_loader, ite)[0] / num_models

    # 2nd Model
    root = os.path.join("./",
                        "something")
    model = UNet().to(device)
    for ite in range(8):
        hp["resume"] = os.path.join(root, "cv_{}_best_score.pth.tar".format(ite))
        if hp["resume"]:
            if os.path.isfile(hp["resume"]):
                checkpoint = torch.load(hp["resume"])
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(hp["resume"], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(hp["resume"]))
                raise ValueError(print(hp["resume"]))
        else:
            raise ValueError(print(hp["resume"]))
        if Pred is None:
            Pred, ids = valid_resize(model, val_loader, ite)
            Pred = Pred / num_models
        else:
            Pred = Pred + valid_resize(model, val_loader, ite)[0] / num_models

    # Pred = Pred / 8
    final = np.zeros((len(Pred), 101, 101))
    for i in tqdm(range(len(Pred))):
        final[i] = resize(Pred[i].reshape(128, 128) / 1600, (101, 101)) * 1600
    Pred = final
    del final
    gc.collect()

    # 3rd Model
    root = os.path.join("./",
                        "something")
    model = UNet().to(device)
    val_loader = torch.utils.data.DataLoader(
        SubmitFolder(
            transform=transforms.Compose([
                transforms.Pad((13, 13, 14, 14), padding_mode="reflect"),
                transforms.ToTensor(),
            ]),
        TTA=True),
        batch_size=32, shuffle=False,
        num_workers=hp["workers"], pin_memory=False, drop_last=False)
    for ite in range(8):
        hp["resume"] = os.path.join(root, "cv_{}_best_score.pth.tar".format(ite))
        if hp["resume"]:
            if os.path.isfile(hp["resume"]):
                checkpoint = torch.load(hp["resume"])
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(hp["resume"], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(hp["resume"]))
                raise ValueError(print(hp["resume"]))
        else:
            raise ValueError(print(hp["resume"]))
        if Pred is None:
            Pred, ids = valid_pad(model, val_loader, ite)
            Pred = Pred / num_models
        else:
            Pred = Pred + valid_pad(model, val_loader, ite)[0] / num_models

    # 4th Model
    root = os.path.join("./",
                        "something")
    model = UNet_supervision_dilate().to(device)
    for ite in range(8):
        hp["resume"] = os.path.join(root, "cv_{}_best_score.pth.tar".format(ite))
        if hp["resume"]:
            if os.path.isfile(hp["resume"]):
                checkpoint = torch.load(hp["resume"])
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(hp["resume"], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(hp["resume"]))
                raise ValueError(print(hp["resume"]))
        else:
            raise ValueError(print(hp["resume"]))
        if Pred is None:
            Pred, ids = valid_pad(model, val_loader, ite)
            Pred = Pred / num_models
        else:
            Pred = Pred + valid_pad(model, val_loader, ite)[0] / num_models


    # Pred = Pred / 8

    Pred = (Pred > threshold).astype(int)
    Predict = {idx: rle_encode(Pred[i]) for i, idx in
               enumerate(ids)}

    sub = pd.DataFrame.from_dict(Predict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('something.csv')


def valid_resize(model, val_loader, ite):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    Prediction = None
    Prediction_flip = None
    ids = []
    with torch.no_grad():
        for i, (input, input_flip, id) in enumerate(val_loader):
            # print(input)
            input = input.to(device)
            input_flip = input_flip.to(device)

            # compute output
            output, logit_image, _ = model(input)
            # output = F.elu(output)
            output_f, logit_image_f, _ = model(input_flip)
            # output_f = F.elu(output_f)

            output = output.detach().cpu().numpy()
            output_f = output_f.detach().cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % hp["print_freq"] == 0:
                print('Test: [{2}][{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(val_loader), ite, batch_time=batch_time))

            if i == 0:
                Prediction = output
                Prediction_flip = output_f
            else:
                Prediction = np.concatenate([Prediction, output], 0)
                Prediction_flip = np.concatenate([Prediction_flip, output_f], 0)

            ids += id

    Prediction = Prediction.squeeze(1)

    Prediction_flip = Prediction_flip.squeeze(1)
    Prediction_flip = horizontal_flip(Prediction_flip)

    Prediction = Prediction + Prediction_flip
    Prediction = Prediction / 2

    return Prediction, ids


def valid_pad(model, val_loader, ite):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    Prediction = None
    Prediction_flip = None
    ids = []
    with torch.no_grad():
        for i, (input, input_flip, id) in enumerate(val_loader):
            input = input.to(device)
            input_flip = input_flip.to(device)

            # compute output
            output, logit_image, _ = model(input)
            # output = F.elu(output)
            output_f, logit_image_f, _ = model(input_flip)
            # output_f = F.elu(output_f)

            output = output.detach().cpu().numpy()
            output_f = output_f.detach().cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % hp["print_freq"] == 0:
                print('Test: [{2}][{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                    i, len(val_loader), ite, batch_time=batch_time))

            if i == 0:
                Prediction = output
                Prediction_flip = output_f
            else:
                Prediction = np.concatenate([Prediction, output], 0)
                Prediction_flip = np.concatenate([Prediction_flip, output_f], 0)

            ids += id

        Prediction = Prediction.squeeze(1)[:, 13:114, 13:114]
        Prediction_flip = Prediction_flip.squeeze(1)[:, 13:114, 13:114]
        Prediction_flip = horizontal_flip(Prediction_flip)
        Prediction = Prediction + Prediction_flip
        Prediction = Prediction / 2

    return Prediction, ids


def horizontal_flip(image):
    image = image[:, :, ::-1]
    return image


def rle_encode(im):
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


if __name__=="__main__":
    main()