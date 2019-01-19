import numpy as np
import pandas as pd
import cv2

from hyper_parameter import hp


def get_iou_vector(A, B, n="GT-P"):
    A[A>=0.5] = 1
    A[A<0.5] = 0
    # print(A)
    # print("######")
    # print(B)
    batch_size = A.shape[0]
    intersection = np.logical_and(A, B)
    # print(intersection)
    union = np.logical_or(A, B)
    # print(union)
    iou = np.sum(intersection.reshape(batch_size,-1) > 0, axis=1) /  \
          np.sum(union.reshape(batch_size,-1) > 0, axis=1)
    s = pd.Series(name=n)
    for thresh in np.arange(0.5,1,0.05):
        s[thresh] = iou > thresh
    return s


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(),
                                  bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:,1:]
    union[union==0] = 1e-9

    #Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1 # Correct objects
        false_positives = np.sum(matches, axis=0) == 0 # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0 # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp +fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


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


def get_mask_type(mask, is_vertical=True):
    border = 10
    outer = np.zeros((128-2*border, 128-2*border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType=cv2.BORDER_CONSTANT, value=1)

    cover = (mask>0.5).sum()
    if cover<1:
        return "empty"
    # if cover==((mask*outer)>0.5).sum():
    if cover < 150:
        return "boarder"
    # if np.all(mask==mask[0]) & is_vertical:
    #     return "vertical"

    percentage = cover/(128*128)
    if percentage < 0.05:
        return "vertical"
    if percentage<0.15:
        return "object015"
    elif percentage<0.25:
        return "object025"
    elif percentage<0.50:
        return "object050"
    elif percentage<0.75:
        return "object075"
    else:
        return "object100"


def add_depth_channels(image_tensor):
    _, _, h, w = image_tensor.size()
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_tensor[:, 1, row, :] = const
    image_tensor[:,2,:,:] = image_tensor[:,0,:,:] * image_tensor[:,1,:,:]
    return image_tensor



if __name__=="__main__":
    x = np.random.randn(1,10,10)
    y = np.random.randn(1,10,10)
    print(x)
    print(y)
    score = get_iou_vector(x,y)
    # print(score)
    # print(score.shape)
    print(score.mean().sum())