import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import os
import shutil
from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def resize_image(img, new_size, interpolation=cv2.INTER_LINEAR):
  # resize an image into new_size (w * h) using specified interpolation
  # opencv has a weird rounding issue & this is a hacky fix
  # ref: https://github.com/opencv/opencv/issues/9096
  mapping_dict = {cv2.INTER_NEAREST: Image.NEAREST}
  if interpolation in mapping_dict:
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(new_size,
                             resample=mapping_dict[interpolation])
    img = np.array(pil_img)
  else:
    img = cv2.resize(img, new_size,
                     interpolation=interpolation)
  return img


def vis_ms(img_ms, r, g, b):
    """

    :param img_ms: tensor images [B, C, H, W]
    :param r: int
    :param g: int
    :param b: int
    :return:
    """
    # extract rgb bands from multispectral image
    img_ms_subspec = torch.cat((img_ms[:, r].unsqueeze(1), img_ms[:, g].unsqueeze(1), img_ms[:, b].unsqueeze(1)), dim=1)
    return img_ms_subspec


SMOOTH = 1e-6

def metric_pytorch(y_pred, y_true):
    """
    :param y_pred: (tensor) 1D
    :param y_true: (tensor) 1D
    :return: precision, recall, f1, accuracy
    """
    assert y_pred.shape == y_true.shape

    n_sample = y_true.numel()
    acc = (y_pred == y_true).sum().float() / n_sample

    # TP = np.logical_and(y_pred, y_true).sum()
    TP = np.logical_and(y_pred.cpu(), y_true.cpu()).sum()
    FP = y_pred.cpu().sum() - TP
    FN = y_true.cpu().sum() - TP
    TN = (y_pred.cpu() == y_true.cpu()).sum() - TP

    precision = (TP + SMOOTH) / (TP + FP + SMOOTH)
    recall = (TP + SMOOTH) / (TP + FN + SMOOTH)
    f1 = 2.0 / (1.0 / precision + 1.0 / recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return precision, recall, f1, accuracy


def save_tensor_imgs(imgs, root_dir, ids, mean=None, std=None, scale=1):
    """

    :param imgs: tensor, [B, C, H, W]
    :return:
    """

    batch = imgs.shape[0]
    for b in range(batch):
        img = imgs[b]
        if mean is not None and std is not None:
            for t, m, s in zip(img, mean, std):
                t.mul_(s).add_(m).mul_(scale)
        else:
            for t in img:
                t.mul_(scale)
        img_np = img.detach().cpu().numpy().transpose((1, 2, 0))
        np.save(os.path.join(root_dir, '{}_sr_ms.npy'.format(ids[b])), img_np)



def show_tensor_img(img):
    """
    show tensor image
    :param img: (tensor) [C, H, W]
    :return:
    """
    plt.figure()
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, root_dir):
    filename_chk = os.path.join(root_dir, 'checkpoint.pth.tar')
    torch.save(state, filename_chk)
    if is_best:
        filename_modelbest = os.path.join(root_dir, 'model_best.pth.tar')
        shutil.copyfile(filename_chk, filename_modelbest)


def iou_pytorch(y_pred, y_true):
    """
    IoU of the outputs
    :param y_pred: (tensor) [B, 1, H, W]
    :param y_true: (tensor) [B, 1, H, W]
    :return: IoU score
    """
    # BATCH x 1 x H x W => BATCH x H x W
    # y_pred = y_pred.squeeze(dim=1)
    # y_true = y_true.squeeze(dim=1)

    # intersection = (y_pred & y_true).sum((1, 2)).float()  # Will be zero if Truth=0 or Prediction=0,
    # union = (y_pred | y_true).sum((1, 2)).float()   # Will be zero if both are 0
    # iou = (intersection + eps) / (union + eps)  # We smooth our devision to avoid 0/0

    intersection = np.logical_and(y_pred.cpu(), y_true.cpu())
    union = np.logical_or(y_pred.cpu(), y_true.cpu())
    iou_score = torch.sum(intersection + SMOOTH) / torch.sum(union + SMOOTH)
    # thresholded = torch.clamp(20 * (iou_score - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    print('iou_score: ', iou_score)
    # return thresholded.mean()
    return iou_score.mean()  # Or thresholded.mean() if you are interested in average across the batch


def weight_tensor(groundtruth, nonflood_weight, flood_weight):
    weight = []
    label = groundtruth.view(-1)
    for px in label:
        if px == 0:
            weight.append(nonflood_weight)
        else:
            weight.append(flood_weight)
    return torch.FloatTensor(weight)
