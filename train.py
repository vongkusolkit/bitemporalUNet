import argparse
import logging
import os
import time

import torch
from tensorboardX import SummaryWriter
from torchvision import transforms

import dataproc as dp
from dualbridgeunet_v2 import DualBridgeUNet
from dicebceloss import DiceBCELoss
from utils import AverageMeter, set_logger, save_checkpoint, iou_pytorch, metric_pytorch, weight_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='train_v4_knn2_wl_mean_std')
parser.add_argument('-t', '--train_batch_size', type=int, default=12)
parser.add_argument('-v', '--valid_batch_size', type=int, default=12)
parser.add_argument('-e', '--n_epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=0.01)  # learning rate
parser.add_argument('--wd', type=float, default=1e-4)  # weight decay
parser.add_argument('--beta1', type=float, default=0.5)  # weight decay
parser.add_argument('--betas', type=float, default=(0.9, 0.999)) # beta for Adam optimization
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--csv_train', type=str, default='./data/combined/florence_train.csv')
parser.add_argument('--csv_valid', type=str, default='./data/combined/florence_valid.csv')
parser.add_argument('--data_root_dir_pre', type=str, default='../florence_myunet/pre')
parser.add_argument('--data_root_dir_post', type=str, default='../florence_myunet/post')
parser.add_argument('--data_root_dir_gt', type=str, default='../florence_myunet/weakly_label_framework')

# training data statistic
mean_pre = [0.5163, 0.0163, 0.5142, 0.0163]
std_pre = [0.2672, 0.0168, 0.2682, 0.0168]
mean_post = [0.4925, 0.0133, 0.4974, 0.0133]
std_post = [0.2680, 0.0134, 0.2691, 0.0135]


def train_net(dataloader, ep, net, optimizer, criterion, writer, device):
    net.train()
    print('Training...')

    epoch_loss = AverageMeter()
    epoch_iou = AverageMeter()
    n_batches = len(dataloader)

    for batch_idx, batched_data in enumerate(dataloader):
        pixel_pre = batched_data['pixel_pre'].to(device=device, dtype=torch.float32)  # [Batch, 4, h, w]
        pixel_post = batched_data['pixel_post'].to(device=device, dtype=torch.float32)
        label = batched_data['label'].to(device=device, dtype=torch.float32)
        batch_size = pixel_post.shape[0]

        train_output = net(pixel_pre, pixel_post)
        train_predict = train_output.squeeze(dim=1)

        print(train_predict.shape)
        print(label.shape)

        train_loss = criterion(train_predict.view(-1), label.view(-1))
        epoch_loss.update(train_loss.item(), batch_size)

        print('training/train_loss', train_loss.item(), ep*n_batches+batch_idx)

        batch_predictions = (train_predict >= 0.5).int() # binarize predicitons to 0/1
        batch_masks = label.int()
        batch_precision, batch_recall, batch_f1, batch_accuracy = metric_pytorch(batch_predictions, batch_masks)
        batch_iou = iou_pytorch(batch_predictions, batch_masks)
        epoch_iou.update(batch_iou.item(), batch_size)

        writer.add_scalar('training/train_loss', train_loss.item(), ep*n_batches+batch_idx)
        writer.add_scalar('training/train_iou', batch_iou.item(), ep*n_batches+batch_idx)
        writer.add_scalar('training/train_precision', batch_precision.item(), ep*n_batches+batch_idx)
        writer.add_scalar('training/train_recall', batch_recall.item(), ep*n_batches+batch_idx)
        writer.add_scalar('training/train_f1', batch_f1.item(), ep*n_batches+batch_idx)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        print('Train Epoch finished ! Loss: {}'.format(epoch_loss.avg))

    return epoch_iou.avg, epoch_loss.avg


def valid_net(dataloader, ep, net, criterion, writer, device):
    net.eval()
    print('Validation...')

    epoch_loss = AverageMeter()
    epoch_iou = AverageMeter()
    n_batches = len(dataloader)

    for batch_idx, batched_data in enumerate(dataloader):
        pixel_pre = batched_data['pixel_pre'].to(device=device, dtype=torch.float32)  # [Batch, 4, h, w]
        pixel_post = batched_data['pixel_post'].to(device=device, dtype=torch.float32)
        label = batched_data['label'].to(device=device, dtype=torch.float32)
        batch_size = pixel_post.shape[0]

        valid_output = net(pixel_pre, pixel_post)

        valid_predict = valid_output.squeeze(dim=1)

        valid_loss = criterion(valid_predict.view(-1), label.view(-1))

        epoch_loss.update(valid_loss.item(), batch_size)

        writer.add_scalar('training/valid_loss', valid_loss.item(), ep * n_batches + batch_idx)

        batch_predictions = (valid_predict >= 0.5).int() # binarize predicitons to 0/1
        batch_masks = label.int()

        batch_iou = iou_pytorch(batch_predictions, batch_masks)
        epoch_iou.update(batch_iou.item(), batch_size)

        print('Valid Epoch finished ! Loss: {}'.format(epoch_loss.avg))

    return epoch_iou.avg, epoch_loss.avg


def main(args):
    log_dir = "./logs/logs_{}".format(args.version)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    model_dir = "./logs/models_{}".format(args.version)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    set_logger(os.path.join(model_dir, 'train.log'))
    writer = SummaryWriter(log_dir)

    logging.info("**************Auto-Encoder: Training****************")
    logging.info('version: {}'.format(args.version))
    logging.info('csv_train: {}'.format(args.csv_train))
    logging.info('csv_valid: {}'.format(args.csv_valid))
    logging.info('data root directory pre: {}'.format(args.data_root_dir_pre))
    logging.info('data root directory post: {}'.format(args.data_root_dir_post))
    logging.info('learning rate: {}'.format(args.lr))
    logging.info('beta1: {}'.format(args.beta1))
    logging.info('epochs: {}'.format(args.n_epochs))
    logging.info('train / valid batch size: {}'.format(args.train_batch_size, args.valid_batch_size))
    logging.info('loss: DiceBCELoss')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rotation_angle = (0, 90, 180, 270)

    # train set
    transform_trn = transforms.Compose([
        dp.RandomFlip(),
        dp.RandomRotate(starting_angle=rotation_angle, perturb_angle=0, prob=0.5),
        dp.ToTensor()
    ])
    # transform_trn = transforms.Compose(
    #     [dp.RandomFlip(),
    #     dp.RandomRotate(starting_angle=rotation_angle, perturb_angle = 0, prob=0.5),
    #     dp.ToTensor(),
    #     dp.Normalize(mean_pre, std_pre, mean_post, std_post)]
    #     )

    trainset = dp.PixelDataset(csv_file=args.csv_train,
                               root_dir_pre=args.data_root_dir_pre,
                               root_dir_post=args.data_root_dir_post,
                               root_dir_gt=args.data_root_dir_gt,
                               is_labeled=True,
                               transform=transform_trn)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    transform_val = transforms.Compose([
        dp.ToTensor()
    ])
    validset = dp.PixelDataset(csv_file=args.csv_valid,
                               root_dir_pre=args.data_root_dir_pre,
                               root_dir_post=args.data_root_dir_post,
                               root_dir_gt=args.data_root_dir_gt,
                               is_labeled=True,
                               transform=transform_val)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=False, num_workers=4)

    net = DualBridgeUNet().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


    min_loss = 1e06
    start_time = time.time()

    for ep in range(args.n_epochs):
        print('Epoch [{}/{}]'.format(ep + 1, args.n_epochs))
        logging.info('learning rate: {:.3e}'.format(args.lr))

        IoU_train, loss_train = train_net(trainloader, ep, net, optimizer, criterion, writer, device)

        writer.add_scalars('training/Loss', {"train": loss_train}, ep + 1)
        writer.add_scalars('training/IoU', {"train": IoU_train}, ep+1)
        logging.info('Train Epoch [{}/{}] [Loss: {:.4f}] [IoU: {:.4f}]'.format(
            ep+1, args.n_epochs, loss_train, IoU_train))

        IoU_valid, loss_valid = valid_net(validloader, ep, net, criterion, writer, device)

        writer.add_scalars('valid/Loss', {"valid": loss_valid}, ep + 1)
        writer.add_scalars('valid/IoU', {'valid': IoU_valid}, ep+1)
        logging.info('Valid Epoch [{}/{}] [Loss: {:.4f}] [IoU: {:.4f}]'.format(ep+1, args.n_epochs, loss_valid, IoU_valid))

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

        # remember best prec@1 and save checkpoint
        is_best = loss_valid < min_loss
        min_loss = min(loss_valid, min_loss)
        scheduler.step(loss_valid)  # reschedule learning rate for ReduceLROnPlateau
        save_checkpoint({
            'epoch': ep + 1,
            'net_state_dict': net.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'min_loss': min_loss,
        }, is_best, root_dir=model_dir)

    logging.info('Training Done....')
    print('Processing time:', time.time() - start_time)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
