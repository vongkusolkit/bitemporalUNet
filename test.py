import argparse
import logging
import os
import time
import rasterio
import torch
from torchvision import transforms

import dataproc as dp
from dualbridgeunet_v2 import DualBridgeUNet
from dicebceloss import DiceBCELoss
from utils import AverageMeter, set_logger, iou_pytorch, metric_pytorch, weight_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='train_harvey_framework')
parser.add_argument('-t', '--test_batch_size', type=int, default=12)
parser.add_argument('--csv_test', type=str, default='../harvey_eval_site_v2_floodmask/harvey_test/index.csv')
parser.add_argument('--data_root_dir_pre', type=str, default='../harvey_eval_site_v2_floodmask/harvey_test/pre')
parser.add_argument('--data_root_dir_post', type=str, default='../harvey_eval_site_v2_floodmask/harvey_test/post')
parser.add_argument('--data_root_dir_gt', type=str, default='../harvey_eval_site_v2_floodmask/harvey_test/label')
parser.add_argument('--weight_non_flood', type=float, default=0.097)
parser.add_argument('--weight_flood', type=float, default=0.903)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def test_net(dataloader, net, criterion, device=device, output_dir='./output/'):
    net.eval()
    print("Testing...")

    epoch_loss = AverageMeter()
    n_batches = len(dataloader)

    for i, batched_data in enumerate(dataloader):
        pixel_pre = batched_data['pixel_pre'].to(device=device, dtype=torch.float32)  # [Batch, 4, h, w]
        pixel_post = batched_data['pixel_post'].to(device=device, dtype=torch.float32)
        label = batched_data['label'].to(device=device, dtype=torch.float32)
        filenames = batched_data['filename']
        batch_size = pixel_pre.shape[0]

        output = net(pixel_pre, pixel_post)
        predict = output.squeeze(dim=1)

        loss = criterion(predict, label)
        epoch_loss.update(loss.item(), batch_size)

        # save predictions and labels
        for j in range(batch_size):
            filename = filenames[j]
            predict_filename = os.path.join(output_dir, f'{filename}.tif')
            label_filename = os.path.join(output_dir, f'{filename}_gt.tif')

            predict_image = predict[j].detach().cpu().numpy()
            label_image = label[j].detach().cpu().numpy()

            with rasterio.open(predict_filename, 'w', driver='GTiff', height=predict_image.shape[1],
                               width=predict_image.shape[2], count=1, dtype=predict_image.dtype) as dst:
                dst.write(predict_image, 1)

            with rasterio.open(label_filename, 'w', driver='GTiff', height=label_image.shape[1],
                               width=label_image.shape[2], count=1, dtype=label_image.dtype) as dst:
                dst.write(label_image, 1)

    return epoch_loss.avg

def main(args):
    model_dir = "./logs/models_{}".format(args.version)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    set_logger(os.path.join(model_dir, 'test.log'))

    # valid set, test time augmentation (TTA)
    transform_test = transforms.Compose([
        dp.ToTensor()
    ])
    testset = dp.PixelDataset(csv_file=args.csv_test,
                              root_dir_pre=args.data_root_dir_pre, root_dir_post=args.data_root_dir_post,
                              root_dir_gt=args.data_root_dir_gt,
                              is_labeled=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    net = DualBridgeUNet().to(device)
    checkpoint = torch.load("{}/model_best.pth.tar".format(model_dir), map_location=device)
    start_epoch = checkpoint['epoch']
    ae_min_loss = checkpoint['min_loss']
    net.load_state_dict(checkpoint['net_state_dict'])

    criterion = DiceBCELoss()

    loss_test = test_net(testloader, net, criterion, device)

    logging.info(
        'Test [Loss: {}]'.format(loss_test)
    print('Processing time:', time.time() - start_time)


if __name__ == '__main__':
    start_time = time.time()  # staring time for training, unit: second
    args = parser.parse_args()
    main(args)
    print('Processing time:', time.time() - start_time)
