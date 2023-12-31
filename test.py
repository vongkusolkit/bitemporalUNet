import argparse
import logging
import os
import time
import numpy as np
import rasterio
import torch
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.special import softmax
import matplotlib.pyplot as plt

import utils.dataproc as dp
from model.dualbridgeunet_v2 import DualBridgeUNet
from utils.dicebceloss import DiceBCELoss
from utils.utils import AverageMeter, set_logger, iou_pytorch, metric_pytorch, weight_tensor

parser = argparse.ArgumentParser()
# parser.add_argument('--version', type=str, default='train_harvey_framework')
parser.add_argument('--version', type=str, default='train_florence')
parser.add_argument('-t', '--test_batch_size', type=int, default=4)
parser.add_argument('--csv_test', type=str, default='data/florence/florence_test/p1.csv')
parser.add_argument('--data_root_dir_pre', type=str, default='data/florence/florence_test/pre')
parser.add_argument('--data_root_dir_post', type=str, default='data/florence/florence_test/post')
parser.add_argument('--data_root_dir_gt', type=str, default='data/florence/florence_test/gt')
# parser.add_argument('--csv_test', type=str, default='/home/ai4sg/jamp/bitemporalUNet/data/harvey/harvey_test/v14.csv')
# parser.add_argument('--data_root_dir_pre', type=str, default='/home/ai4sg/jamp/harvey_test/pre')
# parser.add_argument('--data_root_dir_post', type=str, default='/home/ai4sg/jamp/harvey_test/post')
# parser.add_argument('--data_root_dir_gt', type=str, default='/home/ai4sg/jamp/harvey_test/label')
parser.add_argument('--weight_non_flood', type=float, default=0.097)
parser.add_argument('--weight_flood', type=float, default=0.903)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def test_net(dataloader, net, criterion, output_dir, device=device):
    net.eval()
    print("Testing...")
    epoch_loss = AverageMeter()
    n_batches = len(dataloader)
    predict_list = []
    labels_list = []   # Store labels for AUC and ROC calculation
    for i, batched_data in enumerate(dataloader):
        pixel_pre = batched_data['pixel_pre'].to(device=device, dtype=torch.float32)
        pixel_post = batched_data['pixel_post'].to(device=device, dtype=torch.float32)
        label = batched_data['label'].to(device=device, dtype=torch.float32)
        filenames = batched_data['filename']
        batch_size = pixel_pre.shape[0]
        output = net(pixel_pre, pixel_post)
        predict = output.squeeze(dim=1)
        loss = criterion(predict, label)
        epoch_loss.update(loss.item(), batch_size)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        for j in range(batch_size):
            # filename = filenames[j]
            # predict_filename = os.path.join(output_dir, f'{filename}.tif')
            # label_filename = os.path.join(output_dir, f'{filename}_gt.tif')
            predict_image = predict[j].detach().cpu().numpy()
            predict_image = np.squeeze(predict_image)
            label_image = label[j].detach().cpu().numpy()
            label_image = np.squeeze(label_image)

            predict_list.extend(predict_image.flatten())
            labels_list.extend(label_image.flatten())
            
    predict_array = np.array(predict_list).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(predict_array)
    predict_binarized = kmeans.labels_
    # # if score is low, add "1 - " because could be evaluating the wrong class
    # predict_binarized = 1 - kmeans.labels_ 

    labels_binary = np.where(np.array(labels_list) > 0.5, 1, 0)

    auc = roc_auc_score(labels_binary, predict_binarized)
    fpr, tpr, thresholds = roc_curve(labels_binary, predict_binarized)
    
    return epoch_loss.avg, auc, fpr, tpr   
            
            
            
def main(args):
    model_dir = "./logs/models_{}".format(args.version)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    set_logger(os.path.join(model_dir, 'test.log'))
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

    loss_test, auc, fpr, tpr = test_net(testloader, net, criterion, output_dir=f"output/{args.version}")

    logging.info(
        'Test [Loss: {}]'.format(loss_test))
    print('AUC: ', auc)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Florence F1 ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('florence_f1_roc_curve.png')

    print('Processing time:', time.time() - start_time)


if __name__ == '__main__':
    start_time = time.time()
    args = parser.parse_args()
    main(args)
    print('Processing time:', time.time() - start_time)

