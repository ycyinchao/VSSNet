import numpy as np
import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from data.dataloader import get_loader, test_dataset
from models.detectors.detector_woMSA import PSCCANet_woMSA
from utils.utils import adjust_lr
import torch.nn.functional as F
import logging
import eval.metrics as metrics

from models.detectors.detector import PSCCANet



def load_model_state_dict(model, state_dict, print_matched=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_matched:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


def weighted_bce_and_iou_loss(pred, mask):
    # kernelSize3
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask,weight=weit, reduce='none')

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def val(test_loader,model, epoch, save_path):

    global best_metric_dict, best_score, best_epoch
    SM = metrics.Smeasure()
    WFM = metrics.WeightedFmeasure()
    mae = metrics.MAE()
    EM = metrics.Emeasure()
    metrics_dict = dict()

    model.eval()
    with torch.no_grad():

        for i in range(test_loader.size):
            image, gt, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            # gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, res1 = model(image)
            # eval Dice
            res = F.upsample(res[-1] + res1, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = res*255
            # mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
            SM.step(pred=res, gt=gt)
            WFM.step(pred=res, gt=gt)
            mae.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)
        metrics_dict.update(Sm=SM.get_results()['sm'].round(3))
        metrics_dict.update(weightFm=WFM.get_results()['wfm'].round(3))
        metrics_dict.update(MAE=mae.get_results()['mae'].round(5))
        metrics_dict.update(meanEm=EM.get_results()['em']['curve'].mean().round(3))

        cur_score = ((metrics_dict['Sm'] + metrics_dict['weightFm']+ metrics_dict['meanEm']))

        # mae = mae_sum / test_loader.size
        # writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)

        if epoch == 1:
            best_score = cur_score
            best_metric_dict = metrics_dict
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'best.pth')
                print('>>>Save state_dict successfully! Best epoch:{}.'.format(best_epoch))
        print(
            '[Cur Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})\n[Best Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})'.format(
                epoch, metrics_dict['Sm'], metrics_dict['weightFm'], metrics_dict['MAE'], metrics_dict['meanEm'],
                best_epoch, best_metric_dict['Sm'], best_metric_dict['weightFm'], best_metric_dict['MAE'], best_metric_dict['meanEm']))
        logging.info(
            '[Cur Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})\n[Best Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})'.format(
                epoch, metrics_dict['Sm'], metrics_dict['weightFm'], metrics_dict['MAE'], metrics_dict['meanEm'],
                best_epoch, best_metric_dict['Sm'], best_metric_dict['weightFm'], best_metric_dict['MAE'], best_metric_dict['meanEm']))


def train(train_loader, model, optimizer, epoch):
    model.train()

    for i, pack in enumerate(train_loader, start=1):

        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        # ---- forward ----
        result1, result2 = model(images)
        # ---- loss function ----
        losses = [weighted_bce_and_iou_loss(out, gts) for out in result1]
        loss_result1 = 0
        gamma = 0.2
        for it in range(1, len(result1) + 1):
            loss_result1 += (gamma * it) * losses[it - 1]
        loss_result2 = weighted_bce_and_iou_loss(result2, gts)
        loss = loss_result1 + loss_result2
        # ---- backward ----
        loss.backward()
        optimizer.step()

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss1: {:0.4f}, loss2: {:0.4f}, loss_total: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_result1, loss_result2, loss))
            logging.info(
                'Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss1: {:0.4f}, loss2: {:0.4f}, loss_total: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss_result1, loss_result2, loss))
    # save model
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), opt.save_path + str(epoch) + '.pth')


if __name__ == '__main__':

    # setup_seed()

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=70, help='epoch number')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation', default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size,candidate=352,704,1056')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str, default='../Dataset/TrainDataset',
                        help='path to train dataset')
    parser.add_argument('--test_path', type=str, default='../Dataset/TestDataset',
                        help='path to testing dataset')
    parser.add_argument('--save_path', type=str, default='./checkpoints/' + 'PSCCANet' + '/')
    parser.add_argument('--epoch_save', type=int, default=10, help='every n epochs to save model')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    logging.basicConfig(filename=opt.save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    print(">>>:{}".format(opt))
    logging.info(">>>:{}".format(opt))

    # ---- build models ----
    # torch.cuda.set_device(0)
    model = PSCCANet().cuda()
    if opt.load is not None:
        model_dict = torch.load(opt.load)
        print("start loading model")
        load_model_state_dict(model, model_dict)
        print('Sucefully load model from:', opt.load)

    print('model paramters', sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info("model paramters:" + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)  # weight_decay可以设置为5e-2看看情况
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    logging.info("optimizer:" + str(optimizer))

    image_root = '{}/Imgs/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    val_loader = test_dataset(image_root=opt.test_path + '/COD10K/Image/',
                              gt_root=opt.test_path + '/COD10K/GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # writer = SummaryWriter(opt.save_path + 'summary')

    best_epoch = 0
    best_score = 0
    best_metric_dict = {}

    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, epoch, opt.decay_rate, opt.decay_epoch)  # TODO 调整学习率可以用cosine schedule
        train(train_loader, model, optimizer, epoch)
        if epoch % 2==0:
            val_loader.index = 0
            val(val_loader,model, epoch, opt.save_path)


