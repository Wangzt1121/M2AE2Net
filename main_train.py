import os
import sys
import time
import torch.nn as nn

import torch
import torch.optim as optim
from tqdm import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter
from config import get_option, map_dict

from utils.metrics import torch_psnr
from dataset import get_train_loader, get_val_loader
from utils.model_utils import find_checkpoint, save_models_v2
from utils.dataset_utils import split_image_256_to_128, splice_image_128_to_256
from utils.dataset_utils import MixUp_AUG
from utils.scheduler import GradualWarmupScheduler
from loss import charbonnier_loss
from utils.logger_utils import Logger

sys.path.append("models")
import torch.backends.cudnn as cudnn
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
from ablation import Network, fixed_loss

#torch.set_default_dtype(torch.float16)
def mask_layers(image, l):  # 图片，输入通道，l为输入分层数
    k = 256 // l
    b, c, h, w = image.shape
    img_layers = torch.zeros((b, l, h, w)).cuda()  # 加快运行提前分配出了空间
    for i in range(l):
        count = i*k
        img_temp = image.clone()
        if i == 0:
                index = (img_temp < i + k + count)
        elif i == l-1:
                index = (i + count <= img_temp)
        else:
                index = (i + count <= img_temp) * (img_temp < i + k + count)
        img_temp[~index] = 0
        img_layers[:, i:i+1, :, :] = img_temp
    img_layers = torch.tensor(img_layers).cuda()
    return img_layers

def layer_dissamble(image, l):
    n, c, h, w = image.shape
    img_layers = torch.zeros([n, c*l, h, w]).cuda()
    for i in range(c):
        img_layers[:,i*l:(i+1)*l,:,:] = mask_layers(image[:,i:i+1,:,:], l)
    return img_layers


def eval_step(model, val_loader, val_len, criterion, device='cuda'):
    cudnn.benchmark = False
    model.eval()
    total_loss, total_psnr = 0, 0
    # iteration
    for _, batch_data in enumerate(val_loader):
        batch_size = batch_data[0].size(0)
        noisy_patch, clean_patch = batch_data[0].to(device), batch_data[1].to(device)
        #layer_patch = layer_dissamble(noisy_patch, 4)
        with torch.no_grad():
            # inference
            repair_patch = model(noisy_patch)
            batch_loss = criterion(clean_patch, repair_patch)
            # update eval loss
            total_loss += batch_loss * batch_size
            # calculate psnr and loss of eval dataset)
            repair_patch = torch.clamp(repair_patch, min=0., max=1.)
            target_patch = torch.clamp(clean_patch, min=0., max=1.)
            # calculate psn。r
            psnr_vec = torch_psnr(repair_patch, target_patch)
            total_psnr += torch.sum(psnr_vec)

    avg_psnr = total_psnr / val_len
    avg_loss = total_loss / val_len
    return avg_loss, avg_psnr

def train_pipeline(cfg):
    cudnn.benchmark = True
    # 创建保存文件夹
    if not os.path.exists(cfg['pth_dir']):
        os.makedirs(cfg['pth_dir'])

    # 设置打印日志
    logger = Logger(cfg['pth_dir'] + '/' + cfg['arch'] + '.txt', is_w=True)
    writer = SummaryWriter(log_dir = os.path.join(cfg['log_dir'], cfg['arch']))

    # time
    now_time = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    logger.write("Training start time is: %s." % now_time)
    logger.write("1). Initilization, define network and data loader, Waiting ....")

    # 设置环境
    device = torch.device(cfg["device"])
    model = Network().cuda()
    logger.write("\t[arch name]: " + cfg['arch'])

    # 定义优化器和学习率
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr_init'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['n_epochs']-cfg['n_warmup'], eta_min=cfg['lr_min'])
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg['n_warmup'], after_scheduler=scheduler)

    # 加载训练数据和验证数据

    train_loader, train_len = get_train_loader(data_dir1=cfg['data_dir1'], data_dir2=cfg['data_dir2'], batch_size=cfg['batch_size'], num_workers=8, shuffle=True, img_size=cfg['img_size'])

    val_loader, val_len = get_val_loader(data_dir=cfg['data_dir1'], batch_size=cfg['batch_size'],
                                         num_workers=2, shuffle=False, img_size=256)

    logger.write("\t[datasets]: train length: %d, validate length: %d." % (train_len, val_len))

    # 训练参数
    last_epoch = 0
    best_psnr  = 0.

    # 是否接续断点训练，加载checkpoint文件
    if cfg['is_resume'] is True:
        checkpoint = find_checkpoint(pth_dir=cfg['pth_dir'], device=device, load_tag=cfg['resume_tag'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        last_epoch = checkpoint['epoch']
        best_psnr = checkpoint['avg_psnr'].data
        print('加载checkpoint文件', 'best_psnr =', best_psnr)

    # auto resume
    for _ in range(0, last_epoch + 1):
        scheduler_warmup.step()

    criterion = fixed_loss
    verbose_step = len(train_loader) // 4

    # augment
    mixup_aug = MixUp_AUG()

    # training iteration
    logger.write("2). Training iteration:")
    for epoch in range(last_epoch + 1, cfg['n_epochs'] + 1):
        logger.write("epoch: %3d, lr: %.7f." % (epoch, optimizer.param_groups[0]['lr']))
        logger.write("=====================================")

        now_train_loss, now_iter_size = 0, 0
        train_tbar = tqdm(train_loader)

        model.train()
        for n_count, batch_data in enumerate(train_tbar):

            batch_size = batch_data[0].size(0)
            noisy_patch, clean_patch = batch_data[0].to(device), batch_data[1].to(device)

            if epoch > 5:
                noisy_patch, clean_patch = mixup_aug.aug(noisy_patch, clean_patch)

            #layer_patch = layer_dissamble(noisy_patch, 4)

            # forward and backward
            optimizer.zero_grad()
            res = model(noisy_patch)
            batch_loss = criterion(res, clean_patch)

            batch_loss.backward()

            optimizer.step()  # 优化器设置

            # calculate total loss
            now_train_loss += batch_loss * batch_size
            now_iter_size += batch_size
            # update tdqm bar message
            train_tbar.set_postfix(loss=(now_train_loss/now_iter_size).item())
            #optimizer.step()  # 优化器设置---按照warning提示放到了一个for循环之后

            # validate
            if (n_count + 1) % verbose_step == 0:
                avg_loss, avg_psnr = eval_step(model, val_loader, val_len, criterion)

                model.train()
                ver_step = n_count + (epoch - 1) * len(train_loader)
                writer.add_scalar('val/psnr', avg_psnr, ver_step)
                writer.add_scalar('val/loss', avg_loss, ver_step)
                logger.write("\nvalidate dataset psnr val is: %3.5f, average loss is: %3.6f, best psnr: %3.5f."% (avg_psnr, avg_loss, best_psnr))
                # update best result

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    save_models_v2(cfg['pth_dir'], epoch, best_psnr, model, cfg['arch'], optimizer, tag='best')

            if n_count % 40 == 0:
                ver_step = n_count + (epoch - 1) * len(train_loader)
                writer.add_scalar('train/loss', (now_train_loss/now_iter_size).item(), ver_step)


        save_models_v2(cfg['pth_dir'], epoch, best_psnr, model, cfg['arch'], optimizer, tag='last')
        logger.write("Average loss in this epoch is: %3.7f.\n" % (now_train_loss/now_iter_size))
        gc.collect()
        torch.cuda.empty_cache()
        # update learing rate scheduler
        scheduler_warmup.step()
    #optimizer.step()  # 优化器设置---按照warning提示放到了一个for循环之后
    gc.collect()
    torch.cuda.empty_cache()
    # log time
    now_time = time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime())
    logger.write("End Time Is: %s." % now_time)


if __name__ == "__main__":
    # get argment from command line.

    args = get_option()
    cfg = map_dict(args)
    print(cfg)

    # training process 
    train_pipeline(cfg=cfg)

