import os
import random
import logging
from PIL import ImageFile, Image
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset_multispectral import MultispectralImageFolder
from utils.logger import setup_logger
from utils.utils import save_checkpoint
from utils.optimizers import configure_optimizers
from utils.training import train_one_epoch
from utils.testing_multispectral import test_one_epoch_multispectral
from loss.rd_loss import RateDistortionLoss
from config.args import train_options
from config.config import model_config
from SSDFN import SSDFN


class MultispectralTransform:


    def __init__(self, crop_size=None):
        self.crop_size = crop_size

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        if self.crop_size is not None:
            C, H, W = img.shape
            if H > self.crop_size[0] and W > self.crop_size[1]:
                top = random.randint(0, H - self.crop_size[0])
                left = random.randint(0, W - self.crop_size[1])
                img = img[:, top:top + self.crop_size[0], left:left + self.crop_size[1]]
        return img


def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = train_options()
    config = model_config()

    NUM_CHANNELS = 7

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        random.seed(seed)

    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))

    setup_logger('train', os.path.join('./experiments', args.experiment),
                 'train_' + args.experiment, level=logging.INFO, screen=True, tofile=True)
    setup_logger('val', os.path.join('./experiments', args.experiment),
                 'val_' + args.experiment, level=logging.INFO, screen=True, tofile=True)

    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)

    if not os.path.exists(os.path.join('./experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('./experiments', args.experiment, 'checkpoints'))

    train_transforms = MultispectralTransform(crop_size=args.patch_size)
    test_transforms = MultispectralTransform(crop_size=None)

    train_dataset = MultispectralImageFolder(
        args.dataset,
        split="train",
        transform=train_transforms,
        num_channels=NUM_CHANNELS
    )
    test_dataset = MultispectralImageFolder(
        args.dataset,
        split="test",
        transform=test_transforms,
        num_channels=NUM_CHANNELS
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = SSDFN(in_channels=NUM_CHANNELS,out_channels=NUM_CHANNELS,latent_channels=320,hyper_channels=640,slice_ch=[16, 16, 32, 64, 192],quant='ste')

    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=20 #, verbose=True
    )
    criterion = RateDistortionLoss(lmbda=args.lmbda, metrics=args.metrics)

    if args.checkpoint != None:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint["state_dict"])
        is_fine_tuning = True

        if is_fine_tuning:
            logger_train.info("Fine-tuning mode: Loaded weights only. Resetting optimizer and best_loss.")
            start_epoch = 0
            best_loss = 1e10
            current_step = 0
        else:
            logger_train.info("Resume mode: Restoring full state.")
            optimizer.load_state_dict(checkpoint['optimizer'])
            aux_optimizer.load_state_dict(checkpoint['aux_optimizer'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']
            current_step = start_epoch * math.ceil(len(train_dataloader) / args.batch_size)
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    logger_train.info(f"Training SSDFN with {NUM_CHANNELS}-channel multispectral images")
    logger_train.info(f"Seed: {seed}")
    logger_train.info(args)
    logger_train.info(net)
    logger_train.info(optimizer)
    logger_train.info(aux_optimizer)

    optimizer.param_groups[0]['lr'] = args.learning_rate

    for epoch in range(start_epoch, args.epochs):
        logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        current_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            logger_train,
            tb_logger,
            current_step
        )

        save_dir = os.path.join('./experiments', args.experiment, 'val_images', '%03d' % (epoch + 1))
        loss = test_one_epoch_multispectral(
            epoch,
            test_dataloader,
            net,
            criterion,
            save_dir,
            logger_val,
            tb_logger,
            num_channels=NUM_CHANNELS
        )

        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        net.update(force=True)

        if args.save:
            if (epoch + 1) % 4 == 0 or is_best:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "num_channels": NUM_CHANNELS,
                    },
                    is_best,
                    os.path.join('./experiments', args.experiment, 'checkpoints',
                                 "checkpoint_%03d.pth.tar" % (epoch + 1))
                )
                if is_best:
                    logger_val.info('Best checkpoint saved.')
                else:
                    logger_val.info(f'Checkpoint saved at epoch {epoch + 1}.')


if __name__ == '__main__':
    main()