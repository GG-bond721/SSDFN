import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

from config.args import test_options
from config.config import model_config
from utils.dataset_multispectral import MultispectralImageFolder
from torch.utils.data import DataLoader
from PIL import ImageFile, Image
from SSDFN import SSDFN
from utils.TESTING_save_nir_npy import test_model_multispectral
from utils.logger import setup_logger
import faulthandler

faulthandler.enable()


def main():
    torch.backends.cudnn.deterministic = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options()
    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if not os.path.exists(os.path.join('./experiments', args.experiment)):
        os.makedirs(os.path.join('./experiments', args.experiment))
    setup_logger('test', os.path.join('./experiments', args.experiment),
                 'test_' + args.experiment, level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    NUM_CHANNELS = 7
    METHOD_NAME = "ours"
    TARGET_BPP = None


    test_dataset = MultispectralImageFolder(
        args.dataset,
        split="test",
        transform=None,
        num_channels=NUM_CHANNELS
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    logger_test.info(f"Dataset: {args.dataset}")
    logger_test.info(f"Test images: {len(test_dataset)}")
    logger_test.info(f"Channels: {NUM_CHANNELS}")


    net = SSDFN(in_channels=NUM_CHANNELS,out_channels=NUM_CHANNELS,latent_channels=320,hyper_channels=640,slice_ch=[16, 16, 32, 64, 192],quant='ste')
    net = net.to(device)

    if args.checkpoint is None:
        raise ValueError("Checkpoint path is required for testing!")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    net.update(force=True)
    epoch = checkpoint["epoch"]

    logger_test.info(f"Loaded checkpoint from epoch {epoch}")
    logger_test.info(f"Start testing!")

    save_dir = os.path.join('./experiments', args.experiment, 'test_results', '%03d' % epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_model_multispectral(
        net=net,
        test_dataloader=test_dataloader,
        logger_test=logger_test,
        save_dir=save_dir,
        epoch=epoch,
        num_channels=NUM_CHANNELS,
        method_name=METHOD_NAME,
        target_bpp=TARGET_BPP,
        metric_data_range=65535.0,
        save_full_recon_npy=True,
        save_full_original_npy=False
    )

    logger_test.info("Testing completed!")


if __name__ == '__main__':
    main()