import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class MultispectralImageFolder(Dataset):

    def __init__(self, root, transform=None, split="train", num_channels=7):
        splitdir = Path(root) / split
        self.split = split
        self.num_channels = num_channels
        self.transform = transform

        self.is_landsat = 'landsat' in str(root).lower()

        print(f"Loading {num_channels}-channel multispectral images from: {splitdir}")

        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file()
                               and (f.suffix == '.npy' or f.suffix in ['.png', '.jpg', '.jpeg'])])

        if len(self.samples) == 0:
            raise RuntimeError(f'No valid files found in "{splitdir}"')

        print(f"Found {len(self.samples)} images")

    def __getitem__(self, index):
        imgpath = self.samples[index]

        if imgpath.suffix == '.npy':
            img = np.load(str(imgpath))  # shape: (H, W, C) or (C, H, W)

            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
                if self.is_landsat:
                    img = img / 65535.0
                else:
                    MAX_PIXEL_VAL = 10000.0
                    img = np.clip(img, 0, MAX_PIXEL_VAL)
                    img = img / MAX_PIXEL_VAL

            if img.ndim == 2:
                img = img[np.newaxis, :, :]
            elif img.ndim == 3:
                if img.shape[-1] == self.num_channels:
                    img = img.transpose(2, 0, 1)
                elif img.shape[0] != self.num_channels:
                    raise ValueError(f"Expected {self.num_channels} channels, got {img.shape}")
        else:
            img = np.array(Image.open(str(imgpath)).convert("RGB"))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.samples)


class ImageFolder(Dataset):

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split
        self.split = split
        print(splitdir)
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        imgname = str(self.samples[index])
        img = np.array(Image.open(imgname).convert("RGB"))

        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)