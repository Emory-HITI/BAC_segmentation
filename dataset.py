import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import cv2
import numpy as np
from os.path import splitext
import random
from os import listdir
from glob import glob
import logging
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import collections
import copy
from PIL import Image, ImageEnhance
Image_Size = [512, 512]


class MammoDataset_Normalized(Dataset):

    def __init__(self, rootdir, phase, namelists, img_transform = None, mask_transform = None):
        self.rootdir = rootdir
        self.phase = phase
        self.namelists = namelists
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.namelists)
    
    
    def __getitem__(self, idx):
        fname = self.namelists[idx]
        fpath = os.path.join(self.rootdir, self.phase, fname[:-9]+".png")
        img = Image.open(fpath)
        mpath = os.path.join(self.rootdir, self.phase, fname)
        mask = Image.open(mpath)
        if self.phase == "Train":
            seed = np.random.randint(0, 5)
            if seed == 0:
                img = img.transpose(method = Image.ROTATE_90)
                mask = mask.transpose(method = Image.ROTATE_90)
            elif seed == 1:
                img = img.transpose(method = Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(method = Image.FLIP_LEFT_RIGHT)
            elif seed == 2:
                img = img.transpose(method = Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(method = Image.FLIP_TOP_BOTTOM)
            elif seed == 3:
                img = img.transpose(method = Image.ROTATE_270)
                mask = mask.transpose(method = Image.ROTATE_270)

        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)    
        mask = np.array(mask, dtype=np.float64)
        if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=2)
                mask = mask.transpose((2, 0, 1))
                if mask.max() > 1:
                    mask  = mask / 255.0  
        
        
        img, mask = torch.from_numpy(np.array(img, dtype=np.float64)), torch.from_numpy(mask)
        image = torch.stack([img,img,img], axis = 1)

        return torch.squeeze(image, dim = 0).type(torch.DoubleTensor), mask

