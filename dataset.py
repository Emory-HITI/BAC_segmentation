
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


Image_Size = [512, 512]


class MammoDataset(Dataset):
    def __init__(self, rootdir, img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.2813, 0.1256)])):
        self.rootdir = rootdir
        self.img_transform = img_transform
        self.namelists = [x for x in os.listdir(rootdir) if x.endswith(".png")]

    def __len__(self):
        return len(self.namelists)   
    
    def __getitem__(self, idx):

        fname = self.namelists[idx]
        fpath = os.path.join(self.rootdir, fname)
        img = Image.open(fpath)
        if self.img_transform is not None:
            img = self.img_transform(img)  
        img = torch.from_numpy(np.array(img, dtype=np.float64))   
        image = torch.stack([img,img,img], axis = 1)
        
        return torch.squeeze(image, dim = 0).type(torch.DoubleTensor), fname
