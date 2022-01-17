import os
import cv2
import csv
import json
import shutil
import random
import logging
import argparse
import collections
import numpy as np
from glob import glob
from os import listdir
from os.path import splitext
from PIL import Image, ImageEnhance

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from dataset import MammoDataset
from SCUNet import SCUNet
from utils import *

Image_Size = [512, 512]

# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--ckptpath', type=str,
                    default="./SCUNet_dice_512_512_best_nosigmoid.pt", 
                    help='path to checkpoint')
parser.add_argument('--datapath', type=str, default='../mammo-imgs/xxx/',
                    help='path to the image folder that contains mammogram data ending with .png ')
parser.add_argument('--temppath', type=str, default='../temp/', 
                    help='a temp folder to store intermediate results, clean results will be left after the code running done')
parser.add_argument('--evalpath', type=str, default='../temp/evaluation_all.csv', 
                    help='path to a csv file to save all the statical results collected during evaluation')
parser.add_argument('--prob_thre', type=float, default=0.5, 
                    help='probability threshold to determine BAC pixels, >=prob_thre: bac, <prob_thre: normal/background')



# parse the arguments
args = parser.parse_args()

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device {}".format(torch.cuda.device_count()))
    if str(device) == 'cuda':
        is_gpu = True
    else:
        is_gpu = False 
    
    net = SCUNet().double()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)    
    checkpoint = torch.load(args.ckptpath)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.eval()    
    
    datapath = args.datapath
    temppath = args.emppath    
    evalpath = args.evalpath
    
    imgnames = [x for x in os.listdir(datapath) if x.endswith(".png")]
    
    for imgname in imgnames:
        print("Processing: ",imgname)
        breastarea, pm, am, sim, tamx, tsimx, imgsize = predict(net, datapath, temppath, imgname, args.prob_thre)
        with open(evalpath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([imgname, breastarea, pm, am, sim, tamx, tsimx, imgsize])

        
if __name__ == '__main__':
    main()
