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

from SCUNet import SCUNet

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


#generate patches
def preprocess_img(img_path):
    im = cv2.imread(img_path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    figure_size = 5 # the dimension of the x and y axis of the kernal.
    new_image = cv2.medianBlur(imgray,figure_size)
    return new_image

 
def get_slices(n): # help to determine where to start splitting patches
    slices = []
    for i in range(0, n, 480): 
        if i + Image_Size[0] >= n:
            slices.append(n-Image_Size[0])
            break
        slices.append(i)
    return slices
    
    
def crop_img(img, name, size, savepath): # crop mammogram into patches and save them to a temp place
    x, y = img.shape
    Xs = get_slices(x)
    Ys = get_slices(y)

    save_ext  = ".png"
    cnt = 0
    for i in Xs:
        for j in Ys:
            cropped_image = img[i:i+size,j:j+size]
            if np.mean(cropped_image) <= 20 or np.std(cropped_image) <= 10:
                continue
            cnt+=1           
            save_imgname = savepath+'/'+name[:-4]+"_"+str(i)+"_"+str(j)+save_ext
            cv2.imwrite(save_imgname, np.array(cropped_image))

def generate_patches(datapath, temppath, imgname):        
    if not os.path.exists(os.path.join(temppath, imgname[:-4])):
        os.makedirs(os.path.join(temppath, imgname[:-4]))
    patchpath = os.path.join(temppath, imgname[:-4])
    img = preprocess_img(datapath+imgname)
    crop_img(img,imgname,Image_Size[0],patchpath)
    return patchpath, img
        
tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.2843, 0.1712)])

class MammoDataset(Dataset):
    def __init__(self, rootdir, img_transform = None):
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

def evaluate(img, imgname, wholeimg_rootdir, prob_thre):
    # area, probability, threshold, intensity>100
    breastarea = np.sum((img>0).astype(int))
    pre_mask = np.load(wholeimg_rootdir+'/'+imgname[:-4]+"_premask.npy")
    prob = np.sum(pre_mask)
    pre_mask_thre = (pre_mask>0.5).astype(int)
    area_thre = np.sum(pre_mask_thre)
    pre_image = img[np.where(pre_mask>prob_thre)]
    pre_image2 = pre_image[np.where(pre_image >= 100)]
    sum_intensity = sum(pre_image)
    sum_intensity_100 = sum(pre_image2)
    sum_pixels_100 = sum(pre_image >= 100)
    imgsize = img.shape
    print("breastarea:", breastarea, " prob:", prob, " area_0.5:", area_thre, " sum_intensity:", sum_intensity, " sum_intensity_100:", sum_intensity_100, " area_100:", sum_pixels_100, " image.size:", imgsize)
    return breastarea, prob, area_thre, sum_intensity, sum_intensity_100, sum_pixels_100, imgsize
    
    
def predict(net, datapath, temppath, imgname, prob_thre): 
    testdir, image = generate_patches(datapath, temppath, imgname)
    testset = MammoDataset(rootdir=testdir, img_transform = tfms)
    testloader = DataLoader(testset, batch_size=8, shuffle=False, pin_memory=torch.cuda.is_available(),num_workers=8) 
    wholeimg_rootdir = testdir+"whole"
#     print("wholeimg_rootdir: ",wholeimg_rootdir)
    if os.path.exists(wholeimg_rootdir):
        return evaluate(image, imgname, wholeimg_rootdir, prob_thre)
    patch_pre_mask_dir = testdir+"patch"
#     print("patch_pre_mask_dir: ",patch_pre_mask_dir)
    with torch.no_grad():
        for batch_idx, (img, imgnames) in enumerate(testloader):
            img = img.type(torch.DoubleTensor)
            pre_mask = net(img)
            pmask = pre_mask.data.cpu().numpy()    
            for i in range(pmask.shape[0]):
                name = imgnames[i]
                if not os.path.exists(patch_pre_mask_dir):
                    os.makedirs(patch_pre_mask_dir)
                np.save(patch_pre_mask_dir+'/'+name[:-4]+".npy", pmask[i][0])
            
    print("Collect.....")            
               
    premask_names = os.listdir(patch_pre_mask_dir)
    pre_mask = np.zeros(image.shape)
    patchsize = 512
    for prename in premask_names:
        _, x, y = prename[:-4].split("_")
        x = int(x)
        y = int(y)
        patch_mask = np.load(patch_pre_mask_dir+'/'+prename)
        for i in range(x, x+patchsize):
            for j in range(y, y+patchsize):
                pre_mask[i][j] = max(pre_mask[i][j], patch_mask[i-x][j-y])
    if not os.path.exists(wholeimg_rootdir):
        os.makedirs(wholeimg_rootdir)
    np.save(wholeimg_rootdir+'/'+imgname[:-4]+"_premask.npy",pre_mask)
    if os.path.exists(testdir):
        shutil.rmtree(testdir) 
    if os.path.exists(patch_pre_mask_dir):
        shutil.rmtree(patch_pre_mask_dir)
    return evaluate(image, imgname, wholeimg_rootdir, prob_thre)

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
