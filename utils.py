import os
import cv2
import shutil
import argparse
import numpy as np
from os import listdir

import torch
from torch.utils.data import DataLoader
from dataset import MammoDataset

Image_Size = [512, 512]

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
    testset = MammoDataset(rootdir=testdir)
    testloader = DataLoader(testset, batch_size=8, shuffle=False, pin_memory=torch.cuda.is_available(),num_workers=8) 
    wholeimg_rootdir = testdir+"whole"
    print("wholeimg_rootdir: ",wholeimg_rootdir)
    if os.path.exists(wholeimg_rootdir):
        return evaluate(image, imgname, wholeimg_rootdir, prob_thre)
    patch_pre_mask_dir = testdir+"patch"
    print("patch_pre_mask_dir: ",patch_pre_mask_dir)
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
        print(x, y)
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
