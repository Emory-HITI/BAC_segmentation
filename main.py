import os
import cv2
import numpy as np
import collections
from os import listdir
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils import dice_loss, evaluate
from model.SC-UNet import SCG-Net
from PIL import Image, ImageEnhance
from dataset import MammoDataset_Normalized as MammoDataset
Image_Size = [512, 512]


def train(net, epochs, trainloader , valloader, optimizer, is_gpu, bestdice, Netname):
    print("==> Start training...")
    net.to(torch.double)
    if is_gpu:
        net.cuda()
    best_loss = 1e8
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        Recall = 0
        Precision = 0
        Accuracy = 0
        F_score = 0
        Jaccard = 0
        Dice = 0
        for batch_idx, (img, mask) in enumerate(trainloader):
            if is_gpu:
                img, mask = img.cuda(), mask.cuda()
            img, mask = Variable(img), Variable(mask)
            
            pre_mask = net(img)            
            loss = dice_loss(mask, pre_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            recall, precision, accuracy, f_score, jaccard, dice = evaluate((mask>=0.5).type(torch.ByteTensor), (pre_mask>=0.5).type(torch.ByteTensor))
            Recall += recall.item()
            Precision += precision.item()
            Accuracy += accuracy.item()
            F_score += f_score.item()
            Jaccard += jaccard.item()
            Dice += dice.item()

        running_loss /= len(trainloader)
        print("Train: recall: %.5f, precision: %.5f, accuracy: %.5f, f_score: %.5f, jaccard: %.5f,  dice: %.5f" % ( Recall/len(trainloader), Precision/len(trainloader), Accuracy/len(trainloader), F_score/len(trainloader), Jaccard/len(trainloader), Dice/len(trainloader)))   

        net.eval()
        val_loss = 0.0
        Recall = 0
        Precision = 0
        Accuracy = 0
        F_score = 0
        Jaccard = 0
        Dice = 0
        with torch.no_grad():
            for batch_idx, (img, mask) in enumerate(valloader):
                if is_gpu:
                    img, mask = img.cuda(), mask.cuda()
                img, mask = Variable(img), Variable(mask)
                pre_mask = net(img)
                loss = dice_loss(mask, pre_mask) 
           
                val_loss += loss.item()
                recall, precision, accuracy, f_score, jaccard, dice = evaluate((mask>=0.5).type(torch.ByteTensor), (pre_mask>=0.5).type(torch.ByteTensor))
                Recall += recall.item()
                Precision += precision.item()
                Accuracy += accuracy.item()
                F_score += f_score.item()
                Jaccard += jaccard.item()
                Dice += dice.item()
            val_loss /= len(valloader)
            print("Val: recall: %.5f, precision: %.5f, accuracy: %.5f, f_score: %.5f, jaccard: %.5f,  dice: %.5f" % (Recall/len(valloader), Precision/len(valloader), Accuracy/len(valloader), F_score/len(valloader), Jaccard/len(valloader), Dice/len(valloader)))    

        if Dice/len(valloader) > bestdice:
            bestdice = Dice/len(valloader)
            save_path = "./{}_dice_{}_{}_best.pt".format(Netname, Image_Size[0],Image_Size[1])
            torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'trainloss': running_loss, 'valloss': val_loss, 'valdice': Dice/len(valloader)},save_path )
        print("Training Epoch:{0} | TrainLoss{1} | ValLoss{2}".format(epoch+1, running_loss, val_loss))
      
        
def main():
    Net = "SCU-Net"
    train_tfms = transforms.Compose([
                                     transforms.ColorJitter(brightness=0.2, saturation=0.3, hue=0.25),
                                     transforms.ToTensor(),
                                     transforms.Normalize(0.2843, 0.1712)
                                    ])
    val_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.2843, 0.1712)])
    mask_tfms = transforms.Compose([transforms.ToTensor()])
    valmask_tfms = transforms.Compose([transforms.ToTensor()])

    epochs = 60
    batch_size = 8*4
    num_workers = 6

    rootdir = "./PatchData/"
    trainlist = np.load("./patch_train.npy")
    vallist = np.load("./patch_val.npy") 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device {}".format(device))
    if str(device) == 'cuda':
        is_gpu = True
    else:
        is_gpu = False 
    net = SCU-Net(classes=1)
    net = net.double()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)    
    net.to(device)

    trainset = MammoDataset( rootdir, "Train", trainlist, img_transform = train_tfms, mask_transform = mask_tfms)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=num_workers)
    valset = MammoDataset( rootdir, "Val", vallist, img_transform = val_tfms, mask_transform = valmask_tfms)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=num_workers)
    bestdice = 0        
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    train(net, epochs, trainloader, valloader, optimizer, is_gpu, bestdice, Net)


main()
    
