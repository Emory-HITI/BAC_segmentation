import os
import csv
import argparse
import numpy as np

import torch
import torch.nn as nn

from SCUNet import SCUNet
from utils import *


# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--ckptpath', type=str,
                    default="./SCUNet_dice_512_512_best.pt", 
                    help='path to checkpoint')
parser.add_argument('--datapath', type=str, default='/home/jupyter-anbhimi/bac_16bits/calcium_mammo_pngs/',
                    help='path to the image folder that contains mammogram data ending with .png ')
parser.add_argument('--temppath', type=str, default='/data/BAC/Sophie/Code/Annotation/bac_16bits_preds/', 
                    help='a temp folder to store intermediate results, clean results will be left after the code running done')
parser.add_argument('--evalpath', type=str, default='/data/BAC/Sophie/Code/Annotation/bac_16bits_preds/evaluation_all.csv', 
                    help='path to a csv file to save all the statical results collected during evaluation')
parser.add_argument('--prob_thre', type=float, default=0.65, 
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
    temppath = args.temppath    
    evalpath = args.evalpath
    
    imgnames = [x for x in os.listdir(datapath) if x.endswith(".png")]
    with open(evalpath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["imgpath", "breastarea", "prob", "area_thre_"+str(args.prob_thre), "sum_intensity", "sum_intensity_100", "sum_pixels_100", "imgsize"])
    for  imgname in imgnames:
        print("Processing: ",imgname)
        breastarea, pm, am, sim, tamx, tsimx, imgsize = predict(net, datapath, temppath, imgname, args.prob_thre)
        with open(evalpath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([imgname, breastarea, pm, am, sim, tamx, tsimx, imgsize])

        
if __name__ == '__main__':
    main()
