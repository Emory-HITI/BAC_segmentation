# BAC_segmentation

This repository contains PyTorch implementation for model evaluation of the following paper: SCU-Net: A deep learning method for segmentation and quantification of breast arterial calcifications on mammograms (https://pubmed.ncbi.nlm.nih.gov/34328661/, https://www.medrxiv.org/content/10.1101/2021.07.30.21261406v1.full.pdf).

## Installation
1. First clone the repository
   ```
   git clone https://github.com/Emory-HITI/BAC_segmentation.git
   ```
2. Create the virtual environment via conda
    ```
    conda create -n scunet python=3.7 
    ```
3. Activate the virtual environment.
    ```
    conda activate scunet
    ```
3. Install the dependencies.
   ```
   pip install --user --requirement requirements.txt
   ```
## Model weights

SCUNet_dice_512_512_best_nosigmoid.pt is the model that proposed and evluated in the paper and works the best for 8-bit mammogram images with BAC. 

SCUNet_dice_512_512_best.pt is the model that works the best for 16-bit mammogram images with BAC. 

SCU_Net_512_512_best_BCE_pos_all.pt is the model that works the best for mammogram images with  and without BAC. 

## Evaluation
To run the model on your own mammogram datasets, run the following commands:

``` shell
python AutoSeg.py --ckptpath path/to/checkpoint --datapath path/to/mammogram_png/folder --temppath path/to/temp_folder_for_intermediate_results  --evalpath path/to/folder_for_saving_csv_results --prob_thre probability_threshold
``` 
Please look into the argument part of AutoSeg.py to learn more details for runnning the code

## Cite SCU-Net
If you use this repository or would like to refer the paper, please use the following BibTeX entry
```
@article{guo2021scu,
  title={SCU-Net: A deep learning method for segmentation and quantification of breast arterial calcifications on mammograms},
  author={Guo, Xiaoyuan and O'Neill, W Charles and Vey, Brianna and Yang, Tianen Christopher and Kim, Thomas J and Ghassemi, Maryzeh and Pan, Ian and Gichoya, Judy Wawira and Trivedi, Hari and Banerjee, Imon},
  journal={Medical physics},
  volume={48},
  number={10},
  pages={5851--5861},
  year={2021},
  publisher={Wiley Online Library}
}
```
