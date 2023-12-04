# create distance maps 
import numpy as np
import cv2 as cv
import shutil
import math
import os
import glob
import re

from skimage.morphology import dilation
from skimage.morphology import square
from skimage.segmentation import find_boundaries
#create images for train and validation 
#src_train_images = 

INPUT_PATH = os.getcwd()
DATA_PATH = INPUT_PATH

src_train_folder_gt = os.path.join(DATA_PATH, "train_folder_gt/")
src_valid_folder_gt = os.path.join(DATA_PATH, "valid_folder_gt/")
## mask list
trainpath = '{}*.png'.format(src_train_folder_gt)
validpath = '{}*.png'.format(src_valid_folder_gt)
src_train_masks  = [os.path.basename(x) for x in glob.glob(trainpath)]
src_valid_masks  = [os.path.basename(x) for x in glob.glob(validpath)]
#print(src_train_masks)

train_folder_bd = os.path.join(DATA_PATH, "train_folder_bd/")

valid_folder_bd = os.path.join(DATA_PATH, "valid_folder_bd/")

## create train boundaries 

for filename in src_train_masks:
    #print(filename)
    master_img_gt = cv.imread(os.path.join(src_train_folder_gt, filename))
    boundaries = find_boundaries(master_img_gt, mode = 'thick')
    boundaries = boundaries.astype("uint8")
    boundaries*= 255
    
    img_gt_fname = '{}.{}'.format(filename[:-4], 'png')

    cv.imwrite(os.path.join(train_folder_bd, img_gt_fname), boundaries)

#create validation boundaries 

for filename in src_valid_masks:
    #print(filename)
    master_img_gt = cv.imread(os.path.join(src_valid_folder_gt, filename))
    boundaries = find_boundaries(master_img_gt, mode = 'thick')
    boundaries = boundaries.astype("uint8")
    boundaries*= 255
    
    img_gt_fname = '{}.{}'.format(filename[:-4], 'png')

    cv.imwrite(os.path.join(valid_folder_bd, img_gt_fname), boundaries)
    
