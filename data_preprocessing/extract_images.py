import numpy as np
import cv2 as cv
import shutil
import math
import os
import glob
import re

## set cropping parameters 
## set parameters for patch extraction
test = False
master_size = 5000
image_size = 512
overlap = 0.0
count = math.ceil((master_size - image_size * overlap) / (image_size * (1 - overlap)))
step = (master_size - image_size * overlap) / count
print('count =', count, ', step =', step)

## utility functions 
def get_area(file):
    file_name = os.path.splitext(file)[0]
    area = re.sub("[^a-zA-Z]+", "",file_name)
    return area

def get_number(file):
    file_name = os.path.splitext(file)[0]
    number = int(''.join(filter(str.isdigit, file_name)))
    return number

## return train images and validation images
def train_valid_split(image_list):
    validation_list = []
    training_list =[]
    
    for i in range(len(image_list)):
        if get_area(image_list[i]) == 'austin':
            if get_number(image_list[i]) <=5:
                validation_list.append(image_list[i])
            else:
                training_list.append(image_list[i])
                
        elif get_area(image_list[i]) == 'chicago':
            if get_number(image_list[i]) <=5:
                validation_list.append(image_list[i])
            else:
                training_list.append(image_list[i])            

        elif get_area(image_list[i]) == 'kitsap':
            if get_number(image_list[i]) <=5:
                validation_list.append(image_list[i])
            else:
                training_list.append(image_list[i])
                
        elif get_area(image_list[i]) == 'tyrolw':
            if get_number(image_list[i]) <=5:
                validation_list.append(image_list[i])
            else:
                training_list.append(image_list[i])

        elif get_area(image_list[i]) == 'vienna':
            if get_number(image_list[i]) <=5:
                validation_list.append(image_list[i])
            else:
                training_list.append(image_list[i])

    return validation_list, training_list

## create directory
INPUT_PATH = os.getcwd()
DATA_PATH = INPUT_PATH

#src_train_images = 
src_train_folder = os.path.join(DATA_PATH, "train/images/")
src_train_folder_gt = os.path.join(DATA_PATH, "train/gt/")
imagepath = '{}*.tif'.format(src_train_folder)
src_train_images  = [os.path.basename(x) for x in glob.glob(imagepath)]
print(src_train_images)
valid_images, train_images = train_valid_split(src_train_images)

train_folder = os.path.join(DATA_PATH, "train_folder/")
train_folder_gt =os.path.join(DATA_PATH, "train_folder_gt/")
validation_folder = os.path.join(DATA_PATH, "valid_folder/")
validation_folder_gt = os.path.join(DATA_PATH, "valid_folder_gt/")

# validation data , first five image 


#train dataset
for filename in train_images:
    print(filename)
    master_img = cv.imread(os.path.join(src_train_folder, filename))
    master_img_gt = cv.imread(os.path.join(src_train_folder_gt, filename))

    for i in range(count):
        if i < count - 1:
            y = round(i * step)
        else:
            y = master_size - image_size

        for j in range(count):
            if j < count - 1:
                x = round(j * step)
            else:
                x = master_size - image_size

            img = master_img[y:y+image_size, x:x+image_size]
            img_gt = master_img_gt[y:y+image_size, x:x+image_size]

            img_fname = '{}_{}_{}.{}'.format(filename[:-4], i, j, 'jpg')
            img_gt_fname = '{}_{}_{}.{}'.format(filename[:-4], i, j, 'png')
            cv.imwrite(os.path.join(train_folder, img_fname), img)
            cv.imwrite(os.path.join(train_folder_gt, img_gt_fname), img_gt)

## validation set 
for filename in valid_images:
    print(filename)
    master_img = cv.imread(os.path.join(src_train_folder, filename))
    master_img_gt = cv.imread(os.path.join(src_train_folder_gt, filename))

    for i in range(count):
        if i < count - 1:
            y = round(i * step)
        else:
            y = master_size - image_size

        for j in range(count):
            if j < count - 1:
                x = round(j * step)
            else:
                x = master_size - image_size

            img = master_img[y:y+image_size, x:x+image_size]
            img_gt = master_img_gt[y:y+image_size, x:x+image_size]

            img_fname = '{}_{}_{}.{}'.format(filename[:-4], i, j, 'jpg')
            img_gt_fname = '{}_{}_{}.{}'.format(filename[:-4], i, j, 'png')
            cv.imwrite(os.path.join(validation_folder, img_fname), img)
            cv.imwrite(os.path.join(validation_folder_gt, img_gt_fname), img_gt)
            