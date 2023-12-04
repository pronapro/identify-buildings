import numpy as np
import cv2 as cv
import shutil
import math
import os
import glob
import time
from skimage.morphology import dilation
from skimage.morphology import square
from skimage.segmentation import find_boundaries

import skimage
from scipy.ndimage.morphology import distance_transform_bf
from numpy import asarray
import numpy as np 
import tensorflow as tf

def transform_one(img, threshold=20, bins=10):
    img = tf.cast(img, dtype=tf.uint8)
    u = tf.one_hot(img, 2)
    u = tf.squeeze(u, axis=-2)  # Squeeze the second-last dimension
    distance_build = distance_transform_bf(u[..., 1], sampling=2)
    distance_background = distance_transform_bf(u[..., 0], sampling=2)
    distance_build = np.minimum(distance_build, threshold * (distance_build > 0))
    distance_background = np.minimum(distance_background, threshold * (distance_background > 0))
    distance = (distance_build - distance_background)
    distance = (distance - np.amin(distance)) / (np.amax(distance) - np.amin(distance) + 1e-50) * (bins - 1)
    z = tf.one_hot(distance, bins)
    return z

def transform_fn(img, threshold=20, bins=10):
    img_shape = tf.shape(img)
    img = tf.numpy_function(transform_one, [img, threshold, bins], tf.float32)
    img.set_shape([None, None, bins])
    return img




def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [img_rows, img_cols])
    img = img / 255.0
    return img

def load_mask(mask_path):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [label_rows, label_cols])
    mask = mask / 255.0
    return mask

def create_distance_maps(mask,threshold =30, bins =10):
    sample_mask= transform_fn(mask, threshold=threshold, bins=bins)
    sample_mask = np.array(sample_mask)
    for i in range(bins):
        sample_mask[:,:,i] = sample_mask[:,:,i]*(i+1)
    to_add_masks = []
    for i in range(bins):
        to_add_masks.append(sample_mask[:,:,i])
    #print(np.unique(sample_mask))

    distance_map = np.sum(to_add_masks, axis=0)
    #print(np.unique(distance_map))
    distance_map -= 1
    #print(np.unique(distance_map))
    final_distance_map = (distance_map * 255 / distance_map.max()).astype(np.uint8)
    return final_distance_map
## create directory
INPUT_PATH = os.getcwd()
DATA_PATH = INPUT_PATH

img_rows, img_cols = 512, 512
label_rows, label_cols = 512, 512

src_train_folder_gt = os.path.join(DATA_PATH, "train_folder_gt/")
maskpath = '{}*.png'.format(src_train_folder_gt)
src_train_masks  = [os.path.basename(x) for x in glob.glob(maskpath)]


train_folder_msk =os.path.join(DATA_PATH, "train_folder_distmap/")
start = time.time()

for filename in src_train_masks:
    img_msk_fname = '{}.{}'.format(filename[:-4],'png')
    img = load_mask(os.path.join(src_train_folder_gt, filename))
    sample_mask_dist= create_distance_maps(img, threshold=20, bins=10)
    cv.imwrite(os.path.join(train_folder_msk, img_msk_fname), sample_mask_dist)
end = time.time()
total =end - start
#print(total)